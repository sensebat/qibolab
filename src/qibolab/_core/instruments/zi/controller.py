"""Zurich Instruments controller, built on LabOne Q."""

import re
from contextlib import ExitStack
from dataclasses import dataclass
from functools import reduce
from typing import Literal

import laboneq.simple as lq
import numpy as np
from laboneq.dsl.calibration import (
    ExponentialCompensation,
    FIRCompensation,
    Precompensation,
)
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQA, SHFQC, SHFSG
from laboneq.simple import (
    Calibration,
    Experiment,
    ExperimentSignal,
    ModulationType,
    Oscillator,
    Session,
    SignalCalibration,
)
from pydantic import Field

from qibolab._core.components import (
    AcquisitionChannel,
    Channel,
    Config,
    DcChannel,
    ExponentialFilter,
    FiniteImpulseResponseFilter,
    IqChannel,
)
from qibolab._core.components.filters import Filter
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import Pulse, PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from . import pulse, results, sweep

__all__ = ["ZiController", "ZiInstrument"]

SAMPLING_RATE = 2.0
"""Sampling rate in GSps of SHFSG and SHFQA, and of an HDAWG set up with them."""

HDAWG_SAMPLING_RATE = 2.4
"""Sampling rate in GSps of an HDAWG in a setup without SHF instruments."""

PQSC_UID = "pqsc"
"""Uid of the PQSC in the LabOne Q device setup."""

InstrumentKind = Literal["HDAWG", "SHFSG", "SHFQA", "SHFQC"]

_AVERAGING = {
    AveragingMode.CYCLIC: lq.AveragingMode.CYCLIC,
    AveragingMode.SINGLESHOT: lq.AveragingMode.SINGLE_SHOT,
    AveragingMode.SEQUENTIAL: lq.AveragingMode.SEQUENTIAL,
}

_ACQUISITION = {
    AcquisitionType.INTEGRATION: lq.AcquisitionType.INTEGRATION,
    AcquisitionType.DISCRIMINATION: lq.AcquisitionType.DISCRIMINATION,
    AcquisitionType.RAW: lq.AcquisitionType.RAW,
}

_INSTRUMENT_CLASSES = {"HDAWG": HDAWG, "SHFSG": SHFSG, "SHFQA": SHFQA, "SHFQC": SHFQC}


@dataclass(frozen=True)
class ZiInstrument:
    """User-facing description of one instrument of the rack."""

    kind: InstrumentKind
    """Instrument type."""
    serial: str
    """Device serial, e.g. ``DEV12345``.

    The LabOne data server identifies its instruments by serial.
    """
    options: str | None = None
    """Installed device options, e.g. ``SHFSG8/RTR``.

    The first option names the model (``SHFSG8``, ``SHFQA2``, ``HDAWG4``, ...), which
    fixes the number of ports and, on an SHFSG8, the synthesizer pairs. On an SHFQC
    the model is the option giving the number of SG channels (``QC2CH``, ``QC4CH``,
    ``QC6CH``), e.g. ``SHFQC/QC6CH``.
    """

    @property
    def model(self) -> str | None:
        """Model named in :attr:`options`, e.g. ``SHFSG8``, if any."""
        if self.options is None:
            return None
        for option in self.options.split("/"):
            pattern = r"QC\dCH" if self.kind == "SHFQC" else rf"{self.kind}\d"
            if re.fullmatch(pattern, option):
                return option
        return None

    @property
    def nports(self) -> int:
        """Number of ports, numbered from 0.

        It follows from :attr:`model`, as in LabOne Q: the number in the model name,
        or 8 on an SHFSG4 or SHFSG2 with the ``RTR`` option, whose internal channels
        are addressed as further ports. Without a model, it is the largest one of the
        kind. On an SHFQC these are the SG ports; its QA port is 0.
        """
        model = self.model
        if model is None or self.options is None:
            return _MAX_PORTS[self.kind]
        if self.kind == "SHFSG" and "RTR" in self.options.split("/"):
            return 8
        if self.kind == "SHFQC":
            return int(model[2])
        return int(model[-1])


_MAX_PORTS: dict[InstrumentKind, int] = {"HDAWG": 8, "SHFSG": 8, "SHFQA": 4, "SHFQC": 6}
"""Number of ports of the largest model of each instrument kind."""


def _check_port(name: str, instrument: ZiInstrument, port: int) -> None:
    """Check that ``port`` exists on the instrument called ``name``."""
    if not 0 <= port < instrument.nports:
        raise ValueError(
            f"Port {port} does not exist on '{name}' "
            f"({instrument.model or instrument.kind}, "
            f"ports 0-{instrument.nports - 1})."
        )


def _logical_signal(id: ChannelId) -> ChannelId:
    """Check that a channel id can be used as a LabOne Q logical signal path.

    LabOne Q logical signals are addressed as ``group/name``, which is the form of the
    :meth:`qibolab.Qubit.default` channel ids (e.g. ``q1/drive``).
    """
    parts = id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Channel id '{id}' is not of the form 'group/name', "
            "which is required to use it as a LabOne Q logical signal."
        )
    return id


_PORTS: dict[tuple[type[Channel], InstrumentKind], str] = {
    (IqChannel, "SHFSG"): "SGCHANNELS/{}/OUTPUT",
    (IqChannel, "SHFQA"): "QACHANNELS/{}/OUTPUT",
    (AcquisitionChannel, "SHFQA"): "QACHANNELS/{}/INPUT",
    (DcChannel, "HDAWG"): "SIGOUTS/{}",
    (IqChannel, "SHFQC"): "SGCHANNELS/{}/OUTPUT",
    (AcquisitionChannel, "SHFQC"): "QACHANNELS/{}/INPUT",
}
"""LabOne Q port templates, by channel type and instrument kind.

On an SHFQC, a probe channel goes to the QA output instead (see :func:`_port`).
"""


def _probes(channels: dict[ChannelId, Channel]) -> set[ChannelId]:
    """Probe channels, i.e. the ones referenced by an acquisition channel."""
    return {
        ch.probe
        for ch in channels.values()
        if isinstance(ch, AcquisitionChannel) and ch.probe is not None
    }


def _port(
    id: ChannelId, channel: Channel, instrument: ZiInstrument, probe: bool = False
) -> str:
    """LabOne Q port of a channel, from its type and the instrument kind.

    On an SHFQC, the ``probe`` channel and the acquisition channel are on the QA
    channel, port 0; the other IQ channels on the SG channels.
    """
    template = _PORTS.get((type(channel), instrument.kind))
    if template is None:
        raise ValueError(
            f"Channel '{id}' of type {type(channel).__name__} "
            f"cannot be connected to a {instrument.kind}."
        )
    assert channel.device is not None
    if instrument.kind == "SHFQC" and probe:
        template = "QACHANNELS/{}/OUTPUT"
    qa = instrument.kind == "SHFQC" and template.startswith("QACHANNELS")
    if not qa:
        _check_port(channel.device, instrument, channel.port)
    elif channel.port != 0:
        raise ValueError(
            f"Port {channel.port} does not exist on the QA channel of "
            f"'{channel.device}' (SHFQC, port 0)."
        )
    return template.format(channel.port)


def device_setup(
    address: str,
    instruments: dict[str, ZiInstrument],
    pqsc: str | None,
    channels: dict[ChannelId, Channel],
) -> DeviceSetup:
    """Build the LabOne Q device setup from the qibolab channels.

    The data server is reached at ``address``, of the form ``IP:port``. Each
    instrument is added under its name as uid. Each channel becomes a logical signal,
    with the channel id as path, connected to the port given by
    :attr:`Channel.device` and :attr:`Channel.path`.
    """
    host, port = address.split(":")
    setup = DeviceSetup()
    setup.add_dataserver(host=host, port=int(port))

    for name, instrument in instruments.items():
        cls = _INSTRUMENT_CLASSES[instrument.kind]
        setup.add_instruments(
            cls(uid=name, address=instrument.serial, device_options=instrument.options)
        )
    if pqsc is not None:
        setup.add_instruments(PQSC(uid=PQSC_UID, address=pqsc))

    probes = _probes(channels)
    for id, channel in channels.items():
        if channel.device not in instruments:
            raise ValueError(
                f"Channel '{id}' refers to the unknown instrument '{channel.device}'."
            )
        instrument = instruments[channel.device]
        setup.add_connections(
            channel.device,
            create_connection(
                to_signal=_logical_signal(id),
                ports=_port(id, channel, instrument, id in probes),
            ),
        )
    return setup


def signals(
    channels: dict[ChannelId, Channel], sequences: list[PulseSequence]
) -> list[ChannelId]:
    """Channels to be mapped to the signals of an experiment.

    Other than the channels used in ``sequences``, they include the probe channels of
    the used acquisition ones, and all the DC channels: the offset of a DC channel
    only reaches the instrument if its signal is part of the experiment.
    """
    used = {id for sequence in sequences for id in sequence.channels}
    probes = {channels[id].iqout(id) for id in used} - {None}
    dc = {id for id, ch in channels.items() if isinstance(ch, DcChannel)}
    return sorted(used | probes | dc)


def precompensation(
    filters: list[Filter], sampling_rate: float
) -> Precompensation | None:
    """HDAWG precompensation implementing the qibolab filters of a DC channel.

    An :class:`ExponentialFilter` becomes an exponential compensation, its ``tau`` in
    samples converted to s at ``sampling_rate`` (GSps). The coefficients of all the
    :class:`FiniteImpulseResponseFilter` are convolved into one FIR compensation.
    The instrument needs the ``PC`` option.
    """
    if len(filters) == 0:
        return None
    exponential = [
        ExponentialCompensation(
            timeconstant=f.tau / sampling_rate * pulse.NS, amplitude=f.amplitude
        )
        for f in filters
        if isinstance(f, ExponentialFilter)
    ]
    fir = [
        f.coefficients for f in filters if isinstance(f, FiniteImpulseResponseFilter)
    ]
    return Precompensation(
        exponential=exponential or None,
        FIR=FIRCompensation(coefficients=reduce(np.convolve, fir)) if fir else None,
    )


def calibration(
    ids: list[ChannelId],
    channels: dict[ChannelId, Channel],
    instruments: dict[str, ZiInstrument],
    configs: dict[str, Config],
    sampling_rate: float = SAMPLING_RATE,
    discriminate: bool = False,
) -> Calibration:
    """Build the LabOne Q calibration of the channels ``ids`` from their configs.

    Channels with the same LO name share the LO. An acquisition channel shares the
    oscillator of its probe channel, which is software-modulated (on an SHFQA, or on
    the QA channel of an SHFQC). Filters of a DC channel become its
    precompensation, see :func:`precompensation`.

    With ``discriminate``, each acquisition channel gets the opposite of the
    ``threshold`` of its config, to go with the negated kernel of
    :func:`.pulse.kernel`.
    """
    los: dict[str, Oscillator] = {}
    oscillators: dict[ChannelId, Oscillator] = {}
    probes = _probes(channels)

    def lo(id: ChannelId) -> Oscillator:
        name = channels[id].lo
        if name is None:
            raise ValueError(f"Channel '{id}' has no LO.")
        return los.setdefault(name, Oscillator(frequency=configs[name].frequency))

    def oscillator(id: ChannelId) -> Oscillator:
        kind = instruments[channels[id].device].kind
        software = kind == "SHFQA" or (kind == "SHFQC" and id in probes)
        return oscillators.setdefault(
            id,
            Oscillator(
                frequency=configs[id].frequency - lo(id).frequency,
                modulation_type=ModulationType.SOFTWARE
                if software
                else ModulationType.HARDWARE,
            ),
        )

    calib = Calibration()
    for id in ids:
        channel, config = channels[id], configs[id]
        if isinstance(channel, DcChannel):
            calib[id] = SignalCalibration(
                voltage_offset=config.offset,
                range=config.power_range,
                precompensation=precompensation(config.filters, sampling_rate),
            )
        elif isinstance(channel, AcquisitionChannel):
            calib[id] = SignalCalibration(
                oscillator=oscillator(channel.probe),
                port_delay=config.delay * pulse.NS,
                range=config.power_range,
            )
            if discriminate:
                if config.threshold is None:
                    raise ValueError(f"Channel '{id}' has no threshold.")
                calib[id].threshold = -config.threshold
        else:
            calib[id] = SignalCalibration(
                oscillator=oscillator(id),
                local_oscillator=lo(id),
                range=config.power_range,
            )
    return calib


class ZiController(Controller):
    """Controller for a Zurich Instruments rack, driven through LabOne Q.

    The qibolab channels are the source of truth for the wiring: each channel has the
    instrument name as :attr:`Channel.device` and the 0-based port number as
    :attr:`Channel.path`. The LabOne Q device setup is built from them on
    :meth:`connect`.
    """

    address: str
    """IP address and port of the LabOne data server.

    Has the form XXX.XXX.XXX.XXX:XXX.
    """
    instruments: dict[str, ZiInstrument] = Field(default_factory=dict)
    """Instruments carrying the channels, by name."""
    pqsc: str | None = None
    """Serial of the PQSC synchronizing the instruments, if any."""
    emulation: bool = False
    """Run on the LabOne Q emulator, without hardware."""
    keep_dc_offsets_on: bool = True
    """Keep the DC offsets on after disconnecting.

    With ``False``, :meth:`disconnect` sets the offset of every DC channel to 0.
    """
    reset_pulses: dict[ChannelId, tuple[ChannelId, Pulse]] = Field(default_factory=dict)
    """Fast reset pulse of each acquisition channel, with the channel to play it on.

    Usually the ``RX`` native on the qubit drive. With
    :attr:`ExecutionParameters.fast_reset`, it is played after each acquisition of
    the channel whose state is 1.
    """

    _session: Session | None = None

    @property
    def sampling_rate(self) -> float:
        """Sampling rate of the instruments in GSps.

        As LabOne Q does, all the instruments run at the SHF rate when any SHF
        instrument is present; HDAWGs on their own run at their native rate.
        """
        if all(i.kind == "HDAWG" for i in self.instruments.values()):
            return HDAWG_SAMPLING_RATE
        return SAMPLING_RATE

    @property
    def is_connected(self) -> bool:
        """Determine connection status."""
        return self._session is not None

    @property
    def session(self) -> Session:
        """LabOne Q session, available while connected."""
        assert self._session is not None
        return self._session

    def connect(self) -> None:
        """Build the device setup and connect to the instruments."""
        if self.is_connected:
            return
        setup = device_setup(self.address, self.instruments, self.pqsc, self.channels)
        session = Session(setup)
        session.connect(do_emulation=self.emulation)
        self._session = session

    def disconnect(self) -> None:
        """Disconnect from the instruments.

        DC offsets are zeroed first, unless :attr:`keep_dc_offsets_on` is set.
        """
        if not self.is_connected:
            return
        if not self.keep_dc_offsets_on:
            self._zero_dc_offsets()
        self.session.disconnect()
        self._session = None

    def _experiment(self, ids: list[ChannelId], calib: Calibration) -> Experiment:
        """Experiment with the channels ``ids`` as signals, calibrated by ``calib``."""
        experiment = Experiment(signals=[ExperimentSignal(id) for id in ids])
        for id in ids:
            experiment.map_signal(
                id, self.session.device_setup.logical_signal_by_uid(id)
            )
        experiment.set_calibration(calib)
        return experiment

    def _zero_dc_offsets(self) -> None:
        """Set the offset of every DC channel to 0.

        A mapped signal's ``voltage_offset`` is applied when the experiment is
        initialized, so an experiment with an empty real-time loop is enough.
        """
        flux = [id for id, ch in self.channels.items() if isinstance(ch, DcChannel)]
        if len(flux) == 0:
            return
        calib = Calibration()
        for id in flux:
            calib[id] = SignalCalibration(voltage_offset=0.0)
        experiment = self._experiment(flux, calib)
        with experiment.acquire_loop_rt(count=1):
            pass
        self.session.run(self.session.compile(experiment))

    def _check_fast_reset(
        self, sequences: list[PulseSequence], options: ExecutionParameters
    ) -> list[ChannelId]:
        """Check that fast reset can run, and return the channels of its pulses."""
        if options.acquisition_type is not AcquisitionType.DISCRIMINATION:
            raise NotImplementedError(
                "Fast reset needs DISCRIMINATION: its state decision uses the "
                "threshold and the kernel of the discrimination."
            )
        acquired = {id for sequence in sequences for id, _ in sequence.acquisitions}
        missing = acquired - self.reset_pulses.keys()
        if missing:
            raise ValueError(f"No fast reset pulse for {sorted(missing)}.")
        return [self.reset_pulses[id][0] for id in acquired]

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Execute the sequences in a single LabOne Q experiment.

        Each sequence is followed by the relaxation time. Real-time sweeps run inside
        the acquisition loop, near-time ones around it; see :func:`.sweep._group`
        for which is which.
        All acquisition types and averaging modes are supported, except sequential
        averaging of raw traces, which LabOne Q refuses. Discrimination uses
        the ``threshold`` and ``iq_angle`` of the acquisition configs. Raw traces are
        demodulated at the configured probe frequency, and only one acquisition per
        call is taken: LabOne Q allows a single raw acquire in the averaging loop.

        Fast reset needs discrimination: after each sequence, the pulse in
        :attr:`reset_pulses` of each acquired channel is played if its state is 1
        (LabOne Q ``match``, feedback through the PQSC or within an SHFQC).
        """
        acquisitions = [acq for sequence in sequences for acq in sequence.acquisitions]
        raw = options.acquisition_type is AcquisitionType.RAW
        if raw and len(acquisitions) > 1:
            raise NotImplementedError(
                f"RAW acquisition takes one acquisition per call, not "
                f"{len(acquisitions)}: LabOne Q allows a single raw acquire in the "
                "averaging loop."
            )
        reset = self._check_fast_reset(sequences, options) if options.fast_reset else []
        ids = sorted(set(signals(self.channels, sequences)) | set(reset))
        discriminate = options.acquisition_type is AcquisitionType.DISCRIMINATION
        calib = calibration(
            ids,
            self.channels,
            self.instruments,
            configs,
            self.sampling_rate,
            discriminate,
        )
        experiment = self._experiment(ids, calib)
        swept: sweep.Swept = {}
        slots: pulse.Slots = {}
        near, real, axes = sweep.split(
            sweepers, self.channels, configs, calib, swept, self.sampling_rate
        )
        windows = pulse.windows(sequences, self.channels)
        layouts = [
            pulse.layout(sequence, self.channels, self.sampling_rate, windows, swept)
            for sequence in sequences
        ]
        pulse.check_lines(layouts, self.channels)
        with ExitStack() as stack:
            for group in near:
                stack.enter_context(experiment.sweep(parameter=group))
            stack.enter_context(
                experiment.acquire_loop_rt(
                    count=options.nshots,
                    averaging_mode=_AVERAGING[options.averaging_mode],
                    acquisition_type=_ACQUISITION[options.acquisition_type],
                )
            )
            for group in real:
                stack.enter_context(experiment.sweep(parameter=group))
            for sequence, blocks in zip(sequences, layouts):
                with experiment.section():
                    pulse.play(
                        experiment,
                        blocks,
                        self.channels,
                        configs,
                        self.sampling_rate,
                        swept,
                        slots,
                        options.acquisition_type,
                    )
                for id, acquisition in sequence.acquisitions if reset else []:
                    with experiment.match(handle=str(acquisition.id)):
                        with experiment.case(state=0):
                            pass
                        with experiment.case(state=1):
                            channel, reset_pulse = self.reset_pulses[id]
                            pulse.run(
                                experiment,
                                channel,
                                [reset_pulse],
                                self.channels,
                                self.sampling_rate,
                                {},
                            )
                with experiment.section():
                    for id in experiment.signals:
                        experiment.delay(id, time=options.relaxation_time * pulse.NS)

        data = self.session.run(self.session.compile(experiment))
        parsed = {}
        for id, acquisition in acquisitions:
            result = results.parse(data.get_data(str(acquisition.id)), options, axes)
            if raw:
                probe = self.channels[id].iqout(id)
                frequency = (
                    configs[probe].frequency
                    - configs[self.channels[probe].lo].frequency
                )
                result = results.demodulated(
                    result, frequency * 1e-9, self.sampling_rate
                )
            parsed[acquisition.id] = result
        return parsed
