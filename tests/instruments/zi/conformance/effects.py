"""Read what ``ZiController.play`` hands to LabOne Q, in qibolab terms.

The effect tests check what a qibolab call makes LabOne Q do, not how the driver
builds the experiment. :func:`play` runs ``ZiController.play`` with the session's
``compile`` and ``run`` wrapped, keeps the DSL ``Experiment``, the device setup and the
LabOne Q results, and returns a :class:`Capture` to query them by qibolab meaning:
operations on a qibolab channel, amplitudes, phases, waveforms, frequencies, sweeps,
acquisitions, the acquisition loop and the waits after readouts.

Only public LabOne Q objects are read (``laboneq.simple`` / ``laboneq.dsl``:
``Experiment``, ``Section``, ``Sweep``, ``AcquireLoopRt``, ``PlayPulse``, ``Acquire``,
``Delay``, ``SignalCalibration``, ``Oscillator``, ``SweepParameter``, ``DeviceSetup``
and ``Results.get_data``), with the fields they have in both laboneq 26.4 and 26.10.
Section names, uids, nesting and the order of operations are never read, except the
order of operations on one signal, which LabOne Q plays in order.

Signal to qibolab channel. An experiment signal reaches a qibolab channel through
the port of its logical signal: the instrument uid and port number of the device setup
connection (``SGCHANNELS/0/OUTPUT``, ``QACHANNELS/0/INPUT``, ``SIGOUTS/3``) must equal
``Channel.device`` and ``Channel.port``, with the direction of the channel (input for an
acquisition channel, output otherwise). When several qibolab channels share a port
(multiplexed probes and acquisitions on one SHFQA channel), the logical signal uid
picks among them: ``ZiController`` documents that each channel becomes a logical signal
with the channel id as uid. Such channels are then told apart by their frequency.

Units are qibolab ones: ns, Hz, rad.

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q), and the spy on ``session.run`` in
ElStabilini/qibolab@zi_drivers (e8757e58) tests/instruments/zi/test_controller.py.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from laboneq.core.utilities.pulse_sampler import sample_pulse
from laboneq.dsl.experiment import (
    Acquire,
    AcquireLoopRt,
    Delay,
    PlayPulse,
    PulseFunctional,
    PulseSampled,
    Section,
    Sweep,
)
from laboneq.simple import Experiment

from qibolab._core.components import AcquisitionChannel
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Delay as Wait
from qibolab._core.pulses import Pulse, Rectangular
from qibolab._core.sequence import PulseSequence

__all__ = [
    "Capture",
    "Driven",
    "Missing",
    "Op",
    "SweepLoop",
    "capture",
    "drives_then_mz",
    "iq",
    "play",
    "square",
    "values",
]

NS = 1e-9
"""LabOne Q times are in s, qibolab ones in ns."""


class Missing(AssertionError):
    """An expected effect is not there, e.g. no signal reaches a qibolab channel.

    An ``AssertionError``, so that an injected fault that removes the effect counts as
    caught. Problems of the capture itself raise ``RuntimeError``.
    """


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise Missing(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


_PORT = re.compile(
    r"^(?:SGCHANNELS|QACHANNELS)/(\d+)/(?:OUTPUT|INPUT)$|^SIGOUTS/(\d+)$"
)


def values(x: Any) -> Any:
    """Number or array of a LabOne Q value: a float, or a sweep parameter's values."""
    if x is None:
        return None
    if hasattr(x, "values"):
        return np.asarray(x.values)
    return x


def _drives(parameter: Any, x: Any) -> bool:
    """Whether ``x`` is ``parameter`` or a sweep parameter derived from it."""
    if x is None or not hasattr(x, "uid"):
        return False
    if x is parameter or x.uid == parameter.uid:
        return True
    return any(_drives(parameter, p) for p in getattr(x, "driven_by", None) or [])


@dataclass(frozen=True)
class Op:
    """One LabOne Q operation, with the qibolab channel its signal reaches."""

    op: PlayPulse | Acquire | Delay
    """The LabOne Q operation."""
    signal: str
    """Experiment signal uid."""
    channel: ChannelId | None
    """Qibolab channel reached by the signal, ``None`` if none."""
    real_time: bool
    """Inside the acquisition loop."""


@dataclass(frozen=True)
class SweepLoop:
    """One LabOne Q sweep, outer loops first."""

    parameters: list
    """The swept parameters."""
    real_time: bool
    """Inside the acquisition loop."""

    @property
    def values(self) -> list[np.ndarray]:
        """Values of each parameter, in sweep order."""
        return [np.asarray(p.values) for p in self.parameters]


@dataclass(frozen=True)
class Driven:
    """A quantity driven by a sweep parameter."""

    channel: ChannelId | None
    """Qibolab channel the quantity belongs to."""
    quantity: str
    """``amplitude``, ``phase``, ``length`` or ``time`` of an operation;
    ``frequency``, ``lo``, ``offset`` or ``calibration amplitude`` of a channel."""
    op: Op | None = None
    """The operation, for an operation quantity."""


@dataclass
class Capture:
    """What one ``ZiController.play`` call handed to LabOne Q, and what it got back."""

    controller: Any
    """The ``ZiController``."""
    experiments: list[Experiment] = field(default_factory=list)
    """Experiments compiled during the call, in order."""
    raw: Any = None
    """LabOne Q ``Results`` of the last run."""
    results: dict | None = None
    """Qibolab results returned by ``play``."""

    # experiment

    @property
    def experiment(self) -> Experiment:
        """The experiment of the call (exactly one compiled)."""
        _expect(len(self.experiments) == 1, f"{len(self.experiments)} experiments")
        return self.experiments[0]

    @property
    def setup(self):
        """LabOne Q device setup of the session."""
        return self.controller.session.device_setup

    def _calibration(self, signal: str):
        """Signal calibration: from the experiment, else from the logical signal."""
        exp = self.experiment
        calib = exp.signals[signal].calibration
        if calib is None:
            calib = exp.get_calibration().calibration_items.get(signal)
        if calib is None:
            path = exp.signals[signal].mapped_logical_signal_path
            calib = self.logical_signal(path).calibration
        return calib

    def logical_signal(self, path: str):
        """Logical signal of the device setup at ``path``."""
        for group in self.setup.logical_signal_groups.values():
            for ls in group.logical_signals.values():
                if ls.path == path:
                    return ls
        raise KeyError(f"No logical signal at '{path}'.")

    def calibration(self, channel: ChannelId):
        """LabOne Q signal calibration of the channel's signal."""
        return self._calibration(self.signal_of(channel))

    def port_delay(self, channel: ChannelId) -> float:
        """Port delay of the channel's signal in ns, 0 when not set.

        LabOne Q shifts every operation of a signal by its port delay.
        """
        delay = values(self.calibration(channel).port_delay)
        return 0.0 if delay is None else delay / NS

    # signal <-> channel

    def port(self, signal: str) -> tuple[str, int, bool]:
        """Instrument uid, port number and input flag the signal is connected to."""
        path = self.experiment.signals[signal].mapped_logical_signal_path
        _require(path is not None, f"Signal '{signal}' is not mapped.")
        found = [
            (instrument.uid, connection.local_port)
            for instrument in self.setup.instruments
            for connection in instrument.connections
            if connection.remote_path == path
        ]
        _require(len(found) == 1, f"Signal '{signal}' ({path}) has ports {found}.")
        uid, port = found[0]
        match = _PORT.match(port)
        _require(match is not None, f"Unknown LabOne Q port '{port}'.")
        number = int(match.group(1) or match.group(2))
        return uid, number, port.endswith("INPUT")

    def channel_of(self, signal: str) -> ChannelId | None:
        """Qibolab channel reached by an experiment signal (see the module doc)."""
        uid, number, is_input = self.port(signal)
        candidates = [
            id
            for id, ch in self.controller.channels.items()
            if ch.device == uid
            and ch.port == number
            and isinstance(ch, AcquisitionChannel) == is_input
        ]
        if len(candidates) == 1:
            return candidates[0]
        path = self.experiment.signals[signal].mapped_logical_signal_path
        name = self.logical_signal(path).uid
        return name if name in candidates else None

    def signals_of(self, channel: ChannelId) -> list[str]:
        """Experiment signals reaching a qibolab channel."""
        return [s for s in self.experiment.signals if self.channel_of(s) == channel]

    def signal_of(self, channel: ChannelId) -> str:
        """The one experiment signal reaching a qibolab channel."""
        signals = self.signals_of(channel)
        _expect(len(signals) == 1, f"Channel '{channel}' has signals {signals}.")
        return signals[0]

    # operations

    def _walk(self, node, real_time: bool) -> Iterator[Op]:
        if isinstance(node, Section):
            inside = real_time or isinstance(node, AcquireLoopRt)
            for child in node.children:
                yield from self._walk(child, inside)
        elif isinstance(node, PlayPulse | Acquire | Delay):
            yield Op(node, node.signal, self.channel_of(node.signal), real_time)

    def ops(
        self, channel: ChannelId | None = None, kind: type | None = None
    ) -> list[Op]:
        """Operations, in order, optionally only on ``channel`` and of type ``kind``.

        The order is only meaningful among operations on one signal.
        """
        ops = [op for s in self.experiment.sections for op in self._walk(s, False)]
        return [
            op
            for op in ops
            if (channel is None or op.channel == channel)
            and (kind is None or isinstance(op.op, kind))
        ]

    def plays(self, channel: ChannelId | None = None) -> list[Op]:
        """Pulses played, optionally only on ``channel``."""
        return self.ops(channel, PlayPulse)

    def acquires(self, channel: ChannelId | None = None) -> list[Op]:
        """Acquisitions, optionally only on ``channel``."""
        return self.ops(channel, Acquire)

    def delays(self, channel: ChannelId | None = None) -> list[Op]:
        """Delays, optionally only on ``channel``."""
        return self.ops(channel, Delay)

    # played pulses

    def amplitude(self, op: Op):
        """Amplitude played: operation x pulse x signal calibration amplitude.

        A float, or an array with one value per sweep step when swept.
        """
        amplitude = values(op.op.amplitude)
        amplitude = 1.0 if amplitude is None else amplitude
        pulse = getattr(op.op.pulse, "amplitude", None)
        calib = getattr(self._calibration(op.signal), "amplitude", None)
        for factor in (values(pulse), values(calib)):
            if factor is not None:
                amplitude = amplitude * factor
        return amplitude

    def phase(self, op: Op):
        """Phase of the played pulse, in rad (0 when not given)."""
        phase = values(op.op.phase)
        return 0.0 if phase is None else phase

    def samples(self, op: Op) -> np.ndarray:
        """Complex samples of the played or integration pulse, before amplitude.

        A functional pulse is sampled as LabOne Q samples it (``sample_pulse``), at
        the length of the operation (else of the pulse), with the pulse parameters of
        the pulse and of the operation; a swept length has no single set of samples.
        """
        pulse = op.op.kernel if isinstance(op.op, Acquire) else op.op.pulse
        if isinstance(pulse, PulseFunctional):
            return self._functional_samples(op, pulse)
        _require(isinstance(pulse, PulseSampled), f"{type(pulse).__name__} not sampled")
        samples = np.asarray(pulse.samples)
        if samples.ndim == 2:  # [i, q] rows
            samples = samples[0] + 1j * samples[1]
        return samples.astype(complex)

    def _functional_samples(self, op: Op, pulse: PulseFunctional) -> np.ndarray:
        length = getattr(op.op, "length", None)
        length = pulse.length if length is None else length
        _require(
            length is not None and not hasattr(length, "uid"),
            f"{pulse.function}: swept or missing length, no single set of samples",
        )
        parameters = dict(pulse.pulse_parameters or {})
        parameters.update(getattr(op.op, "pulse_parameters", None) or {})
        sampled = sample_pulse(
            signal_type="iq",
            sampling_rate=self.controller.sampling_rate / NS,
            length=length,
            amplitude=1.0,
            pulse_function=pulse.function,
            pulse_parameters=parameters,
        )
        if isinstance(sampled, dict):  # laboneq 26.4: i and q rows
            return np.asarray(sampled["samples_i"]) + 1j * np.asarray(
                sampled.get("samples_q", 0.0)
            )
        return np.asarray(sampled, dtype=complex)  # laboneq 26.10: complex

    def waveform(self, op: Op) -> np.ndarray:
        """Complex waveform played: amplitude x samples x exp(i phase).

        One row per sweep step when the amplitude or the phase is swept.
        """
        amplitude = np.asarray(self.amplitude(op))
        phase = np.asarray(self.phase(op))
        factor = amplitude * np.exp(1j * phase)
        return np.multiply.outer(factor, self.samples(op))

    def duration(self, op: Op) -> float:
        """Duration in ns: the operation length if given, else the samples."""
        length = values(getattr(op.op, "length", None))
        if length is not None:
            return length / NS
        return len(self.samples(op)) / self.controller.sampling_rate

    # frequencies

    def _lo(self, signal: str):
        calib = self._calibration(signal)
        if calib is not None and calib.local_oscillator is not None:
            return values(calib.local_oscillator.frequency)
        uid, number, _ = self.port(signal)
        for other in self.experiment.signals:
            if other != signal and self.port(other)[:2] == (uid, number):
                calib = self._calibration(other)
                if calib is not None and calib.local_oscillator is not None:
                    return values(calib.local_oscillator.frequency)
        return 0.0

    def frequency(self, channel: ChannelId):
        """Frequency on the channel's port in Hz: LO + oscillator.

        A float, or an array in sweep order when swept. An acquisition channel without
        an LO of its own uses the LO of its port.
        """
        signal = self.signal_of(channel)
        oscillator = self._calibration(signal).oscillator
        _expect(oscillator is not None, f"Channel '{channel}' has no oscillator.")
        return self._lo(signal) + values(oscillator.frequency)

    def oscillator(self, channel: ChannelId):
        """LabOne Q oscillator of the channel's signal."""
        return self._calibration(self.signal_of(channel)).oscillator

    def lo(self, channel: ChannelId):
        """LO frequency of the channel's port in Hz."""
        return self._lo(self.signal_of(channel))

    def offset(self, channel: ChannelId):
        """Voltage offset of a DC channel, in V."""
        return values(self._calibration(self.signal_of(channel)).voltage_offset)

    # sweeps

    def sweeps(self) -> list[SweepLoop]:
        """LabOne Q sweeps, outer first."""
        found = []

        def walk(node, real_time):
            if isinstance(node, Sweep):
                found.append(SweepLoop(list(node.parameters), real_time))
            if isinstance(node, Section):
                inside = real_time or isinstance(node, AcquireLoopRt)
                for child in node.children:
                    walk(child, inside)

        for section in self.experiment.sections:
            walk(section, False)
        return found

    def sweep_of(self, parameter) -> SweepLoop:
        """The sweep running ``parameter`` (or one it is derived from)."""
        loops = [
            loop
            for loop in self.sweeps()
            if any(_drives(p, parameter) for p in loop.parameters)
        ]
        _expect(len(loops) == 1, f"Parameter swept by {len(loops)} sweeps.")
        return loops[0]

    def driven_by(self, parameter) -> list[Driven]:
        """Everything ``parameter`` (or a parameter derived from it) drives."""
        driven = []
        for op in self.ops():
            for quantity in ("amplitude", "phase", "length", "time"):
                if _drives(parameter, getattr(op.op, quantity, None)):
                    driven.append(Driven(op.channel, quantity, op))
        for signal in self.experiment.signals:
            calib = self._calibration(signal)
            if calib is None:
                continue
            channel = self.channel_of(signal)
            quantities = {
                "frequency": getattr(calib.oscillator, "frequency", None),
                "lo": getattr(calib.local_oscillator, "frequency", None),
                "offset": calib.voltage_offset,
                "calibration amplitude": calib.amplitude,
            }
            for quantity, x in quantities.items():
                if _drives(parameter, x):
                    driven.append(Driven(channel, quantity))
        return driven

    # acquisition

    @property
    def acquire_loop(self) -> AcquireLoopRt:
        """The one real-time acquisition loop."""
        loops = []

        def walk(node):
            if isinstance(node, AcquireLoopRt):
                loops.append(node)
            for child in getattr(node, "children", None) or []:
                walk(child)

        for section in self.experiment.sections:
            walk(section)
        _expect(len(loops) == 1, f"{len(loops)} acquisition loops")
        return loops[0]

    @property
    def nshots(self) -> int:
        """Count of the acquisition loop."""
        return self.acquire_loop.count

    @property
    def averaging_mode(self):
        """LabOne Q averaging mode of the acquisition loop."""
        return self.acquire_loop.averaging_mode

    @property
    def acquisition_type(self):
        """LabOne Q acquisition type of the acquisition loop."""
        return self.acquire_loop.acquisition_type

    def handles(self, channel: ChannelId) -> list[str]:
        """Handles of the acquisitions on a qibolab channel, in order."""
        return [op.op.handle for op in self.acquires(channel)]

    def data(self, channel: ChannelId) -> list[np.ndarray]:
        """LabOne Q data of each acquisition on a qibolab channel, in order.

        In emulation LabOne Q returns placeholder data that differ between handles
        (real part) and between points (imaginary part).
        """
        _require(self.raw is not None, "Nothing was run.")
        return [np.asarray(self.raw.get_data(h)) for h in self.handles(channel)]

    # timing

    def waits_after_acquires(self, channel: ChannelId) -> list:
        """Time in ns waited on the channel after each of its acquisitions.

        Sum of the delays following the acquisition on the same signal, up to the next
        play or acquisition there, in one pass of the acquisition loop body. LabOne Q
        plays the operations of one signal in order, so this is the wait before the
        next shot (or the next sequence of the shot).
        """
        signals = set(self.signals_of(channel))
        ops = [op for op in self.ops() if op.real_time and op.signal in signals]
        waits = []
        for n, op in enumerate(ops):
            if not isinstance(op.op, Acquire):
                continue
            wait = 0.0
            for after in ops[n + 1 :]:
                if not isinstance(after.op, Delay):
                    break
                wait = wait + values(after.op.time) / NS
            waits.append(wait)
        return waits


def iq(data) -> np.ndarray:
    """Complex LabOne Q data as qibolab I/Q pairs on the last axis."""
    data = np.asarray(data)
    return np.stack([data.real, data.imag], axis=-1)


def square(duration=40, amplitude=0.3, phase=0.0) -> Pulse:
    """Rectangular pulse, the scenario pulse of the effect tests."""
    return Pulse(
        duration=duration,
        amplitude=amplitude,
        envelope=Rectangular(),
        relative_phase=phase,
    )


def drives_then_mz(platform, drives: dict) -> PulseSequence:
    """Pulses on drive channels, a delay on each readout line, then MZ of those qubits.

    ``drives`` maps a qubit to its drive pulses (qibocal rabi/utils.py:324).
    """
    sequence = PulseSequence()
    for q, pulses in drives.items():
        for p in pulses:
            sequence.append((f"{q}/drive", p))
    longest = max(sum(p.duration for p in pulses) for pulses in drives.values())
    for q in drives:
        sequence.append((f"{q}/acquisition", Wait(duration=longest)))
    for q in drives:
        sequence += platform.natives.single_qubit[q].MZ()
    return sequence


@contextmanager
def capture(controller) -> Iterator[Capture]:
    """Record what ``controller`` compiles and runs while the block executes."""
    cap = Capture(controller)
    session = controller.session
    compile, run = session.compile, session.run

    def compile_spy(experiment, *args, **kwargs):
        cap.experiments.append(experiment)
        return compile(experiment, *args, **kwargs)

    def run_spy(*args, **kwargs):
        cap.raw = run(*args, **kwargs)
        return cap.raw

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session, "compile", compile_spy)
        mp.setattr(session, "run", run_spy)
        yield cap


DEFAULTS = {
    "nshots": 10,
    "relaxation_time": 1000,
    "acquisition_type": AcquisitionType.INTEGRATION,
    "averaging_mode": AveragingMode.CYCLIC,
}
"""Execution parameters of :func:`play` when not given."""


def play(platform, sequences, sweepers=(), configs=None, **options) -> Capture:
    """Call ``ZiController.play`` once and return what it handed to LabOne Q.

    ``options`` are ``ExecutionParameters`` fields, over :data:`DEFAULTS`. The
    qibolab results are in :attr:`Capture.results`.
    """
    controller = platform.instruments["zi"]
    configs = configs if configs is not None else platform.parameters.configs.copy()
    parameters = ExecutionParameters(**(DEFAULTS | options))
    with capture(controller) as cap:
        cap.results = controller.play(
            configs, list(sequences), parameters, list(sweepers)
        )
    return cap
