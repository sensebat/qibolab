"""Conversion of qibolab sequences to LabOne Q operations."""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from laboneq.simple import Experiment, SectionTimingMode, pulse_library
from pydantic import TypeAdapter

from qibolab._core.components import (
    AcquisitionChannel,
    Channel,
    Config,
    DcChannel,
    IqChannel,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import (
    Acquisition,
    Delay,
    Envelope,
    Pulse,
    PulseId,
    PulseLike,
    Readout,
    Rectangular,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter

from .sweep import NS, Swept

__all__ = []

CHIRP_UNITS = {
    "Hz/nsec": 1.0,
    "mHz/nsec": 1e-3,
    "uHz/nsec": 1e-6,
    "pHz/nsec": 1e-12,
    "GHz/sec": 1.0,
    "MHz/sec": 1e-3,
    "KHz/sec": 1e-6,
    "Hz/sec": 1e-9,
    "mHz/sec": 1e-12,
}
"""Factor from each chirp unit (the QUA ones, as in the qm driver) to Hz/ns."""

Chirp = tuple[float, str]
"""Constant chirp rate and its unit."""


def chirp(id: ChannelId, pulse: Pulse) -> Chirp | None:
    """Check the chirp of ``pulse`` on channel ``id``.

    Only a constant rate is supported, given as ``(rate, unit)``.
    """
    if pulse.chirp is None:
        return None
    if len(pulse.chirp) != 2 or isinstance(pulse.chirp[0], list):
        raise NotImplementedError(
            f"Piecewise chirp on channel '{id}' is not supported, only (rate, unit)."
        )
    rate, unit = pulse.chirp
    if unit not in CHIRP_UNITS:
        raise ValueError(
            f"Unknown chirp unit '{unit}' on channel '{id}', "
            f"use one of {list(CHIRP_UNITS)}."
        )
    return rate, unit


def samples(id: ChannelId, op: PulseLike, sampling_rate: float) -> None:
    """Check that the durations of ``op`` on channel ``id`` are whole samples.

    LabOne Q would round them, moving the pulse with respect to the other channels.
    """
    parts = [op.probe, op.acquisition] if isinstance(op, Readout) else [op]
    for part in parts:
        n = part.duration * sampling_rate
        if abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"{type(part).__name__} of {part.duration} ns on channel '{id}' is not "
                f"a whole number of samples at {sampling_rate} GSps."
            )


@cache
def waveform(
    envelope: Envelope,
    duration: float,
    sampling_rate: float,
    real: bool = False,
    chirp: Chirp | None = None,
):
    """Unit-amplitude LabOne Q pulse with the samples of a qibolab envelope.

    Equal envelopes give the same object, since an SHFQA output can only hold one
    waveform. With ``real``, for a DC channel, only the i component is kept.

    A ``chirp`` moves the frequency linearly from the oscillator frequency at the
    start of the pulse, as in QUA: the samples get the phase ``pi * rate * t**2``.
    """
    n = round(duration * sampling_rate)
    samples = envelope.i(n) if real else envelope.i(n) + 1j * envelope.q(n)
    if chirp is not None:
        rate = chirp[0] * CHIRP_UNITS[chirp[1]] * NS  # in GHz/ns
        t = np.arange(n) / sampling_rate
        samples = samples * np.exp(1j * np.pi * rate * t**2)
    return pulse_library.sampled_pulse(samples)


_ENVELOPE = TypeAdapter(Envelope)


@pulse_library.register_pulse_functional
def qibolab_envelope(x, envelope=None, **_):
    """Samples of the qibolab envelope serialized in ``envelope``, one per ``x``.

    LabOne Q samples it for the length given to each play, so that a swept duration
    resamples the envelope (as a new qibolab pulse of that duration would).
    """
    env = _ENVELOPE.validate_python(envelope)
    return env.i(len(x)) + 1j * env.q(len(x))


ENVELOPE = qibolab_envelope(uid="qibolab_envelope")
"""Pulse playing any qibolab envelope, given with its length in each play."""


READOUT_GRID = 8.0
"""Grid in ns of the start and length of a LabOne Q section with a readout.

It is 16 samples at 2 GSps, the SHFQA sequencer grid. Sections with a readout are
compiled in strict timing mode, so LabOne Q raises an error if this is wrong,
rather than moving the readout.
"""


@dataclass(frozen=True)
class Measure:
    """Probe pulse and acquisition that start together."""

    acquisition: ChannelId
    """Acquisition channel."""
    probe: Pulse
    """Probe pulse, played on the probe channel of :attr:`acquisition`."""
    duration: float
    """Acquisition duration in ns."""
    handle: PulseId
    """Id of the acquisition, by which the results are returned."""
    sweep: PulseId
    """Id by which a swept probe amplitude is found."""
    wait: float = 0.0
    """Time in ns from the start of the section to the measure."""
    length: float = 0.0
    """Length in ns of the section, on :data:`READOUT_GRID`."""
    window: float = 0.0
    """Time in ns the acquisition takes, the longest one on its QA line.

    The SHFQA integrates for the same time on all the integrators of a QA line, so
    LabOne Q extends shorter acquisitions; the weights after their end are zero.
    """

    @property
    def end(self) -> float:
        """End in ns of the measure, from the start of its section."""
        return self.wait + max(self.window, self.probe.duration)


Run = tuple[ChannelId, list[PulseLike]]
"""Consecutive operations of one channel."""

Block = tuple[float, float, Run | Measure]
"""Start and end in ns of a run or a measure, which becomes one LabOne Q section."""


def _grid(time: float, up: bool = True) -> float:
    """Point of :data:`READOUT_GRID` at or after ``time``, or before with ``up=False``."""
    rounding = np.ceil if up else np.floor
    return float(rounding(time / READOUT_GRID + (-1e-9 if up else 1e-9)) * READOUT_GRID)


Timeline = list[tuple[float, PulseLike]]
"""Operations of one channel with their start time in ns."""


def _timeline(ops: list[PulseLike]) -> Timeline:
    """Start time of each operation of one channel."""
    times = np.cumsum([0.0] + [op.duration for op in ops])
    return [(float(time), op) for time, op in zip(times, ops)]


def _measures(
    id: ChannelId, ops: Timeline, probe: ChannelId, probe_ops: Timeline
) -> list[tuple[float, Measure]]:
    """Measures of an acquisition channel, with their start time.

    A :class:`Readout` is one measure. An :class:`Acquisition` is one measure with
    the :class:`Pulse` that starts at the same time on the probe channel. Other
    than these, both channels may only hold delays.
    """
    pulses = {round(t, 6): op for t, op in probe_ops if isinstance(op, Pulse)}
    measures: list[tuple[float, Measure]] = []
    for time, op in ops:
        if isinstance(op, Readout):
            measure = Measure(id, op.probe, op.acquisition.duration, op.id, op.id)
            measures.append((time, measure))
        elif isinstance(op, Acquisition):
            pulse = pulses.pop(round(time, 6), None)
            if pulse is None:
                raise ValueError(
                    f"Acquisition at {time} ns on '{id}' has no probe pulse starting "
                    f"at the same time on '{probe}'."
                )
            measures.append((time, Measure(id, pulse, op.duration, op.id, pulse.id)))
        elif not isinstance(op, Delay):
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{id}' is not supported."
            )
    if len(pulses) > 0:
        raise ValueError(
            f"Pulse at {min(pulses)} ns on '{probe}' has no acquisition starting at "
            f"the same time on '{id}'."
        )
    for _, op in probe_ops:
        if not isinstance(op, (Pulse, Delay)):
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{probe}' is not supported."
            )
    return sorted(measures, key=lambda measure: measure[0])


Line = tuple[str | None, str | None]
"""QA line, as the instrument and port of its acquisition channels."""


def _qa_line(channel: Channel) -> Line:
    """QA line of an acquisition channel."""
    return (channel.device, channel.path)


def windows(
    sequences: list[PulseSequence], channels: dict[ChannelId, Channel]
) -> dict[Line, float]:
    """Longest acquisition in ns on each QA line, see :attr:`Measure.window`."""
    out: dict[Line, float] = {}
    for sequence in sequences:
        for id, op in sequence.acquisitions:
            duration = (
                op.acquisition.duration if isinstance(op, Readout) else op.duration
            )
            line = _qa_line(channels[id])
            out[line] = max(out.get(line, 0.0), duration)
    return out


def _gap(
    id: ChannelId, ops: Timeline, begin: float, end: float, cut: float, swept: Swept
) -> list[PulseLike]:
    """Delays of channel ``id`` between ``begin`` and ``end``, shortened by ``cut``.

    The delays keep their ids, so that swept durations apply. ``cut`` (ns) is taken
    from the delays that are not swept, the last ones first. A swept delay must
    change by multiples of :data:`READOUT_GRID`: LabOne Q would otherwise move the
    readout after it onto the grid, without an error.
    """
    delays = [op for t, op in ops if begin - 1e-6 <= t < end - 1e-6]
    for op in delays:
        parameter = swept.get((op.id, Parameter.duration))
        if parameter is None:
            continue
        steps = (np.asarray(parameter.values) / NS - op.duration) / READOUT_GRID
        if not np.allclose(steps, np.round(steps), atol=1e-6):
            raise ValueError(
                f"The swept delay before the readout at {end} ns on '{id}' takes "
                f"values {np.asarray(parameter.values) / NS} ns: they must differ "
                f"from {op.duration} ns by multiples of {READOUT_GRID} ns, the "
                "readout grid."
            )
    out: list[PulseLike] = []
    for op in reversed(delays):
        if cut > 1e-9 and (op.id, Parameter.duration) not in swept:
            taken = min(cut, op.duration)
            cut -= taken
            op = op.model_copy(update={"duration": op.duration - taken})
        if op.duration > 1e-9 or (op.id, Parameter.duration) in swept:
            out.append(op)
    if cut > 1e-9:
        raise ValueError(
            f"The delays before the readout at {end} ns on '{id}' are swept: the "
            f"readout section starts on the {READOUT_GRID} ns grid, which needs "
            f"{cut} ns more of fixed delay."
        )
    return out[::-1]


def _line(
    id: ChannelId,
    ops: Timeline,
    probe: ChannelId,
    probe_ops: Timeline,
    window: float,
    swept: Swept,
) -> list[Block]:
    """Blocks of an acquisition channel and its probe channel.

    They are the measures (see :func:`_measures`) and the delays between them. The
    section of a measure starts at the grid point before it and lasts until the
    grid point after it; the delays of the acquisition channel fill the gaps, with
    their ids, shortened (see :func:`_gap`) by the part of the gap the measure
    sections take. LabOne Q needs the sections with readouts on one QA line either
    to start together or not to overlap.
    """
    blocks: list[Block] = []
    start = 0.0
    previous = 0.0
    for begin, measure in _measures(id, ops, probe, probe_ops):
        anchor = _grid(begin, up=False)
        if anchor < start:
            raise ValueError(
                f"Readout at {begin} ns on '{id}' is too close to the previous one, "
                f"which ends at {start} ns: it lasts as long as the longest "
                f"acquisition on its QA line, then up to the {READOUT_GRID} ns grid."
            )
        gap = _gap(
            id, ops, previous, begin, (start - previous) + (begin - anchor), swept
        )
        if len(gap) > 0:
            blocks.append((start, anchor, (id, gap)))
        measure = replace(measure, wait=begin - anchor, window=window)
        end = _grid(anchor + measure.end)
        blocks.append((anchor, end, replace(measure, length=end - anchor)))
        start = end
        previous = begin + next(
            op.duration
            for t, op in ops
            if abs(t - begin) < 1e-6 and not isinstance(op, Delay)
        )
    time = max(t + op.duration for t, op in ops + probe_ops)
    if time > start:
        blocks.append((start, time, (id, [Delay(duration=time - start)])))
    return blocks


def check_lines(layouts: list[list[Block]], channels: dict[ChannelId, Channel]) -> None:
    """Check the SHFQA limits on the readouts of one QA line.

    Overlapping readouts must start together, and all the readouts must start at the
    same time modulo :data:`READOUT_GRID`, since the delay of the acquisition from
    the grid is one setting. LabOne Q reports both with less clear messages, or,
    for the second one, only on some versions.
    """
    offsets: dict[Line, tuple[float, ChannelId]] = {}
    for blocks in layouts:
        lines: dict[Line, list[tuple[float, float, ChannelId]]] = {}
        for start, _, block in blocks:
            if not isinstance(block, Measure):
                continue
            line = _qa_line(channels[block.acquisition])
            id, begin, end = block.acquisition, start + block.wait, start + block.end
            for other_begin, other_end, other in lines.get(line, []):
                if begin < other_end and other_begin < end and begin != other_begin:
                    raise ValueError(
                        f"Readouts on '{other}' at {other_begin} ns and on '{id}' at "
                        f"{begin} ns overlap on port {line[1]} of '{line[0]}': "
                        "readouts on one QA line must start together or not overlap."
                    )
            lines.setdefault(line, []).append((begin, end, id))
            first, other = offsets.setdefault(line, (begin, id))
            if not np.isclose((begin - first) % READOUT_GRID, 0) and not np.isclose(
                (begin - first) % READOUT_GRID, READOUT_GRID
            ):
                raise ValueError(
                    f"Readouts on '{other}' at {first} ns and on '{id}' at {begin} ns "
                    f"on port {line[1]} of '{line[0]}': readouts on one QA line must "
                    f"start at the same time modulo {READOUT_GRID} ns."
                )


def layout(
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    sampling_rate: float,
    windows: dict[Line, float],
    swept: Swept | None = None,
) -> list[Block]:
    """Split ``sequence`` in blocks, sorted by start time.

    Each block becomes one LabOne Q section. Since LabOne Q starts a section once
    the previous sections on its signals are over, playing them in this order keeps
    the qibolab timing of each channel. Durations must be whole samples.

    The operations of a channel without readouts are one block. An acquisition
    channel and its probe channel are split in measures and delays, see
    :func:`_line`: readouts may start at different times, and one qubit may be
    read out more than once, within the limits checked by :func:`check_lines`.
    Delays whose duration is in ``swept`` keep it; the layout itself (grid, checks)
    uses the durations in the sequence.
    """
    blocks: list[Block] = []
    sequence = sequence.align_to_delays()
    for id, op in sequence:
        samples(id, op, sampling_rate)
    timelines = {id: _timeline(list(sequence.channel(id))) for id in sequence.channels}
    lines = {
        id: channel.probe
        for id, channel in channels.items()
        if isinstance(channel, AcquisitionChannel)
        and channel.probe is not None
        and (id in timelines or channel.probe in timelines)
    }
    for id, probe in lines.items():
        blocks += _line(
            id,
            timelines.get(id, []),
            probe,
            timelines.get(probe, []),
            windows.get(_qa_line(channels[id]), 0.0),
            swept or {},
        )
    for id, ops in timelines.items():
        if id not in lines and id not in lines.values():
            blocks.append(
                (0.0, ops[-1][0] + ops[-1][1].duration, (id, [op for _, op in ops]))
            )
    return sorted(blocks, key=lambda block: block[:2])


def _op(
    experiment: Experiment,
    id: ChannelId,
    op: PulseLike,
    channel: Channel,
    sampling_rate: float,
    swept: Swept,
) -> None:
    """Add one operation of a run to the current section.

    A delay on an acquisition channel also delays its probe channel. Pulses play
    :data:`ENVELOPE` with their length, so that it can be swept; a chirped pulse
    plays its samples, and its duration cannot be swept.
    """
    duration = swept.get((op.id, Parameter.duration), op.duration * NS)
    if isinstance(op, Delay):
        for signal in {id, channel.iqout(id)} - {None}:
            experiment.delay(signal, time=duration)
    elif isinstance(op, Pulse) and isinstance(channel, (IqChannel, DcChannel)):
        dc = isinstance(channel, DcChannel)
        if dc and (op.relative_phase != 0 or op.chirp is not None):
            raise ValueError(f"A pulse on the DC channel '{id}' has no phase or chirp.")
        if dc and (op.id, Parameter.relative_phase) in swept:
            raise ValueError(f"A pulse on the DC channel '{id}' has no phase to sweep.")
        rate = None if dc else chirp(id, op)
        if rate is not None and (op.id, Parameter.duration) in swept:
            raise NotImplementedError(
                f"Sweeping the duration of a chirped pulse on '{id}' is not supported."
            )
        kwargs = {}
        if not dc:
            kwargs["phase"] = swept.get(
                (op.id, Parameter.relative_phase), op.relative_phase
            )
        if rate is None:
            kwargs.update(
                pulse=ENVELOPE,
                length=duration,
                pulse_parameters={"envelope": op.envelope.model_dump(mode="json")},
            )
        else:
            kwargs["pulse"] = waveform(
                op.envelope, op.duration, sampling_rate, chirp=rate
            )
        experiment.play(
            id,
            amplitude=swept.get((op.id, Parameter.amplitude), op.amplitude),
            **kwargs,
        )
    elif isinstance(op, VirtualZ) and isinstance(channel, IqChannel):
        experiment.play(
            id,
            pulse=None,
            increment_oscillator_phase=swept.get((op.id, Parameter.phase), op.phase),
        )
    else:
        raise NotImplementedError(
            f"{type(op).__name__} on channel '{id}' is not supported."
        )


@cache
def discrimination_sign() -> float:
    """1 if LabOne Q assigns state 1 above the threshold of a 2-state
    discrimination, -1 if it assigns state 0.

    LabOne Q sets the assignment vector of the SHFQA multistate unit with
    ``_calc_theoretical_assignment_vec`` (``controller/devices/qachannel.py``). For
    2 states it gives state 0 above the threshold in laboneq 26.4.0 (zhinst-utils
    0.7.1), and state 1 in laboneq 26.10.1 (zhinst-utils 0.7.2 swaps the two states
    of each weight in ``QuditSettings``).
    """
    from laboneq.controller.devices.qachannel import _calc_theoretical_assignment_vec

    return 1.0 if _calc_theoretical_assignment_vec(1)[1] == 1 else -1.0


@cache
def kernel(
    duration: float,
    sampling_rate: float,
    weights: tuple[complex, ...] | None = None,
    iq_angle: float | None = None,
):
    """LabOne Q kernel of an acquisition; equal kernels give the same object.

    The integration weights ``w`` are the ``weights`` of the acquisition config (the
    kernel as qibocal computes it), or unit weights. To discriminate, pass the
    ``iq_angle`` of the config: the weights become ``sign * exp(1j * iq_angle) * w``,
    and :func:`.controller.calibration` sets the threshold to ``sign * threshold``,
    with ``sign`` from :func:`discrimination_sign`. Qibolab (as qm and qibocal)
    assigns state 1 when ``Re(sum(w * s) * exp(1j * iq_angle)) > threshold``.

    The kernel given to LabOne Q is ``w`` as it is: LabOne Q integrates
    ``sum(kernel * s)`` over the trace ``s`` its RAW acquisition returns, the
    convention of ``laboneq.analysis.calculate_integration_kernels`` (the conjugate
    of the difference of the state traces, used directly as kernel), which is also
    how qibocal computes ``w`` from qibolab RAW traces. In the compiled artifacts
    the kernel wave is ``conj(w)`` times the oscillator; the controller conjugates
    it again on upload (``controller/devices/qachannel.py``). Not checked on
    hardware.

    Different kernels on one acquisition signal would set up more than 2 states, so
    equal ones must be the same object.
    """
    if weights is None and iq_angle is None:
        return waveform(Rectangular(), duration, sampling_rate)
    n = round(duration * sampling_rate)
    w = np.ones(n, dtype=complex) if weights is None else np.array(weights)
    if iq_angle is not None:
        w = discrimination_sign() * np.exp(1j * iq_angle) * w
    return pulse_library.sampled_pulse(w)


Slots = dict[ChannelId, list[tuple]]
"""Probe settings of each readout slot, by acquisition channel; see :func:`_slot`."""


def _slot(
    experiment: Experiment, slots: Slots, id: ChannelId, probe: ChannelId, key: tuple
) -> tuple[ChannelId, ChannelId, list[ChannelId]]:
    """Probe and acquisition signals of a measure, and all the signals of its line.

    An SHFQA generator holds one probe pulse per signal and real-time loop: one
    waveform, amplitude and phase. A measure whose probe differs (``key``) from the
    earlier ones on the same acquisition channel gets a new slot: new experiment
    signals ``{probe}/{n}`` and ``{id}/{n}``, mapped to the same ports, with the
    same calibration. Each slot takes one generator waveform and one integrator.
    """
    keys = slots.setdefault(id, [])
    if key not in keys:
        keys.append(key)
        if len(keys) > 1:
            for base in (probe, id):
                signal = experiment.signals[base]
                new = experiment.add_signal(
                    f"{base}/{len(keys) - 1}",
                    connect_to=signal.mapped_logical_signal_path,
                )
                new.calibration = signal.calibration
    names = [(probe, id)] + [(f"{probe}/{n}", f"{id}/{n}") for n in range(1, len(keys))]
    probe_signal, acquisition_signal = names[keys.index(key)]
    return probe_signal, acquisition_signal, [s for pair in names for s in pair]


def _measure(
    experiment: Experiment,
    measure: Measure,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    swept: Swept,
    slots: Slots,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
) -> None:
    """Add the probe pulse and the acquisition of a measure to the current section.

    The acquisition integrates with unit weights, or with the kernel of the
    acquisition config, which must have one weight per sample; to discriminate, the
    weights are rotated by the ``iq_angle`` of the config (see :func:`kernel`). A raw
    acquisition has no kernel, only the length of the acquisition.

    The signals come from :func:`_slot`; the other signals of the line are reserved,
    so that the measures of one channel keep their order.
    """
    id = measure.acquisition
    probe = channels[id].iqout(id)
    assert probe is not None
    config = configs[id]
    weights = config.kernel
    if weights is not None and len(weights) != round(measure.duration * sampling_rate):
        raise ValueError(
            f"The kernel of '{id}' has {len(weights)} samples, but the acquisition "
            f"lasts {round(measure.duration * sampling_rate)} samples."
        )
    discriminate = acquisition_type is AcquisitionType.DISCRIMINATION
    weights_key = None if weights is None else tuple(np.asarray(weights).tolist())
    iq_angle = (config.iq_angle or 0.0) if discriminate else None
    pulse = measure.probe
    ids = {measure.sweep, pulse.id}
    if {p for i, p in swept if i in ids} - {Parameter.amplitude}:
        raise NotImplementedError(
            f"Sweeping the probe of '{id}' is supported only in amplitude."
        )
    amplitude = swept.get(
        (measure.sweep, Parameter.amplitude),
        swept.get((pulse.id, Parameter.amplitude), pulse.amplitude),
    )
    key = (
        pulse.envelope,
        pulse.duration,
        chirp(probe, pulse),
        ("swept", amplitude.uid) if hasattr(amplitude, "uid") else amplitude,
        pulse.relative_phase,
        measure.duration,
    )
    probe_signal, signal, line = _slot(experiment, slots, id, probe, key)
    for other in set(line) - {probe_signal, signal}:
        experiment.reserve(other)
    if measure.wait > 0:
        experiment.delay(probe_signal, time=measure.wait * NS)
        experiment.delay(signal, time=measure.wait * NS)
    experiment.play(
        probe_signal,
        pulse=waveform(
            pulse.envelope, pulse.duration, sampling_rate, chirp=chirp(probe, pulse)
        ),
        amplitude=amplitude,
        phase=pulse.relative_phase,
    )
    if acquisition_type is AcquisitionType.RAW:
        experiment.acquire(
            signal, handle=str(measure.handle), length=measure.duration * NS
        )
    else:
        experiment.acquire(
            signal,
            handle=str(measure.handle),
            kernel=kernel(measure.duration, sampling_rate, weights_key, iq_angle),
        )
    for name, duration in [(probe_signal, pulse.duration), (signal, measure.window)]:
        experiment.delay(name, time=(measure.length - measure.wait - duration) * NS)


def play(
    experiment: Experiment,
    blocks: list[Block],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    swept: Swept,
    slots: Slots,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
) -> None:
    """Add the blocks of a sequence to the current section, one subsection each.

    See :func:`layout`. Pulses play on IQ channels and, with real samples, on DC
    channels (flux and coupler lines). A :class:`VirtualZ` increments the phase of
    the channel's oscillator by its phase. A measure plays its probe pulse and
    acquires at the same time. Swept values are taken from ``swept``, by the id of the
    operation; the amplitude of a probe by the id of its readout or of the probe
    pulse. Other sweeps of a probe are refused: an SHFQA generator holds one
    waveform per signal. ``slots``
    is shared by the sequences of one experiment, see :func:`_slot`. The kernel
    follows ``acquisition_type``, see :func:`_measure`.
    """
    for _, _, block in blocks:
        if isinstance(block, Measure):
            with experiment.section(
                length=block.length * NS, section_timing_mode=SectionTimingMode.STRICT
            ):
                _measure(
                    experiment,
                    block,
                    channels,
                    configs,
                    sampling_rate,
                    swept,
                    slots,
                    acquisition_type,
                )
        else:
            with experiment.section():
                id, ops = block
                run(experiment, id, ops, channels, sampling_rate, swept)


def run(
    experiment: Experiment,
    id: ChannelId,
    ops: list[PulseLike],
    channels: dict[ChannelId, Channel],
    sampling_rate: float,
    swept: Swept,
) -> None:
    """Add the operations ``ops`` of channel ``id`` to the current section."""
    for op in ops:
        _op(experiment, id, op, channels[id], sampling_rate, swept)
