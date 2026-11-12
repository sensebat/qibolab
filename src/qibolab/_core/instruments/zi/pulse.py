"""Conversion of qibolab sequences to LabOne Q operations."""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np
from laboneq.simple import Experiment, SectionTimingMode, pulse_library

from qibolab._core.components import (
    AcquisitionChannel,
    Channel,
    Config,
    DcChannel,
    IqChannel,
)
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

from .sweep import Amplitudes

__all__ = []

NS = 1e-9
"""Qibolab times are in ns, LabOne Q ones in s."""

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
    for time, pulse in pulses.items():
        raise ValueError(
            f"Pulse at {time} ns on '{probe}' has no acquisition starting at the "
            f"same time on '{id}'."
        )
    for _, op in probe_ops:
        if not isinstance(op, (Pulse, Delay)):
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{probe}' is not supported."
            )
    return sorted(measures, key=lambda measure: measure[0])


def _line(
    id: ChannelId, ops: Timeline, probe: ChannelId, probe_ops: Timeline
) -> list[Block]:
    """Blocks of an acquisition channel and its probe channel.

    They are the measures (see :func:`_measures`) and the delays between them. The
    section of a measure starts at the grid point before it and lasts until the
    grid point after it; delays fill the gaps. LabOne Q needs the sections with
    readouts on one QA line either to start together or not to overlap.
    """
    blocks: list[Block] = []
    start = 0.0
    for begin, measure in _measures(id, ops, probe, probe_ops):
        anchor = _grid(begin, up=False)
        if anchor < start:
            raise ValueError(
                f"Readout at {begin} ns on '{id}' is too close to the previous one, "
                f"whose section ends at {start} ns on the {READOUT_GRID} ns grid."
            )
        if anchor > start:
            blocks.append((start, anchor, (id, [Delay(duration=anchor - start)])))
        end = _grid(begin + max(measure.duration, measure.probe.duration))
        measure = replace(measure, wait=begin - anchor, length=end - anchor)
        blocks.append((anchor, end, measure))
        start = end
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
    offsets: dict[tuple, tuple[float, ChannelId]] = {}
    for blocks in layouts:
        lines: dict[tuple, list[tuple[float, float, ChannelId]]] = {}
        for start, _, block in blocks:
            if not isinstance(block, Measure):
                continue
            channel = channels[block.acquisition]
            line = (channel.device, channel.path)
            id, begin = block.acquisition, start + block.wait
            end = begin + max(block.duration, block.probe.duration)
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
    sequence: PulseSequence, channels: dict[ChannelId, Channel], sampling_rate: float
) -> list[Block]:
    """Split ``sequence`` in blocks, sorted by start time.

    Each block becomes one LabOne Q section. Since LabOne Q starts a section once
    the previous sections on its signals are over, playing them in this order keeps
    the qibolab timing of each channel. Durations must be whole samples.

    The operations of a channel without readouts are one block. An acquisition
    channel and its probe channel are split in measures and delays, see
    :func:`_line`: readouts may start at different times, and one qubit may be
    read out more than once, within the limits checked by :func:`check_lines`.
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
        blocks += _line(id, timelines.get(id, []), probe, timelines.get(probe, []))
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
    amplitudes: Amplitudes,
) -> None:
    """Add one operation of a run to the current section."""
    if isinstance(op, Delay):
        experiment.delay(id, time=op.duration * NS)
    elif isinstance(op, Pulse) and isinstance(channel, IqChannel):
        experiment.play(
            id,
            pulse=waveform(
                op.envelope, op.duration, sampling_rate, chirp=chirp(id, op)
            ),
            amplitude=amplitudes.get(op.id, op.amplitude),
            phase=op.relative_phase,
        )
    elif isinstance(op, Pulse) and isinstance(channel, DcChannel):
        if op.relative_phase != 0 or op.chirp is not None:
            raise ValueError(f"A pulse on the DC channel '{id}' has no phase or chirp.")
        experiment.play(
            id,
            pulse=waveform(op.envelope, op.duration, sampling_rate, real=True),
            amplitude=amplitudes.get(op.id, op.amplitude),
        )
    elif isinstance(op, VirtualZ) and isinstance(channel, IqChannel):
        experiment.play(id, pulse=None, increment_oscillator_phase=op.phase)
    else:
        raise NotImplementedError(
            f"{type(op).__name__} on channel '{id}' is not supported."
        )


def _measure(
    experiment: Experiment,
    measure: Measure,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    amplitudes: Amplitudes,
) -> None:
    """Add the probe pulse and the acquisition of a measure to the current section."""
    id = measure.acquisition
    probe = channels[id].iqout(id)
    assert probe is not None
    if configs[id].kernel is not None:
        raise NotImplementedError(f"Integration kernel on '{id}' is not supported.")
    if measure.wait > 0:
        experiment.delay(probe, time=measure.wait * NS)
        experiment.delay(id, time=measure.wait * NS)
    experiment.play(
        probe,
        pulse=waveform(
            measure.probe.envelope,
            measure.probe.duration,
            sampling_rate,
            chirp=chirp(probe, measure.probe),
        ),
        amplitude=amplitudes.get(measure.sweep, measure.probe.amplitude),
        phase=measure.probe.relative_phase,
    )
    experiment.acquire(
        id,
        handle=str(measure.handle),
        kernel=waveform(Rectangular(), measure.duration, sampling_rate),
    )
    for signal, duration in [(probe, measure.probe.duration), (id, measure.duration)]:
        experiment.delay(signal, time=(measure.length - measure.wait - duration) * NS)


def play(
    experiment: Experiment,
    blocks: list[Block],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    amplitudes: Amplitudes,
) -> None:
    """Add the blocks of a sequence to the current section, one subsection each.

    See :func:`layout`. Pulses play on IQ channels and, with real samples, on DC
    channels (flux and coupler lines). A :class:`VirtualZ` increments the phase of
    the channel's oscillator by its phase. A measure plays its probe pulse and
    acquires at the same time, integrating with unit weights. Swept amplitudes are
    taken from ``amplitudes``, by the id of the pulse or of its readout.
    """
    for _, _, block in blocks:
        if isinstance(block, Measure):
            with experiment.section(
                length=block.length * NS, section_timing_mode=SectionTimingMode.STRICT
            ):
                _measure(
                    experiment, block, channels, configs, sampling_rate, amplitudes
                )
        else:
            with experiment.section():
                id, ops = block
                for op in ops:
                    _op(experiment, id, op, channels[id], sampling_rate, amplitudes)
