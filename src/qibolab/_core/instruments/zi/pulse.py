"""Conversion of qibolab sequences to LabOne Q operations."""

from functools import cache

from laboneq.simple import Experiment, pulse_library

from qibolab._core.components import Channel, Config, IqChannel
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import (
    Delay,
    Envelope,
    Pulse,
    PulseLike,
    Readout,
    Rectangular,
)
from qibolab._core.sequence import PulseSequence

from .sweep import Amplitudes

__all__ = []

NS = 1e-9
"""Qibolab times are in ns, LabOne Q ones in s."""


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
def waveform(envelope: Envelope, duration: float, sampling_rate: float):
    """Unit-amplitude LabOne Q pulse with the samples of a qibolab envelope.

    Equal envelopes give the same object, since an SHFQA output can only hold one
    waveform.
    """
    samples = round(duration * sampling_rate)
    return pulse_library.sampled_pulse(envelope.i(samples) + 1j * envelope.q(samples))


def play(
    experiment: Experiment,
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    amplitudes: Amplitudes,
) -> None:
    """Add the operations of ``sequence`` to the current section of ``experiment``.

    A :class:`Readout` plays its probe pulse and acquires at the same time, integrating
    with unit weights (custom kernels are not supported). A delay on an acquisition
    channel also delays its probe channel, to keep them aligned. Swept amplitudes are
    taken from ``amplitudes``, by the id of the pulse or of its readout. Durations
    must be whole samples.
    """
    for id, op in sequence.align_to_delays():
        samples(id, op, sampling_rate)
        channel = channels[id]
        probe = channel.iqout(id)
        if isinstance(op, Delay):
            for signal in {id, probe} - {None}:
                experiment.delay(signal, time=op.duration * NS)
        elif isinstance(op, Pulse) and isinstance(channel, IqChannel):
            experiment.play(
                id,
                pulse=waveform(op.envelope, op.duration, sampling_rate),
                amplitude=amplitudes.get(op.id, op.amplitude),
                phase=op.relative_phase,
            )
        elif isinstance(op, Readout) and configs[id].kernel is None:
            experiment.play(
                probe,
                pulse=waveform(op.probe.envelope, op.probe.duration, sampling_rate),
                amplitude=amplitudes.get(op.id, op.probe.amplitude),
                phase=op.probe.relative_phase,
            )
            experiment.acquire(
                id,
                handle=str(op.id),
                kernel=waveform(Rectangular(), op.acquisition.duration, sampling_rate),
            )
        else:
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{id}' is not supported."
            )
