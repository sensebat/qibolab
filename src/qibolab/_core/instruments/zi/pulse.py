"""Conversion of qibolab sequences to LabOne Q operations."""

from functools import cache

from laboneq.simple import Experiment, pulse_library
from pydantic import TypeAdapter

from qibolab._core.components import Channel, Config, IqChannel
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Delay, Envelope, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter

from .sweep import NS, Swept

__all__ = []


@cache
def waveform(envelope: Envelope, duration: float, sampling_rate: float):
    """Unit-amplitude LabOne Q pulse with the samples of a qibolab envelope.

    Equal envelopes give the same object, since an SHFQA output can only hold one
    waveform.
    """
    samples = int(duration * sampling_rate)
    return pulse_library.sampled_pulse(envelope.i(samples) + 1j * envelope.q(samples))


_ENVELOPE = TypeAdapter(Envelope)


@pulse_library.register_pulse_functional
def qibolab_envelope(x, envelope=None, **_):
    """Samples of the qibolab envelope serialized in ``envelope``, one per ``x``.

    LabOne Q samples it for the length given to each play, so that a swept duration
    resamples the envelope (as a new qibolab pulse of that duration would), and the
    pulse and its delays are rounded on the same time grid.
    """
    env = _ENVELOPE.validate_python(envelope)
    return env.i(len(x)) + 1j * env.q(len(x))


ENVELOPE = qibolab_envelope(uid="qibolab_envelope")
"""Pulse playing any qibolab envelope, given with its length in each play."""


def play(
    experiment: Experiment,
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    swept: Swept,
) -> None:
    """Add the operations of ``sequence`` to the current section of ``experiment``.

    A :class:`Readout` plays its probe pulse and acquires at the same time, integrating
    with unit weights (custom kernels are not supported). A delay on an acquisition
    channel also delays its probe channel, to keep them aligned. Swept values are
    taken from ``swept``; the amplitude of a probe by the id of the readout or of its
    probe pulse.
    """
    for id, op in sequence.align_to_delays():
        channel = channels[id]
        probe = channel.iqout(id)
        if isinstance(op, Delay):
            time = swept.get((op.id, Parameter.duration), op.duration * NS)
            for signal in {id, probe} - {None}:
                experiment.delay(signal, time=time)
        elif isinstance(op, Pulse) and isinstance(channel, IqChannel):
            experiment.play(
                id,
                pulse=ENVELOPE,
                length=swept.get((op.id, Parameter.duration), op.duration * NS),
                pulse_parameters={"envelope": op.envelope.model_dump(mode="json")},
                amplitude=swept.get((op.id, Parameter.amplitude), op.amplitude),
                phase=op.relative_phase,
            )
        elif isinstance(op, Readout) and configs[id].kernel is None:
            experiment.play(
                probe,
                pulse=waveform(op.probe.envelope, op.probe.duration, sampling_rate),
                amplitude=swept.get(
                    (op.id, Parameter.amplitude),
                    swept.get((op.probe.id, Parameter.amplitude), op.probe.amplitude),
                ),
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
