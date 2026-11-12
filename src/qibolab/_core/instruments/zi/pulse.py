"""Conversion of qibolab sequences to LabOne Q operations."""

from functools import cache

import numpy as np
from laboneq.simple import Experiment, pulse_library

from qibolab._core.components import Channel, Config, IqChannel
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Delay, Envelope, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence

from .sweep import Amplitudes

__all__ = []

NS = 1e-9
"""Qibolab times are in ns, LabOne Q ones in s."""


@cache
def waveform(envelope: Envelope, duration: float, sampling_rate: float):
    """Unit-amplitude LabOne Q pulse with the samples of a qibolab envelope.

    Equal envelopes give the same object, since an SHFQA output can only hold one
    waveform.
    """
    samples = int(duration * sampling_rate)
    return pulse_library.sampled_pulse(envelope.i(samples) + 1j * envelope.q(samples))


@cache
def kernel(duration: float, sampling_rate: float, iq_angle: float | None = None):
    """Integration kernel of an acquisition: unit weights.

    To discriminate, pass the ``iq_angle`` of the acquisition config: the weights are
    multiplied by ``-exp(1j * iq_angle)``, and :func:`.controller.calibration` sets the
    threshold to ``-threshold``. Qibolab (as qm and qibocal) assigns state 1 when
    ``Re((I + 1j * Q) * exp(1j * iq_angle)) > threshold``, with ``I + 1j * Q`` the
    integrated result of unit weights. LabOne Q assigns state 0 when the integrated
    result is above its threshold (2-state assignment vector of its multistate
    setup), and its integration is linear in the kernel, which is the convention of
    ``laboneq.analysis.calculate_integration_kernels``.

    Equal kernels give the same object: different kernels on one acquisition signal
    would set up more than 2 states.
    """
    if iq_angle is None:
        return waveform(Rectangular(), duration, sampling_rate)
    samples = int(duration * sampling_rate)
    return pulse_library.sampled_pulse(-np.exp(1j * iq_angle) * np.ones(samples))


def play(
    experiment: Experiment,
    sequence: PulseSequence,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    sampling_rate: float,
    amplitudes: Amplitudes,
    acquisition_type: AcquisitionType,
) -> None:
    """Add the operations of ``sequence`` to the current section of ``experiment``.

    A :class:`Readout` plays its probe pulse and acquires at the same time, integrating
    with unit weights (custom kernels are not supported). A delay on an acquisition
    channel also delays its probe channel, to keep them aligned. Swept amplitudes are
    taken from ``amplitudes``, by the id of the pulse or of its readout.

    The kernel follows ``acquisition_type``: see :func:`kernel`. Raw acquisitions
    have no kernel, only the length of the acquisition.
    """
    discriminate = acquisition_type is AcquisitionType.DISCRIMINATION
    for id, op in sequence.align_to_delays():
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
            if acquisition_type is AcquisitionType.RAW:
                experiment.acquire(
                    id, handle=str(op.id), length=op.acquisition.duration * NS
                )
                continue
            iq_angle = (configs[id].iq_angle or 0.0) if discriminate else None
            experiment.acquire(
                id,
                handle=str(op.id),
                kernel=kernel(op.acquisition.duration, sampling_rate, iq_angle),
            )
        else:
            raise NotImplementedError(
                f"{type(op).__name__} on channel '{id}' is not supported."
            )
