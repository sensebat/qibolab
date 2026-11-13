"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter

__all__ = []

Amplitudes = dict[PulseId, SweepParameter]
"""Swept amplitude of each pulse, by pulse id."""

Axes = list[int | None]
"""Qibolab sweeper of each axis of the LabOne Q results, ``None`` for the shots."""


def _channels_of(sequences: list[PulseSequence]) -> dict[PulseId, ChannelId]:
    """Channel each operation of the sequences is played on, by its id."""
    return {op.id: id for sequence in sequences for id, op in sequence}


def _real_time(id: ChannelId, calibration: Calibration) -> bool:
    """Whether a sweep of the channel ``id`` runs in real time.

    It does if the channel's oscillator is modulated by the hardware (drive), and
    not if it is modulated by the software (probe and acquisition): an SHFQA
    generator cannot be swept in real time in integration mode.
    """
    oscillator = calibration[id].oscillator
    return (
        oscillator is not None and oscillator.modulation_type is ModulationType.HARDWARE
    )


def _group(
    group: ParallelSweepers,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    amplitudes: Amplitudes,
    pulse_channels: dict[PulseId, ChannelId],
) -> tuple[list[SweepParameter], bool]:
    """Sweep parameters of one group of parallel sweepers, and their execution type.

    Supported sweeps, both on the channel's own time scale (:func:`_real_time`):

    - frequency of a channel: the parameter replaces the frequency of the channel's
      oscillator in ``calibration``;
    - amplitude of pulses: the parameter is recorded in ``amplitudes``, to be used
      when the pulse is played. The channel of a pulse is the one it is played on in
      the sequences, which for a :class:`Readout` is its acquisition channel.
    """
    parameters = []
    real_time = set()
    for sweeper in group:
        if sweeper.parameter is Parameter.frequency:
            for id in sweeper.channels:
                oscillator = calibration[id].oscillator
                lo = configs[channels[id].lo].frequency
                oscillator.frequency = SweepParameter(values=sweeper.values - lo)
                parameters.append(oscillator.frequency)
                real_time.add(_real_time(id, calibration))
        elif sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                amplitudes[pulse.id] = SweepParameter(values=sweeper.values)
                parameters.append(amplitudes[pulse.id])
                real_time.add(_real_time(pulse_channels[pulse.id], calibration))
        else:
            raise NotImplementedError(
                f"Sweeping {sweeper.parameter.name} is not supported."
            )
    if len(real_time) > 1:
        raise NotImplementedError(
            "Sweeping in parallel in near time and in real time is not supported."
        )
    return parameters, real_time == {True}


def split(
    sweepers: list[ParallelSweepers],
    sequences: list[PulseSequence],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    amplitudes: Amplitudes,
) -> tuple[list[list[SweepParameter]], list[list[SweepParameter]], Axes]:
    """Split the sweepers in near-time and real-time groups, keeping their order.

    Each group of parallel sweepers becomes one list of sweep parameters, to be
    opened as a single LabOne Q sweep. See :func:`_group` for the supported sweeps
    and for which of them run in real time; ``sequences`` are the ones to be played,
    which tell on which channel each swept pulse is.

    The last element maps the axes of the LabOne Q results, which are the near-time
    sweeps, then the shots, then the real-time ones, back to the qibolab sweepers.
    """
    near: list[list[SweepParameter]] = []
    real: list[list[SweepParameter]] = []
    near_axes: Axes = []
    real_axes: Axes = []
    pulse_channels = _channels_of(sequences)
    for index, group in enumerate(sweepers):
        parameters, real_time = _group(
            group, channels, configs, calibration, amplitudes, pulse_channels
        )
        if real_time:
            real.append(parameters)
            real_axes.append(index)
        else:
            near.append(parameters)
            near_axes.append(index)
    return near, real, near_axes + [None] + real_axes
