"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sweeper import ParallelSweepers, Parameter

__all__ = []

Swept = dict[tuple[PulseId | ChannelId, Parameter], SweepParameter]
"""Swept value of each pulse or channel, by its id and the swept parameter."""

NS = 1e-9
"""Qibolab times are in ns, LabOne Q ones in s."""

_PULSES = {
    Parameter.amplitude: (Parameter.amplitude, 1.0, False),
    Parameter.duration: (Parameter.duration, NS, True),
    Parameter.duration_interpolated: (Parameter.duration, NS, True),
    Parameter.relative_phase: (Parameter.relative_phase, 1.0, True),
}
"""Swept pulse parameters: key in :data:`Swept`, scale to LabOne Q, real time."""

Axes = list[int | None]
"""Qibolab sweeper of each axis of the LabOne Q results, ``None`` for the shots."""


def _group(
    group: ParallelSweepers,
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    swept: Swept,
) -> tuple[list[SweepParameter], bool]:
    """Sweep parameters of one group of parallel sweepers, and their execution type.

    Supported sweeps:

    - frequency of a channel: the parameter replaces the frequency of the channel's
      oscillator in ``calibration``. It runs in real time if the oscillator is
      modulated by the hardware (drive), in near time if by the software (probe),
      since an SHFQA generator cannot be swept in real time in integration mode;
    - parameters of pulses (:data:`_PULSES`): the parameter is recorded in
      ``swept``, to be used when the pulse is played. Amplitude runs in near time;
      duration (in s, also interpolated: both resample the envelope) and relative
      phase run in real time.
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
                real_time.add(oscillator.modulation_type is ModulationType.HARDWARE)
        elif sweeper.parameter in _PULSES:
            name, scale, rt = _PULSES[sweeper.parameter]
            for pulse in sweeper.pulses:
                swept[pulse.id, name] = SweepParameter(values=sweeper.values * scale)
                parameters.append(swept[pulse.id, name])
                real_time.add(rt)
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
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    swept: Swept,
) -> tuple[list[list[SweepParameter]], list[list[SweepParameter]], Axes]:
    """Split the sweepers in near-time and real-time groups, keeping their order.

    Each group of parallel sweepers becomes one list of sweep parameters, to be
    opened as a single LabOne Q sweep. See :func:`_group` for the supported sweeps
    and for which of them run in real time.

    The last element maps the axes of the LabOne Q results, which are the near-time
    sweeps, then the shots, then the real-time ones, back to the qibolab sweepers.
    """
    near: list[list[SweepParameter]] = []
    real: list[list[SweepParameter]] = []
    near_axes: Axes = []
    real_axes: Axes = []
    for index, group in enumerate(sweepers):
        parameters, real_time = _group(group, channels, configs, calibration, swept)
        if real_time:
            real.append(parameters)
            real_axes.append(index)
        else:
            near.append(parameters)
            near_axes.append(index)
    return near, real, near_axes + [None] + real_axes
