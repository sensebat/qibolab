"""Conversion of qibolab sweepers to LabOne Q sweep parameters."""

import numpy as np
from laboneq.simple import Calibration, ModulationType, SweepParameter

from qibolab._core.components import (
    AcquisitionChannel,
    Channel,
    Config,
    DcChannel,
    IqChannel,
)
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
    Parameter.phase: (Parameter.phase, 1.0, True),
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
    sampling_rate: float | None = None,
) -> tuple[list[SweepParameter], bool]:
    """Sweep parameters of one group of parallel sweepers, and their execution type.

    Supported sweeps:

    - frequency of a channel: the parameter replaces the frequency of the channel's
      oscillator in ``calibration``. It runs in real time if the oscillator is
      modulated by the hardware (drive), in near time if by the software (probe),
      since an SHFQA generator cannot be swept in real time in integration mode;
    - offset of a DC channel: the parameter replaces the ``voltage_offset`` in
      ``calibration``, with absolute values, and runs in near time (it is a node
      setting);
    - parameters of pulses (:data:`_PULSES`): the parameter is recorded in
      ``swept``, to be used when the pulse is played. Amplitude runs in near time;
      duration (in s, also interpolated: both resample the envelope), relative
      phase and the phase of a virtual Z run in real time.

    A group that mixes near-time and real-time sweeps runs entirely in near time.
    Given ``sampling_rate`` (GSps), swept durations must be whole samples, as fixed
    ones (:func:`.pulse.samples`): LabOne Q would round them.
    """
    parameters = []
    real_time = set()
    for sweeper in group:
        if sweeper.parameter is Parameter.frequency:
            for id in sweeper.channels:
                channel = channels[id]
                # ValueError, as for the other channel checks of the driver
                if isinstance(channel, AcquisitionChannel):
                    raise ValueError(  # noqa: TRY004
                        f"Cannot sweep the frequency of the acquisition channel "
                        f"'{id}': sweep the frequency of its probe channel "
                        f"'{channel.probe}'."
                    )
                if not isinstance(channel, IqChannel):
                    raise ValueError(  # noqa: TRY004
                        f"Cannot sweep the frequency of '{id}': it is not an IQ "
                        "channel."
                    )
                oscillator = calibration[id].oscillator
                lo = configs[channels[id].lo].frequency
                oscillator.frequency = SweepParameter(values=sweeper.values - lo)
                parameters.append(oscillator.frequency)
                real_time.add(oscillator.modulation_type is ModulationType.HARDWARE)
        elif sweeper.parameter is Parameter.offset:
            for id in sweeper.channels:
                if not isinstance(channels[id], DcChannel):
                    raise ValueError(  # noqa: TRY004
                        f"Cannot sweep the offset of '{id}': it is not a DC channel."
                    )
                calib = calibration[id]
                calib.voltage_offset = SweepParameter(values=sweeper.values)
                parameters.append(calib.voltage_offset)
                real_time.add(False)
        elif sweeper.parameter in _PULSES:
            name, scale, rt = _PULSES[sweeper.parameter]
            if name is Parameter.duration and sampling_rate is not None:
                n = np.asarray(sweeper.values) * sampling_rate
                if not np.allclose(n, np.round(n), atol=1e-6):
                    raise ValueError(
                        f"Swept durations {sweeper.values} ns are not all a whole "
                        f"number of samples at {sampling_rate} GSps."
                    )
            for pulse in sweeper.pulses:
                swept[pulse.id, name] = SweepParameter(values=sweeper.values * scale)
                parameters.append(swept[pulse.id, name])
                real_time.add(rt)
        else:
            raise NotImplementedError(
                f"Sweeping {sweeper.parameter.name} is not supported."
            )
    return parameters, real_time == {True}


def split(
    sweepers: list[ParallelSweepers],
    channels: dict[ChannelId, Channel],
    configs: dict[str, Config],
    calibration: Calibration,
    swept: Swept,
    sampling_rate: float | None = None,
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
        parameters, real_time = _group(
            group, channels, configs, calibration, swept, sampling_rate
        )
        if real_time:
            real.append(parameters)
            real_axes.append(index)
        else:
            near.append(parameters)
            near_axes.append(index)
    return near, real, near_axes + [None] + real_axes
