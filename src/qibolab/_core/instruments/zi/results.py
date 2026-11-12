"""Conversion of LabOne Q results to qibolab ones."""

import numpy as np

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.pulses.modulation import demodulate

from .sweep import Axes

__all__ = []


def parse(data: np.ndarray, options: ExecutionParameters, axes: Axes) -> Result:
    """Convert the data of one acquisition to the qibolab shape.

    LabOne Q returns complex data with axes (near-time sweeps, shots, real-time
    sweeps), while qibolab expects (shots, sweeps in their order, i/q). ``axes``
    gives the qibolab sweeper of each LabOne Q axis, as built by
    :func:`.sweep.split`; the shots axis is there only when averaging in single shot.
    Discriminated data (states, or their average) has no i/q axis. Raw data has
    the samples before the i/q axis.
    """
    shots = options.averaging_mode is AveragingMode.SINGLESHOT
    source = [axis for axis in axes if shots or axis is not None]
    data = np.moveaxis(
        data,
        list(range(len(source))),
        [0 if axis is None else axis + int(shots) for axis in source],
    )
    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
        return np.real(data)
    return np.stack([np.real(data), np.imag(data)], axis=-1)


def demodulated(data: Result, frequency: float, sampling_rate: float) -> Result:
    """Bring parsed raw data, with axes (..., samples, i/q), to baseband.

    The SHFQA scope records the input relative to the LO, so the trace still
    oscillates at the probe ``frequency`` relative to the LO, in GHz.
    :func:`qibolab._core.pulses.modulation.demodulate` removes it, as in the qblox and
    qm drivers.
    """
    iq = demodulate(np.moveaxis(data, -1, 0), frequency, sampling_rate)
    return np.moveaxis(iq, 0, -1)
