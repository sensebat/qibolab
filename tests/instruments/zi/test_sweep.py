"""Tests the split of the qibolab sweepers in near-time and real-time groups."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.instruments.zi.controller import calibration
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.pulses import Pulse, Rectangular
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import INSTRUMENTS
from .test_calibration import DRIVE_LO, PROBE_LO, channels, configs  # noqa: F401


def test_split(channels, configs):  # noqa: F811
    """The drive frequency goes to real time, the rest stays in near time."""
    ids = ["q0/drive", "q0/probe", "q0/acquisition"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    drive = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.30e9, 6.31e9]),
        channels=["q0/drive"],
    )
    amplitude = Sweeper(
        parameter=Parameter.amplitude, values=np.array([0.1, 0.2]), pulses=[pulse]
    )
    probe = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([7.0e9]),
        channels=["q0/probe"],
    )
    swept = {}

    near, real, axes = split(
        [[drive], [amplitude], [probe]], [], channels, configs, calib, swept
    )

    # LabOne Q returns the near-time sweeps (1 and 2), the shots, then the drive (0)
    assert axes == [1, 2, None, 0]
    assert len(near) == 2 and len(real) == 1
    # frequencies reach LabOne Q as intermediate frequencies
    assert real[0][0].values == pytest.approx(drive.values - DRIVE_LO)
    assert near[1][0].values == pytest.approx(probe.values - PROBE_LO)
    assert near[0][0].values == pytest.approx(amplitude.values)
    assert swept[pulse.id, Parameter.amplitude] is near[0][0]
    # the swept parameters are the oscillators of the calibration
    assert calib["q0/drive"].oscillator.frequency is real[0][0]
    assert calib["q0/probe"].oscillator.frequency is near[1][0]
    # the acquisition demodulates with the swept probe oscillator
    assert calib["q0/acquisition"].oscillator is calib["q0/probe"].oscillator
