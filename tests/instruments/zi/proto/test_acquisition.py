"""Tests the acquisition options added to the ZI driver, in emulation.

LabOne Q emulation compiles the experiment and returns placeholder data, so these
tests check what reaches LabOne Q and the shape of the results, not their values.
"""

import pytest

pytest.importorskip("laboneq")

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.sweeper import Parameter, Sweeper

from ..test_calibration import channels, configs  # noqa: F401
from ..test_play import connected, drives, frequencies, options  # noqa: F401


def test_sequential(connected, configs):  # noqa: F811
    """Sequential averaging returns the same axes as cyclic averaging.

    A real-time (drive frequency) and a near-time (probe amplitude) sweep of
    different lengths make a swapped axis visible in the shape.
    """
    sequence = drives([0, 1])
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 0.5, 0.1),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    sweepers = [frequencies(configs, ["q0/drive", "q1/drive"], points=3), [amplitude]]
    opts = options(AveragingMode.SEQUENTIAL)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert len(results) == 2
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers) == (3, 4, 2)
