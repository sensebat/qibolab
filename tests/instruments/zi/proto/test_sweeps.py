"""Sweeps added on ``orkun/proto-sweeps``, checked in LabOne Q emulation.

Each test checks what reaches the LabOne Q experiment (the operation that carries the
sweep parameter) and that the emulated run gives the qibolab result shape. Emulation
returns placeholder data, so nothing here says what the instruments output.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.simple import Experiment, ExperimentSignal, SweepParameter

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.zi import pulse as zipulse
from qibolab._core.instruments.zi.controller import SAMPLING_RATE
from qibolab._core.sweeper import Parameter, Sweeper

from ..test_calibration import channels, configs  # noqa: F401
from ..test_play import connected, drives, options, readouts  # noqa: F401

AVERAGING = [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]


def operations(sequence, channels, configs, swept):  # noqa: F811
    """LabOne Q operations played for ``sequence``, by signal."""
    experiment = Experiment(signals=[ExperimentSignal(id) for id in channels])
    with experiment.acquire_loop_rt(count=1), experiment.section() as section:
        zipulse.play(experiment, sequence, channels, configs, SAMPLING_RATE, swept)
    ops = {}
    for op in section.children:
        ops.setdefault(op.signal, []).append(op)
    return ops


def run(connected, configs, sequence, sweepers, averaging_mode):  # noqa: F811
    """Run in emulation and check the shape of every result."""
    opts = options(averaging_mode)
    results = connected.play(configs, [sequence], opts, sweepers)
    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()
    return results


def test_probe_amplitude_lookup(channels, configs):  # noqa: F811
    """G1: a sweeper given ``Readout.probe`` sweeps the probe amplitude."""
    sequence = readouts([0])
    readout = sequence[0][1]
    parameter = SweepParameter(values=np.array([0.1, 0.2]))
    swept = {(readout.probe.id, Parameter.amplitude): parameter}

    ops = operations(sequence, channels, configs, swept)

    assert ops["q0/probe"][0].amplitude is parameter


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_probe_amplitude(connected, configs, averaging_mode):  # noqa: F811
    """G1: punchout with the sweeper on the probe pulses, as qibocal may build it."""
    sequence = readouts([0, 1])
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        values=np.array([0.1, 0.5, 0.9]),
        pulses=[ro.probe for _, ro in sequence.acquisitions],
    )
    run(connected, configs, sequence, [[amplitude]], averaging_mode)
