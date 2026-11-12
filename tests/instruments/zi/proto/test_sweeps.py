"""Sweeps added on ``orkun/proto-sweeps``, checked in LabOne Q emulation.

Each test checks what reaches the LabOne Q experiment (the operation that carries the
sweep parameter) and that the emulated run gives the qibolab result shape. Emulation
returns placeholder data, so nothing here says what the instruments output.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.simple import Calibration, Experiment, ExperimentSignal, SweepParameter

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.zi import pulse as zipulse
from qibolab._core.instruments.zi.controller import SAMPLING_RATE
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.pulses import Delay, Gaussian, Pulse
from qibolab._core.sequence import PulseSequence
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


DURATIONS = np.array([40, 80, 120])


@pytest.mark.parametrize(
    "parameter", [Parameter.duration, Parameter.duration_interpolated]
)
def test_duration_lookup(channels, configs, parameter):  # noqa: F811
    """F06, F07, F09: the swept length of a pulse and of its delay, in s."""
    sequence = drives([0])
    (_, drive), (_, delay) = list(sequence)[:2]
    swept = {}
    duration = Sweeper(parameter=parameter, values=DURATIONS, pulses=[drive, delay])
    split([[duration]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    played = swept[drive.id, Parameter.duration]
    assert played.values == pytest.approx(DURATIONS * 1e-9)
    assert ops["q0/drive"][0].length is played
    # the delay on the acquisition channel also delays the probe
    for signal in ("q0/acquisition", "q0/probe"):
        assert ops[signal][0].time is swept[delay.id, Parameter.duration]


@pytest.mark.parametrize("averaging_mode", AVERAGING)
@pytest.mark.parametrize(
    "parameter", [Parameter.duration, Parameter.duration_interpolated]
)
def test_rabi_length(connected, configs, averaging_mode, parameter):  # noqa: F811
    """F06, F09: duration of the drive and of the delay before the readout."""
    sequence = drives([0, 1])
    duration = Sweeper(
        parameter=parameter,
        values=DURATIONS,
        pulses=[op for _, op in list(sequence)[:4]],
    )
    run(connected, configs, sequence, [[duration]], averaging_mode)


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_delay_duration(connected, configs, averaging_mode):  # noqa: F811
    """F07, X23: T1, the delay between a pi pulse and the readout is swept."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    delay = Delay(duration=40)
    sequence = PulseSequence(
        [("q0/drive", drive), ("q0/acquisition", Delay(duration=40))]
    )
    sequence.append(("q0/acquisition", delay))
    sequence |= readouts([0])
    duration = Sweeper(parameter=Parameter.duration, values=DURATIONS, pulses=[delay])
    run(connected, configs, sequence, [[duration]], averaging_mode)


def test_fractional_duration(connected, configs):  # noqa: F811
    """G6: a drive pulse of 41.3 ns compiles, with the delay that follows it."""
    sequence = drives([0], duration=41.3)
    run(connected, configs, sequence, [], AveragingMode.CYCLIC)


PHASES = np.linspace(0, 2 * np.pi, 5)


def test_relative_phase_lookup(channels, configs):  # noqa: F811
    """F10: the swept phase of the drive pulse, in rad, in real time."""
    sequence = drives([0])
    drive = sequence[0][1]
    swept = {}
    phase = Sweeper(parameter=Parameter.relative_phase, values=PHASES, pulses=[drive])
    _, real, _ = split([[phase]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    assert real == [[swept[drive.id, Parameter.relative_phase]]]
    assert ops["q0/drive"][0].phase is real[0][0]
    assert real[0][0].values == pytest.approx(PHASES)


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_relative_phase(connected, configs, averaging_mode):  # noqa: F811
    """F10: relative phase of the drive pulses of two qubits."""
    sequence = drives([0, 1])
    phase = Sweeper(
        parameter=Parameter.relative_phase,
        values=PHASES,
        pulses=[sequence[0][1], sequence[2][1]],
    )
    run(connected, configs, sequence, [[phase]], averaging_mode)
