"""Tests emulated execution, replicating the calls of qibocal protocols."""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.pulses import Delay, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import NQUBITS
from .test_calibration import channels, configs  # noqa: F401

NSHOTS = 7


@pytest.fixture
def connected(controller, channels):  # noqa: F811
    controller.channels = channels
    controller.connect()
    yield controller
    controller.disconnect()


def readouts(qubits) -> PulseSequence:
    """Multiplexed readout, as the ``MZ`` natives of a platform."""
    probe = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())
    return PulseSequence(
        [(f"q{q}/acquisition", Readout.from_probe(probe.new())) for q in qubits]
    )


def drives(qubits, duration: int = 200) -> PulseSequence:
    """Drive pulse on each qubit, then a delayed readout, as qubit spectroscopy."""
    drive = Pulse(duration=duration, amplitude=1.0, envelope=Rectangular())
    sequence = PulseSequence()
    for q in qubits:
        sequence.append((f"q{q}/drive", drive.new()))
        sequence.append((f"q{q}/acquisition", Delay(duration=duration)))
    return sequence | readouts(qubits)


def frequencies(configs, channels, points: int = 5) -> list[Sweeper]:  # noqa: F811
    """Absolute frequency sweeper of each channel, as the qibocal protocols."""
    return [
        Sweeper(
            parameter=Parameter.frequency,
            values=configs[channel].frequency + np.linspace(-2e6, 2e6, points),
            channels=[channel],
        )
        for channel in channels
    ]


def options(averaging_mode) -> ExecutionParameters:
    return ExecutionParameters(
        nshots=NSHOTS,
        relaxation_time=1000,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=averaging_mode,
    )


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_resonator_spectroscopy(connected, configs, averaging_mode):  # noqa: F811
    sequence = readouts(range(NQUBITS))
    probes = [f"q{q}/probe" for q in range(NQUBITS)]
    sweepers = [frequencies(configs, probes)]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_multiple_sequences(connected, configs, averaging_mode):  # noqa: F811
    sequences = [readouts([0, 1]), readouts([0])]
    opts = options(averaging_mode)

    results = connected.play(configs, sequences, opts, [])

    assert len(results) == 3
    for result in results.values():
        assert result.shape == opts.results_shape([])


def test_resonator_punchout(connected, configs):  # noqa: F811
    sequence = readouts(range(NQUBITS))
    probes = [f"q{q}/probe" for q in range(NQUBITS)]
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    sweepers = [[amplitude], frequencies(configs, probes)]
    opts = options(AveragingMode.CYCLIC)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_qubit_spectroscopy(connected, configs, averaging_mode):  # noqa: F811
    """The drive frequency is swept in real time, inside the acquisition loop."""
    sequence = drives(range(NQUBITS))
    drive = [f"q{q}/drive" for q in range(NQUBITS)]
    sweepers = [frequencies(configs, drive)]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_near_and_real_time_axes(connected, configs, averaging_mode):  # noqa: F811
    """A near-time and a real-time sweep come back in the order qibolab asked.

    The two have different lengths, so a swapped axis changes the shape. The
    real-time drive sweep is given first, although LabOne Q returns it last.
    """
    sequence = drives([0])
    sweepers = [
        frequencies(configs, ["q0/drive"], points=3),
        frequencies(configs, ["q0/probe"], points=5),
    ]
    opts = options(averaging_mode)

    results = connected.play(configs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()


def test_unsupported(connected, configs):  # noqa: F811
    sequence = readouts([0])
    with pytest.raises(NotImplementedError):
        connected.play(configs, [sequence], ExecutionParameters(nshots=1), [])
    opts = options(AveragingMode.CYCLIC)
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence([("q0/drive", drive)]) | sequence
    # the drive frequency is swept in real time, the probe one in near time
    parallel = frequencies(configs, ["q0/drive", "q0/probe"])
    with pytest.raises(NotImplementedError, match="real time"):
        connected.play(configs, [sequence], opts, [parallel])
    for update in (
        {"fast_reset": True},
        {"averaging_mode": AveragingMode.SEQUENTIAL},
        {"acquisition_type": AcquisitionType.DISCRIMINATION},
        {"acquisition_type": AcquisitionType.RAW},
    ):
        with pytest.raises(NotImplementedError):
            connected.play(configs, [sequence], opts.model_copy(update=update), [])
