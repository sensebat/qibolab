"""Pulses added on orkun/proto-pulses, run in LabOne Q emulation.

Emulation compiles the experiment and returns placeholder data. Timing and samples
are checked on the compiled output with the LabOne Q output simulator.
"""

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

from ..test_calibration import channels, configs  # noqa: F401

OPTIONS = ExecutionParameters(
    nshots=3,
    relaxation_time=1000,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
)

PROBE = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())


@pytest.fixture
def connected(controller, channels):  # noqa: F811
    controller.channels = channels
    controller.connect()
    yield controller
    controller.disconnect()


def readout(qubit: int = 0, probe: Pulse = PROBE) -> tuple[str, Readout]:
    return (f"q{qubit}/acquisition", Readout.from_probe(probe.new()))


def run(controller, configs, sequences, sweepers=()):  # noqa: F811
    """Play ``sequences``, checking one result of the expected shape per readout."""
    results = controller.play(configs, sequences, OPTIONS, list(sweepers))
    ids = {ro.id for seq in sequences for _, ro in seq.acquisitions}
    assert set(results) == ids
    for result in results.values():
        assert result.shape == OPTIONS.results_shape(list(sweepers))
    return results


@pytest.mark.parametrize(
    "op",
    [
        ("q0/drive", Pulse(duration=41.3, amplitude=0.5, envelope=Rectangular())),
        ("q0/acquisition", Delay(duration=10.25)),
        readout(probe=PROBE.model_copy(update={"duration": 999.9})),
    ],
)
def test_sub_sample_duration_refused(connected, configs, op):  # noqa: F811
    with pytest.raises(ValueError, match="whole number of samples"):
        connected.play(configs, [PulseSequence([op, readout()])], OPTIONS, [])


def test_half_ns_duration(connected, configs):  # noqa: F811
    """41.5 ns is 83 samples at 2 GSps."""
    drive = Pulse(duration=41.5, amplitude=0.5, envelope=Rectangular())
    run(connected, configs, [PulseSequence([("q0/drive", drive), readout()])])
