"""Tests the acquisition options added to the ZI driver, in emulation.

LabOne Q emulation compiles the experiment and returns placeholder data, so these
tests check what reaches LabOne Q and the shape of the results, not their values.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.dsl.experiment import Acquire
from laboneq.simple import Experiment

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.sweeper import Parameter, Sweeper

from ..test_calibration import channels, configs  # noqa: F401
from ..test_play import (  # noqa: F401
    NSHOTS,
    connected,
    drives,
    frequencies,
    options,
)


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


def compiled(controller, monkeypatch) -> list:
    """Record the experiments the controller compiles."""
    experiments = []
    compile = controller.session.compile

    def record(experiment, *args, **kwargs):
        experiments.append(experiment)
        return compile(experiment, *args, **kwargs)

    monkeypatch.setattr(controller.session, "compile", record)
    return experiments


def operations(node, kind):
    """All the operations of type ``kind`` in a LabOne Q experiment or section."""
    children = node.sections if isinstance(node, Experiment) else node.children
    for child in children:
        if isinstance(child, kind):
            yield child
        elif hasattr(child, "children"):
            yield from operations(child, kind)


def discriminating(configs, threshold=0.3, iq_angle=0.7):  # noqa: F811
    """Configs with a threshold and an angle on every acquisition channel."""
    return {
        id: (
            config.model_copy(update={"threshold": threshold, "iq_angle": iq_angle})
            if id.endswith("acquisition")
            else config
        )
        for id, config in configs.items()
    }


@pytest.mark.parametrize("averaging_mode", list(AveragingMode))
def test_discrimination(connected, configs, averaging_mode):  # noqa: F811
    """Discriminated results have no i/q axis."""
    sequence = drives([0, 1])
    sweepers = [frequencies(configs, ["q0/drive", "q1/drive"], points=3)]
    opts = options(averaging_mode).model_copy(
        update={"acquisition_type": AcquisitionType.DISCRIMINATION}
    )

    results = connected.play(discriminating(configs), [sequence], opts, sweepers)

    assert len(results) == 2
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert np.isrealobj(result)


def test_discrimination_kernel_threshold(connected, configs, monkeypatch):  # noqa: F811
    """LabOne Q gets the opposite threshold and the kernel ``-exp(1j * iq_angle)``.

    LabOne Q assigns state 0 above its threshold, qibolab state 1 above
    ``threshold`` after rotating by ``iq_angle`` (see :func:`zi.pulse.kernel`).
    """
    experiments = compiled(connected, monkeypatch)
    opts = options(AveragingMode.SINGLESHOT).model_copy(
        update={"acquisition_type": AcquisitionType.DISCRIMINATION}
    )
    connected.play(discriminating(configs), [drives([0])], opts, [])

    (experiment,) = experiments
    assert experiment.get_calibration()["q0/acquisition"].threshold == -0.3
    (acquire,) = operations(experiment, Acquire)
    samples = acquire.kernel.samples
    assert len(samples) == 2000
    np.testing.assert_allclose(samples, -np.exp(0.7j))


def test_discrimination_needs_threshold(connected, configs):  # noqa: F811
    opts = options(AveragingMode.CYCLIC).model_copy(
        update={"acquisition_type": AcquisitionType.DISCRIMINATION}
    )
    with pytest.raises(ValueError, match="q0/acquisition"):
        connected.play(configs, [drives([0])], opts, [])


def test_default_options(connected, configs):  # noqa: F811
    """The qibolab defaults (single-shot discrimination) run (X01)."""
    opts = ExecutionParameters(nshots=NSHOTS, relaxation_time=1000)
    sequence = drives([0])

    results = connected.play(discriminating(configs), [sequence], opts, [])

    (result,) = results.values()
    assert result.shape == (NSHOTS,)
