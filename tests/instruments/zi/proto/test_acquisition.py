"""Tests the acquisition options added to the ZI driver, in emulation.

LabOne Q emulation compiles the experiment and returns placeholder data, so these
tests check what reaches LabOne Q and the shape of the results, not their values.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import Acquire, Match
from laboneq.simple import Experiment, ModulationType

from qibolab._core.components import AcquisitionChannel, IqChannel, OscillatorConfig
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiController,
    ZiInstrument,
    ZiIqConfig,
    map_ports,
    results,
)
from qibolab._core.instruments.zi.controller import calibration, device_setup
from qibolab._core.pulses import Gaussian, Pulse
from qibolab._core.qubits import Qubit
from qibolab._core.sweeper import Parameter, Sweeper

from ..test_calibration import channels, configs  # noqa: F401
from ..test_play import (  # noqa: F401
    NSHOTS,
    connected,
    drives,
    frequencies,
    options,
    readouts,
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


def raw(averaging_mode) -> ExecutionParameters:
    return options(averaging_mode).model_copy(
        update={"acquisition_type": AcquisitionType.RAW}
    )


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_raw(connected, configs, averaging_mode):  # noqa: F811
    """Raw results have the samples before the i/q axis."""
    sequence = readouts([0])
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 0.4, 0.1),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    opts = raw(averaging_mode)

    results = connected.play(configs, [sequence], opts, [[amplitude]])

    (result,) = results.values()
    # 1000 ns at 2 GSa/s
    assert result.shape == opts.results_shape([[amplitude]], samples=2000)


def test_raw_sequential(connected, configs):  # noqa: F811
    """LabOne Q refuses sequential averaging of raw traces."""
    with pytest.raises(LabOneQException, match="Sequential"):
        connected.play(configs, [readouts([0])], raw(AveragingMode.SEQUENTIAL), [])


def test_raw_single_acquisition(connected, configs):  # noqa: F811
    with pytest.raises(NotImplementedError, match="RAW"):
        connected.play(configs, [readouts([0, 1])], raw(AveragingMode.CYCLIC), [])
    with pytest.raises(NotImplementedError, match="RAW"):
        connected.play(
            configs, [readouts([0]), readouts([0])], raw(AveragingMode.CYCLIC), []
        )


def test_raw_demodulated_at_probe_frequency(connected, configs, monkeypatch):  # noqa: F811
    """Raw traces are demodulated at the probe frequency relative to its LO."""
    frequencies = []
    demodulated = results.demodulated

    def record(data, frequency, sampling_rate):
        frequencies.append((frequency, sampling_rate))
        return demodulated(data, frequency, sampling_rate)

    monkeypatch.setattr(results, "demodulated", record)
    connected.play(configs, [readouts([2])], raw(AveragingMode.CYCLIC), [])

    lo = configs[connected.channels["q2/probe"].lo].frequency
    intermediate = (configs["q2/probe"].frequency - lo) * 1e-9
    assert frequencies == [(pytest.approx(intermediate), 2.0)]


def test_demodulated():
    """A trace at the intermediate frequency, with an offset, becomes constant."""
    frequency, rate, samples = 0.2, 2.0, 1000
    n = np.arange(samples)
    trace = 0.3 + (1 + 0.5j) * np.exp(2j * np.pi * frequency / rate * n)
    data = np.stack([trace.real, trace.imag], axis=-1)[np.newaxis]

    baseband = results.demodulated(data, frequency, rate)

    assert baseband.shape == (1, samples, 2)
    np.testing.assert_allclose(baseband[..., 0], np.sqrt(2) * 1.0, atol=1e-9)
    np.testing.assert_allclose(baseband[..., 1], np.sqrt(2) * 0.5, atol=1e-9)


def rx(qubit) -> tuple[str, Pulse]:
    return (
        f"q{qubit}/drive",
        Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2)),
    )


def fast_reset(averaging_mode) -> ExecutionParameters:
    return options(averaging_mode).model_copy(
        update={
            "acquisition_type": AcquisitionType.DISCRIMINATION,
            "fast_reset": True,
        }
    )


@pytest.mark.parametrize(
    "averaging_mode", [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]
)
def test_fast_reset(connected, configs, monkeypatch, averaging_mode):  # noqa: F811
    """After each readout, the reset pulse is played on state 1, on its channel.

    The sequences have no drive pulse, so the drive signals come from the resets.
    """
    connected.reset_pulses = {f"q{q}/acquisition": rx(q) for q in (0, 1)}
    experiments = compiled(connected, monkeypatch)
    sequences = [readouts([0, 1]), readouts([0])]
    opts = fast_reset(averaging_mode)

    results = connected.play(discriminating(configs), sequences, opts, [])

    assert len(results) == 3
    for result in results.values():
        assert result.shape == opts.results_shape([])
    (experiment,) = experiments
    matches = list(operations(experiment, Match))
    assert sorted(m.handle for m in matches) == sorted(
        str(ro.id) for seq in sequences for _, ro in seq.acquisitions
    )
    for match in matches:
        zero, one = match.children
        assert (zero.state, one.state) == (0, 1)
        assert zero.children == []
        (play,) = one.children
        assert play.signal in {"q0/drive", "q1/drive"}
        assert play.amplitude == 0.5


def test_fast_reset_needs_discrimination(connected, configs):  # noqa: F811
    connected.reset_pulses = {"q0/acquisition": rx(0)}
    opts = fast_reset(AveragingMode.CYCLIC).model_copy(
        update={"acquisition_type": AcquisitionType.INTEGRATION}
    )
    with pytest.raises(NotImplementedError, match="DISCRIMINATION"):
        connected.play(configs, [readouts([0])], opts, [])


def test_fast_reset_needs_pulse(connected, configs):  # noqa: F811
    connected.reset_pulses = {"q0/acquisition": rx(0)}
    with pytest.raises(ValueError, match="q1/acquisition"):
        connected.play(
            discriminating(configs),
            [readouts([0, 1])],
            fast_reset(AveragingMode.CYCLIC),
            [],
        )


SHFQC_LO = {"qc/qa/lo": 7.0e9, "qc/sg0/lo": 5.0e9}
"""LOs of the SHFQC rack: one for the QA channel, one for SG ports 0 and 1."""


@pytest.fixture
def shfqc():
    """Two qubits on one SHFQC (invented serial), driven on SG ports 0 and 1."""
    chs = {}
    cfgs = {lo: OscillatorConfig(frequency=f, power=0) for lo, f in SHFQC_LO.items()}
    for q in (0, 1):
        chs[f"q{q}/drive"] = IqChannel(device="qc", path=str(q), lo="qc/sg0/lo")
        chs[f"q{q}/probe"] = IqChannel(device="qc", path="0", lo="qc/qa/lo")
        chs[f"q{q}/acquisition"] = AcquisitionChannel(
            device="qc", path="0", probe=f"q{q}/probe"
        )
        cfgs[f"q{q}/drive"] = ZiIqConfig(frequency=5.1e9 + q * 1e6, power_range=10)
        cfgs[f"q{q}/probe"] = ZiIqConfig(frequency=7.1e9 + q * 1e6, power_range=5)
        cfgs[f"q{q}/acquisition"] = ZiAcquisitionConfig(
            delay=200, smearing=0, power_range=-30, threshold=0.1, iq_angle=0.2
        )
    controller = ZiController(
        address="127.0.0.1:8004",
        instruments={
            "qc": ZiInstrument(kind="SHFQC", serial="DEV12301", options="SHFQC/QC6CH")
        },
        channels=chs,
        emulation=True,
    )
    controller.connect()
    yield controller, cfgs
    controller.disconnect()


def test_shfqc_ports(shfqc):
    controller, _ = shfqc
    setup = controller.session.device_setup
    ports = {
        id: setup.logical_signal_by_uid(id).physical_channel.uid
        for id in controller.channels
    }
    assert ports == {
        "q0/drive": "qc/sgchannels_0_output",
        "q1/drive": "qc/sgchannels_1_output",
        "q0/probe": "qc/qachannels_0_output",
        "q1/probe": "qc/qachannels_0_output",
        "q0/acquisition": "qc/qachannels_0_input",
        "q1/acquisition": "qc/qachannels_0_input",
    }


def test_shfqc_oscillators(shfqc):
    controller, cfgs = shfqc
    ids = sorted(controller.channels)
    calib = calibration(ids, controller.channels, controller.instruments, cfgs)
    assert calib["q0/drive"].oscillator.modulation_type == ModulationType.HARDWARE
    assert calib["q0/probe"].oscillator.modulation_type == ModulationType.SOFTWARE
    assert calib["q0/acquisition"].oscillator is calib["q0/probe"].oscillator


@pytest.mark.parametrize(
    "acquisition_type", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
def test_shfqc_play(shfqc, acquisition_type):
    """Drive, sweep and readout run on an SHFQC, also with fast reset."""
    controller, cfgs = shfqc
    sequence = drives([0, 1])
    sweepers = [frequencies(cfgs, ["q0/drive", "q1/drive"], points=3)]
    opts = options(AveragingMode.CYCLIC).model_copy(
        update={"acquisition_type": acquisition_type}
    )

    results = controller.play(cfgs, [sequence], opts, sweepers)

    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)

    if acquisition_type is AcquisitionType.DISCRIMINATION:
        controller.reset_pulses = {f"q{q}/acquisition": rx(q) for q in (0, 1)}
        opts = opts.model_copy(update={"fast_reset": True})
        results = controller.play(cfgs, [sequence], opts, sweepers)
        assert len(results) == 2


def test_shfqc_qa_port():
    """An SHFQC has one QA channel, port 0, and its SG ports follow the model."""
    qc = ZiInstrument(kind="SHFQC", serial="DEV12301", options="SHFQC/QC4CH")
    assert qc.model == "QC4CH"
    assert qc.nports == 4
    chs = {
        "q0/probe": IqChannel(device="qc", path="1"),
        "q0/acquisition": AcquisitionChannel(device="qc", path="1", probe="q0/probe"),
    }
    with pytest.raises(ValueError, match="QA channel"):
        device_setup("127.0.0.1:8004", {"qc": qc}, None, chs)
    with pytest.raises(ValueError, match="Port 4"):
        device_setup(
            "127.0.0.1:8004",
            {"qc": qc},
            None,
            {"q0/drive": IqChannel(device="qc", path="4")},
        )


def test_shfqc_map_ports():
    qc = ZiInstrument(kind="SHFQC", serial="DEV12301")
    with pytest.raises(ValueError, match="write its channels directly"):
        map_ports({"qc": (qc, {0: ["q0"]})}, {"q0": Qubit.default("q0")})
