"""Tests the LabOne Q calibration built from the qibolab configs."""

import pytest

pytest.importorskip("laboneq")

from laboneq.simple import Experiment, ExperimentSignal, ModulationType, pulse_library

from qibolab._core.components import DcChannel, ExponentialFilter, OscillatorConfig
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiDcConfig,
    ZiIqConfig,
    infer_los,
)
from qibolab._core.instruments.zi.controller import calibration, signals
from qibolab._core.pulses import Acquisition, Delay, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence

from .conftest import CLUSTER, INSTRUMENTS, NQUBITS

DRIVE_LO = 6.2e9
PROBE_LO = 6.8e9


def _recipe(scheduled):
    """Device initializations: ``recipe`` in laboneq 26.4, ``artifacts`` in 26.10."""
    return getattr(scheduled, "recipe", None) or scheduled.artifacts


@pytest.fixture
def channels(channels):
    """Channels with the LO names from :func:`infer_los`."""
    los = infer_los(CLUSTER)
    for (qubit, is_probe), lo in los.items():
        id = f"{qubit}/probe" if is_probe else f"{qubit}/drive"
        channels[id] = channels[id].model_copy(update={"lo": lo})
    return channels


@pytest.fixture
def configs(channels):
    configs = {}
    for i in range(NQUBITS):
        q = f"q{i}"
        configs[f"{q}/drive"] = ZiIqConfig(frequency=6.3e9 + i * 1e6, power_range=10)
        configs[f"{q}/probe"] = ZiIqConfig(frequency=7.0e9 + i * 1e6, power_range=5)
        configs[f"{q}/acquisition"] = ZiAcquisitionConfig(
            delay=200, smearing=0, power_range=-30
        )
        configs[f"{q}/flux"] = ZiDcConfig(offset=0.1 * i, power_range=5)
    for id, ch in channels.items():
        if getattr(ch, "lo", None) is not None:
            lo = PROBE_LO if id.endswith("probe") else DRIVE_LO
            configs[ch.lo] = OscillatorConfig(frequency=lo, power=0)
    return configs


def test_drive(channels, configs):
    calib = calibration(["q1/drive"], channels, INSTRUMENTS, configs)["q1/drive"]
    assert calib.oscillator.frequency == pytest.approx(6.301e9 - DRIVE_LO)
    assert calib.oscillator.modulation_type == ModulationType.HARDWARE
    assert calib.local_oscillator.frequency == DRIVE_LO
    assert calib.range == 10


def test_drive_extra_shares_lo(channels, configs):
    channels["q0/drive12"] = channels["q0/drive"]
    configs["q0/drive12"] = ZiIqConfig(frequency=6.1e9, power_range=10)
    calib = calibration(["q0/drive", "q0/drive12"], channels, INSTRUMENTS, configs)
    drive, drive12 = calib["q0/drive"], calib["q0/drive12"]
    assert drive.local_oscillator is drive12.local_oscillator
    assert drive.oscillator is not drive12.oscillator
    assert drive12.oscillator.frequency == pytest.approx(6.1e9 - DRIVE_LO)


def test_probe_and_acquisition_share_oscillator(channels, configs):
    ids = ["q2/probe", "q2/acquisition"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    probe, acq = calib["q2/probe"], calib["q2/acquisition"]
    assert probe.oscillator is acq.oscillator
    assert probe.oscillator.modulation_type == ModulationType.SOFTWARE
    assert probe.oscillator.frequency == pytest.approx(7.002e9 - PROBE_LO)
    assert probe.local_oscillator.frequency == PROBE_LO
    assert probe.range == 5
    assert acq.port_delay == pytest.approx(200e-9)
    assert acq.range == -30


def test_flux(channels, configs):
    calib = calibration(["q3/flux"], channels, INSTRUMENTS, configs)["q3/flux"]
    assert calib.voltage_offset == pytest.approx(0.3)
    assert calib.range == 5


def test_flux_filters(channels, configs):
    """Filters become the HDAWG precompensation (refused before orkun/proto-pulses)."""
    filters = [ExponentialFilter(amplitude=-0.1, tau=40)]
    configs["q0/flux"] = configs["q0/flux"].model_copy(update={"filters": filters})
    calib = calibration(["q0/flux"], channels, INSTRUMENTS, configs)
    (exponential,) = calib["q0/flux"].precompensation.exponential
    assert exponential.timeconstant == pytest.approx(20e-9)
    assert exponential.amplitude == pytest.approx(-0.1)


@pytest.mark.parametrize("id", ["q0/drive", "q0/probe"])
def test_missing_lo(channels, configs, id):
    channels[id] = channels[id].model_copy(update={"lo": None})
    with pytest.raises(ValueError, match=id):
        calibration([id], channels, INSTRUMENTS, configs)


def test_signals(channels):
    seq = PulseSequence(
        [
            ("q0/drive", Pulse(duration=40, amplitude=0.5, envelope=Rectangular())),
            ("q1/acquisition", Delay(duration=10)),
            ("q1/acquisition", Acquisition(duration=1000)),
        ]
    )
    flux = {id for id, ch in channels.items() if isinstance(ch, DcChannel)}
    assert len(flux) == NQUBITS
    assert (
        set(signals(channels, [seq]))
        == {
            "q0/drive",
            "q1/acquisition",
            "q1/probe",
        }
        | flux
    )


def test_compile(controller, channels, configs):
    """The calibration compiles, and every flux offset reaches the HDAWG."""
    controller.channels = channels
    controller.connect()
    seq = PulseSequence(
        [
            ("q0/drive", Pulse(duration=40, amplitude=0.5, envelope=Rectangular())),
            ("q0/acquisition", Acquisition(duration=1000)),
        ]
    )
    ids = signals(channels, [seq])
    experiment = Experiment(uid="calib", signals=[ExperimentSignal(id) for id in ids])
    for id in ids:
        experiment.map_signal(
            id, controller.session.device_setup.logical_signal_by_uid(id)
        )
    experiment.set_calibration(calibration(ids, channels, INSTRUMENTS, configs))
    with experiment.acquire_loop_rt(count=1):
        with experiment.section(uid="drive"):
            experiment.play("q0/drive", pulse_library.const(length=40e-9))
        with experiment.section(uid="readout", play_after="drive"):
            experiment.play("q0/probe", pulse_library.const(length=1e-6))
            experiment.acquire(
                "q0/acquisition",
                handle="r",
                kernel=pulse_library.const(length=1e-6),
            )

    compiled = controller.session.compile(experiment)
    controller.disconnect()

    (hdawg,) = [
        init
        for init in _recipe(compiled.scheduled_experiment).initializations
        if init.device_uid == "hdawg"
    ]
    assert {o.channel: o.offset for o in hdawg.outputs} == pytest.approx(
        {i: 0.1 * i for i in range(NQUBITS)}
    )
