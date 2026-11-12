"""Fixtures for the conformance suite: a small fake rack in LabOne Q emulation.

Each test carries ``@pytest.mark.feature("F01")`` or ``@pytest.mark.checkpoint("CP1")``.
The id, its title and its checkpoint go into the junit xml as test properties, and the
laboneq and zhinst-core versions go in as suite properties.
"""

import re
from importlib.metadata import version

import pytest

pytest.importorskip("laboneq")

from laboneq.core.exceptions import LabOneQException

from qibolab._core.components import OscillatorConfig
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiConfigs,
    ZiController,
    ZiDcConfig,
    ZiInstrument,
    ZiIqConfig,
    infer_los,
    map_ports,
)
from qibolab._core.native import Native, SingleQubitNatives
from qibolab._core.parameters import ConfigKinds, NativeGates, Parameters, Settings
from qibolab._core.platform import Platform
from qibolab._core.pulses import Gaussian, Pulse, Readout, Rectangular
from qibolab._core.qubits import Qubit

from .features import BY_ID

QUBITS = ["q0", "q1"]
NSHOTS = 10
RELAXATION = 1000

RACK = {
    "hdawg": (
        ZiInstrument(kind="HDAWG", serial="DEV8901", options="HDAWG8/PC"),
        {i: [q] for i, q in enumerate(QUBITS)},
    ),
    "shfsg": (
        ZiInstrument(kind="SHFSG", serial="DEV12901", options="SHFSG8"),
        {i: [q] for i, q in enumerate(QUBITS)},
    ),
    "shfqa": (
        ZiInstrument(kind="SHFQA", serial="DEV12951", options="SHFQA4"),
        {0: QUBITS},
    ),
}
"""Fake rack, with invented serials."""

PQSC = "DEV10901"
DRIVE_LO = 6.2e9
PROBE_LO = 6.8e9

VERSIONS = {name: version(name) for name in ("laboneq", "zhinst-core")}


def pytest_configure(config):
    config.addinivalue_line("markers", "feature(id): conformance feature from FEATURES")
    config.addinivalue_line("markers", "checkpoint(id): Elisa's checkpoint CP1..CP5")


def pytest_report_header(config):
    return " ".join(f"{name}=={v}" for name, v in VERSIONS.items())


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Write the feature or checkpoint id in the junit xml, like record_property."""
    for marker in item.iter_markers("feature"):
        feature = BY_ID[marker.args[0]]
        item.user_properties += [
            ("feature", feature["id"]),
            ("title", feature["title"]),
            ("group", feature["group"]),
            ("checkpoint", feature["checkpoint"]),
        ]
    for marker in item.iter_markers("checkpoint"):
        item.user_properties.append(("checkpoint", marker.args[0]))


def refused(message):
    """Strict xfail that holds only for this exact refusal."""
    return pytest.mark.xfail(
        raises=pytest.RaisesExc(NotImplementedError, match=f"^{re.escape(message)}$"),
        strict=True,
        reason=f"refused: {message}",
    )


def broken(*messages, reason=None):
    """Strict xfail for a call the driver accepts but LabOne Q rejects.

    Several messages: any of them, for a check whose wording differs between
    laboneq versions.
    """
    return pytest.mark.xfail(
        raises=pytest.RaisesExc(
            LabOneQException, match="|".join(re.escape(m) for m in messages)
        ),
        strict=True,
        reason=reason or f"LabOne Q error: {messages[0]}",
    )


@pytest.fixture(scope="session", autouse=True)
def versions(record_testsuite_property):
    for name, value in VERSIONS.items():
        record_testsuite_property(name, value)


@pytest.fixture(autouse=True)
def zi_kinds():
    ConfigKinds.extend([ZiConfigs])
    yield
    ConfigKinds.reset()


def configs(channels) -> dict:
    configs = {}
    for i, q in enumerate(QUBITS):
        configs[f"{q}/drive"] = ZiIqConfig(frequency=6.3e9 + i * 1e6, power_range=10)
        configs[f"{q}/probe"] = ZiIqConfig(frequency=7.0e9 + i * 1e6, power_range=5)
        configs[f"{q}/acquisition"] = ZiAcquisitionConfig(
            delay=200, smearing=0, power_range=-30
        )
        configs[f"{q}/flux"] = ZiDcConfig(offset=0.1, power_range=5)
    for id, ch in channels.items():
        if getattr(ch, "lo", None) is not None:
            lo = PROBE_LO if id.endswith("probe") else DRIVE_LO
            configs[ch.lo] = OscillatorConfig(frequency=lo, power=0)
    return configs


def natives(qubits) -> NativeGates:
    rx = Pulse(duration=48, amplitude=0.5, envelope=Gaussian(rel_sigma=1 / 6))
    rx90 = rx.model_copy(update={"amplitude": 0.25})
    probe = Pulse(duration=1024, amplitude=0.1, envelope=Rectangular())
    return NativeGates(
        single_qubit={
            q: SingleQubitNatives(
                RX=Native([(qubit.drive, rx)]),
                RX90=Native([(qubit.drive, rx90)]),
                MZ=Native([(qubit.acquisition, Readout.from_probe(probe))]),
            )
            for q, qubit in qubits.items()
        }
    )


@pytest.fixture
def platform():
    """Connected platform, made of one ZiController in emulation."""
    qubits = {q: Qubit.default(q) for q in QUBITS}
    channels = map_ports(RACK, qubits)
    for (qubit, is_probe), lo in infer_los(RACK).items():
        id = f"{qubit}/probe" if is_probe else f"{qubit}/drive"
        channels[id] = channels[id].model_copy(update={"lo": lo})
    controller = ZiController(
        address="127.0.0.1:8004",
        instruments={name: instrument for name, (instrument, _) in RACK.items()},
        pqsc=PQSC,
        channels=channels,
        emulation=True,
    )
    parameters = Parameters(
        settings=Settings(nshots=NSHOTS, relaxation_time=RELAXATION),
        configs=configs(channels),
        native_gates=natives(qubits),
    )
    platform = Platform(
        name="conformance",
        parameters=parameters,
        instruments={"zi": controller},
        qubits=qubits,
    )
    platform.connect()
    yield platform
    platform.disconnect()
