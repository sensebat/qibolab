"""Fixtures for the conformance suite: a small fake rack in LabOne Q emulation.

Each test carries ``@pytest.mark.feature("F01")`` or ``@pytest.mark.checkpoint("CP1")``.
The id, its title, its checkpoint and its effect tests go into the junit xml as test
properties, with the kind of test (``effect`` for the files in :data:`EFFECT_FILES`,
``contract`` otherwise); the laboneq and zhinst-core versions go in as suite properties.
A test with an xfail mark also gets ``xfail_kind`` (``not implemented yet`` or
``unclear message``, see :func:`xfail_kind`) and ``xfail_reason``.

Fixtures: ``platform`` (every qubit has the same RX and the same probe pulse) and
``platform_uneven`` (RX and probe lengths differ between qubits, see
:data:`UNEVEN_RX` and :data:`UNEVEN_PROBE`).
"""

import re
from fnmatch import fnmatch
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

RX_DURATION = 48
"""RX length of every qubit in ``platform``, ns."""
PROBE_DURATION = 1024
"""Probe length of every qubit in ``platform``, ns."""
UNEVEN_RX = {"q0": 40, "q1": 48}
"""RX length of each qubit in ``platform_uneven``, ns.

A qibocal rabi on both qubits reads each one out after its own RX (rabi/utils.py:327),
so the readouts start at different times.
"""
UNEVEN_PROBE = {"q0": 1024, "q1": 512}
"""Probe (and integration) length of each qubit in ``platform_uneven``, ns."""

EFFECT_FILES = ("test_effects_*.py", "test_references.py", "test_missing_infra.py")
"""Files whose tests are effect tests (glob patterns, in this folder)."""

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
    config.addinivalue_line("markers", "checkpoint(id): checkpoint CP1..CP5")


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
            ("effects", " ".join(feature["effects"])),
        ]
    for marker in item.iter_markers("checkpoint"):
        item.user_properties.append(("checkpoint", marker.args[0]))
    name = item.path.name
    effect = any(fnmatch(name, pattern) for pattern in EFFECT_FILES)
    item.user_properties.append(("kind", "effect" if effect else "contract"))
    xfail = item.get_closest_marker("xfail")
    if xfail is not None:
        reason = xfail.kwargs.get("reason", "")
        item.user_properties += [
            ("xfail_kind", xfail_kind(reason)),
            ("xfail_reason", reason),
        ]


XFAIL_MISSING = "not implemented yet"
XFAIL_UNCLEAR = "unclear message"


def xfail_kind(reason: str) -> str:
    """Kind of an xfail, from its reason: ``unclear message`` for :func:`unclear`,
    ``not implemented yet`` for every other one (:func:`missing`, :func:`refused`,
    :func:`broken`): a feature the driver lacks or that LabOne Q rejects as built.
    """
    return (
        XFAIL_UNCLEAR if reason.startswith("unclear error message") else XFAIL_MISSING
    )


def missing(reason: str, raises: type[BaseException] | None = None):
    """Strict xfail of an effect test for a feature the driver lacks.

    ``raises`` is the exception type the driver raises today (the xfail holds only
    for it). Without ``raises``, the call runs today and the test fails on one of its
    assertions (the xfail holds only for an ``AssertionError``): the driver is
    silently wrong. Either way, the test turns red (XPASS strict) the day the feature
    works.
    """
    return pytest.mark.xfail(
        raises=raises or AssertionError,
        strict=True,
        reason=f"{XFAIL_MISSING}: {reason}",
    )


def unclear(current: type, clear: str):
    """Strict xfail while the call fails with ``current``, whatever its message.

    For tests that check a message with ``expect_clear`` or ``refused_or_works`` of
    test_effects_configs_errors.py (which fail with an ``AssertionError`` starting
    with the name of the error raised). ``clear`` says
    what a clear message would say.
    """
    return pytest.mark.xfail(
        raises=pytest.RaisesExc(AssertionError, match=f"^{current.__name__}: "),
        strict=True,
        reason=f"unclear error message: {current.__name__} instead of {clear}",
    )


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


def natives(qubits, rx_duration=None, probe_duration=None) -> NativeGates:
    """RX, RX90 and MZ of each qubit.

    ``rx_duration`` and ``probe_duration`` map a qubit to its RX and probe lengths
    (ns); missing qubits get :data:`RX_DURATION` and :data:`PROBE_DURATION`.
    """
    rx_duration = rx_duration or {}
    probe_duration = probe_duration or {}

    def single(q, qubit):
        rx = Pulse(
            duration=rx_duration.get(q, RX_DURATION),
            amplitude=0.5,
            envelope=Gaussian(rel_sigma=1 / 6),
        )
        rx90 = rx.model_copy(update={"amplitude": 0.25})
        probe = Pulse(
            duration=probe_duration.get(q, PROBE_DURATION),
            amplitude=0.1,
            envelope=Rectangular(),
        )
        return SingleQubitNatives(
            RX=Native([(qubit.drive, rx)]),
            RX90=Native([(qubit.drive, rx90)]),
            MZ=Native([(qubit.acquisition, Readout.from_probe(probe))]),
        )

    return NativeGates(
        single_qubit={q: single(q, qubit) for q, qubit in qubits.items()}
    )


def create_platform(rx_duration=None, probe_duration=None) -> Platform:
    """Platform made of one ZiController in emulation, not connected.

    The natives are made by :func:`natives` with the given lengths.
    """
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
        native_gates=natives(qubits, rx_duration, probe_duration),
    )
    return Platform(
        name="conformance",
        parameters=parameters,
        instruments={"zi": controller},
        qubits=qubits,
    )


@pytest.fixture
def platform():
    """Connected platform; every qubit has the same RX and the same probe pulse."""
    platform = create_platform()
    platform.connect()
    yield platform
    platform.disconnect()


@pytest.fixture
def platform_uneven():
    """Connected platform whose RX and probe lengths differ between qubits.

    Same rack and configs as ``platform``; RX lengths from :data:`UNEVEN_RX`, probe
    and integration lengths from :data:`UNEVEN_PROBE`.
    """
    platform = create_platform(UNEVEN_RX, UNEVEN_PROBE)
    platform.connect()
    yield platform
    platform.disconnect()
