"""Effect tests for the features F48-F54, and checks of the missing-feature tooling.

F48-F54 are gaps of the driver found by the audit of ElStabilini/qibolab@zi_drivers
(e8757e58) that had no row in :mod:`features` (sources/review/unimplemented-audit).
Each test checks what LabOne Q must receive, or what qibolab must return, once the
feature works. While the driver lacks it, the test is a strict xfail
(:func:`conftest.missing`): with ``raises`` when the call fails today, without when
it runs silently wrong (then only an assertion keeps it xfail). Any change of
behaviour turns it red: XPASS when the feature works, a failure when the call starts
failing in another way (e.g. a clear refusal instead of a silently wrong run). Where the supported subset of the same call exists, it runs as
a ``control`` case with the same assertions and must pass, so the check itself is
shown to hold on a working call.

Tooling checks: the ``platform_uneven`` fixture and the xfail kinds written to the
junit xml.

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q); qibolab ``ExecutionParameters.results_shape``
(sources/research/driver-tests/README.md, item 2).
"""

import re

import laboneq.simple as lq
import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import Acquire, PlayPulse
from laboneq.dsl.experiment import Delay as LqDelay

from qibolab._core.components import (
    AcquisitionChannel,
    IqChannel,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiController,
    ZiInstrument,
    ZiIqConfig,
)
from qibolab._core.parameters import Parameters, Settings
from qibolab._core.platform import Platform
from qibolab._core.pulses import Delay, Pulse, Readout, Rectangular
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from . import effects
from .conftest import (
    NSHOTS,
    RELAXATION,
    UNEVEN_PROBE,
    UNEVEN_RX,
    missing,
    natives,
    unclear,
    xfail_kind,
)
from .effects import square
from .mutants import kills
from .test_conformance import mz, readout_of, updated

NS = effects.NS


# helpers


def start(cap, op) -> float:
    """Start of ``op`` in ns from the start of its sequence, on its own signal.

    Sum of the lengths of the operations before it on the same signal, in the real
    time loop: delays, plays and acquisitions (an acquisition holds its signal for
    its integration length). LabOne Q plays the operations of one signal in order;
    the sequence section reserves every signal, so all of them start together.
    """
    time = 0.0
    for other in cap.ops():
        if other.op is op.op:
            return time
        if not other.real_time or other.signal != op.signal:
            continue
        if isinstance(other.op, LqDelay):
            time += effects.values(other.op.time) / NS
        elif isinstance(other.op, PlayPulse):
            time += cap.duration(other)
        elif isinstance(other.op, Acquire):
            time += integration_length(cap, other)
    raise RuntimeError("Operation not found.")


def integration_length(cap, op) -> float:
    """Integration length of an acquisition in ns: its length, else its kernel."""
    length = effects.values(getattr(op.op, "length", None))
    if length is not None:
        return length / NS
    return len(cap.samples(op)) / cap.controller.sampling_rate


def readout_on(sequence, q):
    """The readout of qubit ``q`` in ``sequence``."""
    (readout,) = [ro for ch, ro in sequence.acquisitions if ch == f"{q}/acquisition"]
    return readout


def rabi(platform, qubits=("q0", "q1")) -> PulseSequence:
    """RX of each qubit, then its readout after its own RX, as qibocal rabi does
    (rabi/utils.py:327: ``Delay(duration=durations[q])`` on the readout)."""
    sequence = PulseSequence()
    for q in qubits:
        natives = platform.natives.single_qubit[q]
        channel, rx = natives.RX()[0]
        sequence.append((channel, rx))
        sequence.append((f"{q}/acquisition", Delay(duration=rx.duration)))
        sequence += natives.MZ()
    return sequence


# tooling


def test_uneven_fixture_lengths(platform_uneven):
    """``platform_uneven`` has the RX and probe lengths it documents.

    Pattern: none harvested; a check of the fixture, not an effect test.
    """
    natives = platform_uneven.natives.single_qubit
    for q, length in UNEVEN_RX.items():
        assert natives[q].RX()[0][1].duration == length
    for q, length in UNEVEN_PROBE.items():
        readout = natives[q].MZ()[0][1]
        assert readout.probe.duration == length
        assert readout.acquisition.duration == length
    assert len(set(UNEVEN_RX.values())) > 1
    assert len(set(UNEVEN_PROBE.values())) > 1


def test_xfail_kinds():
    """The xfail reasons map to the two kinds the report counts.

    Pattern: none harvested; a check of conftest, not an effect test.
    """
    marks = [
        missing("x").mark,
        missing("x", raises=KeyError).mark,
        unclear(KeyError, "y").mark,
    ]
    kinds = [xfail_kind(mark.kwargs["reason"]) for mark in marks]
    assert kinds == ["not implemented yet", "not implemented yet", "unclear message"]
    assert all(mark.kwargs["strict"] for mark in marks)
    assert xfail_kind("refused: Fast reset is not supported.") == "not implemented yet"
    assert xfail_kind("LabOne Q error: ...") == "not implemented yet"


# F23 with probes of different lengths (works today)


@pytest.mark.feature("F23")
@kills("readout_probe_swapped", "integration_length_from_probe")
def test_probes_of_different_length_on_one_line(platform_uneven):
    """Two qubits on one SHFQA channel, read out together with probes of different
    lengths: each probe plays its own length and each acquisition integrates it.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform_uneven, "q0", "q1")
    integrate = {"q0": 600, "q1": 300}
    for q, length in integrate.items():
        ro = readout_on(sequence, q)
        acquisition = ro.acquisition.model_copy(update={"duration": length})
        sequence = PulseSequence(
            [
                (ch, op.model_copy(update={"acquisition": acquisition}))
                if op is ro
                else (ch, op)
                for ch, op in sequence
            ]
        )

    cap = effects.play(platform_uneven, [sequence])

    for q, length in UNEVEN_PROBE.items():
        (probe,) = cap.plays(f"{q}/probe")
        assert cap.duration(probe) == pytest.approx(length), q
        (acquire,) = cap.acquires(f"{q}/acquisition")
        assert integration_length(cap, acquire) == pytest.approx(integrate[q]), q


# F48 readouts at different times


@pytest.mark.feature("F48")
@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param("platform", id="control"),
        pytest.param(
            "platform_uneven",
            id="uneven",
            marks=missing(
                "readouts that start at different times in one sequence",
                raises=LabOneQException,
            ),
        ),
    ],
)
def test_readouts_at_different_times(request, fixture):
    """Each qubit is read out right after its own RX, also when the RX lengths
    differ (qibocal rabi on several qubits; mid-circuit measurement).

    Working: each probe and its acquisition start when the RX of the same qubit
    ends, and each readout returns its result. Today, with RX lengths of 40 and 48
    ns, LabOne Q refuses the experiment: "all play signals must start at the same
    time" (one section per sequence, zi/controller.py:410).

    Pattern: qibocal rabi/utils.py:327 (readout delayed by the qubit's own RX);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    platform = request.getfixturevalue(fixture)
    sequence = rabi(platform)

    cap = effects.play(platform, [sequence])

    for q in ("q0", "q1"):
        rx = platform.natives.single_qubit[q].RX()[0][1].duration
        (drive,) = cap.plays(f"{q}/drive")
        assert start(cap, drive) == pytest.approx(0.0), q
        (probe,) = cap.plays(f"{q}/probe")
        assert start(cap, probe) == pytest.approx(rx), q
        (acquire,) = cap.acquires(f"{q}/acquisition")
        assert start(cap, acquire) == pytest.approx(rx), q
        assert cap.results[readout_on(sequence, q).id].shape == (2,), q


# F49 two probe waveforms for one qubit


@pytest.mark.feature("F49")
@pytest.mark.parametrize(
    "second",
    [
        pytest.param(1024, id="control"),
        pytest.param(
            512,
            id="two-lengths",
            marks=missing(
                "two probe waveforms for one qubit in one call",
                raises=LabOneQException,
            ),
        ),
    ],
)
def test_two_probe_lengths_one_qubit(platform, second):
    """Two sequences in one call read out q0 with probes of 1024 ns and ``second``
    ns: q0's probe plays each length in turn and integrates over it.

    Today, with two lengths, LabOne Q refuses the experiment: "SHFQA generators can
    only hold a single pulse waveform" (one sampled waveform per envelope and
    length, zi/pulse.py:20-28).

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    first = mz(platform)
    probe = Pulse(duration=second, amplitude=0.1, envelope=Rectangular())
    other = PulseSequence([("q0/acquisition", Readout.from_probe(probe))])

    cap = effects.play(platform, [first, other])

    lengths = [1024, second]
    plays = cap.plays("q0/probe")
    assert [cap.duration(op) for op in plays] == pytest.approx(lengths)
    acquires = cap.acquires("q0/acquisition")
    assert [integration_length(cap, op) for op in acquires] == pytest.approx(lengths)
    for sequence in (first, other):
        assert cap.results[readout_of(sequence).id].shape == (2,)


# F50 chirp


@pytest.mark.feature("F50")
@missing("pulse chirp is ignored: the pulse plays without it")
def test_chirp_reaches_waveform(platform):
    """A drive pulse with a ``chirp`` plays chirped.

    The phase of the waveform LabOne Q plays changes along the pulse
    (here by about pi over 200 ns at 25 kHz/ns), or a LabOne Q sweep drives the
    drive frequency during it. Today the chirp is not read (zi/pulse.py:52-58) and
    the pulse plays with a constant phase.

    Pattern: ElStabilini/qibolab@zi_drivers (e8757e58) qm driver,
    qm/program/instructions.py (``qua.play(..., chirp=...)`` from ``Pulse.chirp``).
    """
    pulse = square(duration=200).model_copy(update={"chirp": (25000, "Hz/nsec")})
    sequence = PulseSequence(
        [("q0/drive", pulse), ("q0/acquisition", Delay(duration=200))]
    ) + mz(platform)

    cap = effects.play(platform, [sequence])

    (op,) = cap.plays("q0/drive")
    phase = np.unwrap(np.angle(cap.waveform(op)))
    swept = [
        d for loop in cap.sweeps() for p in loop.parameters for d in cap.driven_by(p)
    ]
    assert np.ptp(phase) > 1.0 or any(
        d.channel == "q0/drive" and d.quantity == "frequency" for d in swept
    ), f"phase span {np.ptp(phase):.3g} rad: no chirp"


# F51 durations off the sample grid


@pytest.mark.feature("F51")
@pytest.mark.parametrize(
    "duration",
    [
        pytest.param(40.0, id="control"),
        pytest.param(
            40.4,
            id="off-grid",
            marks=missing("durations off the sample grid are rounded down"),
        ),
    ],
)
def test_duration_off_sample_grid(platform, duration):
    """A drive pulse whose length is not a whole number of samples is refused with an
    error naming the duration or the sample grid, or plays within half a sample of
    its length, and the readout after it does not start before it ends.

    Refusing is accepted as in test_missing_pulses.py::
    test_off_grid_duration_refused_or_exact (same feature F51). Today 40.4 ns (80.8
    samples at 2 GSa/s) plays 80 samples, 40.0 ns (``int(duration *
    sampling_rate)``, zi/pulse.py:27), while the delay before the readout stays
    40.4 ns.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests
    (refuse with a clear message); zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    sequence = PulseSequence(
        [
            ("q0/drive", square(duration=duration)),
            ("q0/acquisition", Delay(duration=duration)),
        ]
    ) + mz(platform)

    try:
        cap = effects.play(platform, [sequence])
    except (ValueError, NotImplementedError, LabOneQException) as error:
        message = f"{type(error).__name__}: {error}"
        assert re.search(r"sampl|grid|[Dd]uration", str(error)), message
        return

    half = 0.5 / platform.sampling_rate
    (drive,) = cap.plays("q0/drive")
    played = cap.duration(drive)
    assert abs(played - duration) <= half + 1e-9, f"{played} ns played"
    (probe,) = cap.plays("q0/probe")
    assert start(cap, probe) >= start(cap, drive) + played - 1e-9


# F52 instrument kinds


def _shfqc():
    instruments = {
        "shfqc": ZiInstrument(kind="SHFQC", serial="DEV12991", options="SHFQC/QC6CH")
    }
    channels = {
        "q0/drive": IqChannel(device="shfqc", path="0", lo="shfqc/sg0/lo"),
        "q0/probe": IqChannel(device="shfqc", path="0", lo="shfqc/qa0/lo"),
        "q0/acquisition": AcquisitionChannel(
            device="shfqc", path="0", probe="q0/probe"
        ),
    }
    ports = {
        "q0/drive": ("shfqc", {"SGCHANNELS/0/OUTPUT"}),
        "q0/probe": ("shfqc", {"QACHANNELS/0/OUTPUT"}),
        "q0/acquisition": ("shfqc", {"QACHANNELS/0/INPUT"}),
    }
    return instruments, None, channels, ports


def _shfqa_readout():
    instruments = {
        "shfqa": ZiInstrument(kind="SHFQA", serial="DEV12951", options="SHFQA4")
    }
    channels = {
        "q0/probe": IqChannel(device="shfqa", path="0", lo="shfqa/o0/lo"),
        "q0/acquisition": AcquisitionChannel(
            device="shfqa", path="0", probe="q0/probe"
        ),
    }
    ports = {
        "q0/probe": ("shfqa", {"QACHANNELS/0/OUTPUT"}),
        "q0/acquisition": ("shfqa", {"QACHANNELS/0/INPUT"}),
    }
    return instruments, channels, ports


def _hdawg_iq_drive():
    instruments, channels, ports = _shfqa_readout()
    instruments["hdawg"] = ZiInstrument(
        kind="HDAWG", serial="DEV8901", options="HDAWG8"
    )
    channels["q0/drive"] = IqChannel(device="hdawg", path="0", lo="ext/lo")
    ports["q0/drive"] = ("hdawg", {"SIGOUTS/0", "SIGOUTS/1"})
    return instruments, "DEV10901", channels, ports


def _uhfqa_readout():
    instruments = {
        "hdawg": ZiInstrument(kind="HDAWG", serial="DEV8901", options="HDAWG8"),
        "uhfqa": ZiInstrument(kind="UHFQA", serial="DEV2901"),
    }
    channels = {
        "q0/drive": IqChannel(device="hdawg", path="0", lo="ext/lo"),
        "q0/probe": IqChannel(device="uhfqa", path="0", lo="ext/probe_lo"),
        "q0/acquisition": AcquisitionChannel(
            device="uhfqa", path="0", probe="q0/probe"
        ),
    }
    ports = {
        "q0/drive": ("hdawg", {"SIGOUTS/0", "SIGOUTS/1"}),
        "q0/probe": ("uhfqa", {"SIGOUTS/0", "SIGOUTS/1"}),
        "q0/acquisition": ("uhfqa", None),
    }
    return instruments, None, channels, ports


def _rack_platform(instruments, pqsc, channels) -> Platform:
    qubits = {"q0": Qubit.default("q0")}
    controller = ZiController(
        address="127.0.0.1:8004",
        instruments=instruments,
        pqsc=pqsc,
        channels=channels,
        emulation=True,
    )
    configs = {
        "q0/drive": ZiIqConfig(frequency=6.3e9, power_range=10),
        "q0/probe": ZiIqConfig(frequency=7.0e9, power_range=5),
        "q0/acquisition": ZiAcquisitionConfig(delay=200, smearing=0, power_range=-30),
    }
    for id in ("q0/drive", "q0/probe"):
        lo = 6.8e9 if id.endswith("probe") else 6.2e9
        configs[channels[id].lo] = OscillatorConfig(frequency=lo, power=0)
    parameters = Parameters(
        settings=Settings(nshots=NSHOTS, relaxation_time=RELAXATION),
        configs=configs,
        native_gates=natives(qubits),
    )
    return Platform(
        name="instruments",
        parameters=parameters,
        instruments={"zi": controller},
        qubits=qubits,
    )


def _ports(setup, signal) -> tuple[set[str], set[str]]:
    """Instrument uids and ports the logical signal ``signal`` is connected to."""
    found = [
        (instrument.uid, connection.local_port)
        for instrument in setup.instruments
        for connection in instrument.connections
        if connection.remote_path.endswith(f"/{signal}")
        or connection.remote_path == signal
    ]
    return {uid for uid, _ in found}, {port for _, port in found}


@pytest.mark.feature("F52")
@pytest.mark.parametrize(
    "rack",
    [
        pytest.param(
            _shfqc,
            id="SHFQC",
            marks=missing("SHFQC instruments", raises=KeyError),
        ),
        pytest.param(
            _hdawg_iq_drive,
            id="HDAWG-IQ",
            marks=missing("IQ channels on HDAWG outputs", raises=ValueError),
        ),
        pytest.param(
            _uhfqa_readout,
            id="UHFQA",
            marks=missing("UHFQA instruments", raises=KeyError),
        ),
    ],
)
def test_instrument_kinds(rack):
    """A drive and a readout on an SHFQC, on HDAWG IQ outputs, or on a UHFQA (with
    an HDAWG drive) reach those instruments, and the readout returns its result.

    Assumed wiring: :attr:`Channel.device` names the instrument and
    :attr:`Channel.path` the port, as for the supported kinds; on an SHFQC the probe
    channel (named by an :class:`AcquisitionChannel`) is a QA channel and the drive an
    SG channel; an HDAWG IQ channel at port 0 uses outputs 0 and 1. Today the device
    setup refuses these kinds (zi/controller.py:53,60,137-153). SHFPPC (TWPA pump) is
    not tested: qibolab has no way to say which pump channel serves which line.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; LabOne Q
    device setup connections (``create_connection(to_signal, ports)``).
    """
    instruments, pqsc, channels, ports = rack()
    platform = _rack_platform(instruments, pqsc, channels)
    zi = platform.instruments["zi"]
    zi.connect()
    try:
        sequence = PulseSequence(
            [("q0/drive", square()), ("q0/acquisition", Delay(duration=40))]
        ) + mz(platform)
        cap = effects.play(platform, [sequence])
        setup = cap.setup
        for id, (uid, expected) in ports.items():
            uids, found = _ports(setup, id)
            assert uids == {uid}, f"{id} on {uids}"
            if expected is not None:
                assert found == expected, f"{id} on {found}"
        played = [op for op in cap.plays() if op.signal == "q0/drive"]
        assert len(played) == 1
        assert cap.results[readout_of(sequence).id].shape == (2,)
    finally:
        zi.disconnect()


# F53 default execute options


@pytest.mark.feature("F53")
@missing(
    "platform.execute with the default options (DISCRIMINATION, SINGLESHOT)",
    raises=NotImplementedError,
)
def test_default_execute_options(platform):
    """``platform.execute`` with no options runs with the qibolab defaults:
    DISCRIMINATION and SINGLESHOT, the nshots of the settings, one state per shot.

    Today DISCRIMINATION is refused (zi/controller.py:383), so the defaults are.

    Pattern: qibolab ``ExecutionParameters`` defaults
    (core/execution_parameters.py:76-78) and ``Platform.execute`` docstring;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    # a calibrated platform, as test_conformance.py::test_acquisition_type[F17]
    platform.parameters.configs.update(
        updated(platform, q0__acquisition={"threshold": 0.0, "iq_angle": 0.0})
    )
    with effects.capture(platform.instruments["zi"]) as cap:
        results = platform.execute([sequence])

    assert cap.acquisition_type == lq.AcquisitionType.DISCRIMINATION
    assert cap.averaging_mode == lq.AveragingMode.SINGLE_SHOT
    assert cap.nshots == platform.settings.nshots
    defaults = ExecutionParameters(nshots=platform.settings.nshots)
    assert results[readout_of(sequence).id].shape == defaults.results_shape([])


# F54 parallel near-time and real-time sweepers


@pytest.mark.feature("F54")
@missing(
    "parallel sweepers mixing near time and real time",
    raises=NotImplementedError,
)
def test_parallel_drive_and_probe_frequency(platform):
    """One parallel group sweeps the drive frequency (real time) and the probe
    frequency (near time) of q0 together: both oscillators step through their
    values in one LabOne Q sweep, and the results have one axis of that length.

    Today: "Sweeping in parallel in near time and in real time is not supported."
    (zi/sweep.py:56-59).

    Pattern: ElStabilini/qibolab@zi_drivers (e8757e58)
    tests/instruments/zi/test_play.py:172-174 (the same group, asserted as refused);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = PulseSequence(
        [("q0/drive", square()), ("q0/acquisition", Delay(duration=40))]
    ) + mz(platform)
    steps = np.array([0.0, 1e6, 2e6])
    drive = platform.config("q0/drive").frequency + steps
    probe = platform.config("q0/probe").frequency + steps
    group = [
        Sweeper(parameter=Parameter.frequency, values=drive, channels=["q0/drive"]),
        Sweeper(parameter=Parameter.frequency, values=probe, channels=["q0/probe"]),
    ]

    cap = effects.play(platform, [sequence], [group])

    loops = {}
    for n, loop in enumerate(cap.sweeps()):
        for parameter in loop.parameters:
            for driven in cap.driven_by(parameter):
                if driven.quantity == "frequency":
                    loops.setdefault(driven.channel, set()).add(n)
    assert loops.get("q0/drive") and loops.get("q0/probe"), loops
    assert loops["q0/drive"] == loops["q0/probe"], f"stepped apart: {loops}"
    assert np.allclose(cap.frequency("q0/drive"), drive)
    assert np.allclose(cap.frequency("q0/probe"), probe)
    shape = ExecutionParameters(**effects.DEFAULTS).results_shape([group])
    assert cap.results[readout_of(sequence).id].shape == shape
