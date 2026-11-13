"""Effect tests for the sweeps, acquisition types, averaging modes and execution
options that ElStabilini/qibolab@zi_drivers (e8757e58) lacks.

Features: F05-F12 (flux amplitude, duration, interpolated duration, relative phase,
virtual Z phase and flux offset sweeps; F07 is audit item X23), F13 with mixed
timing (audit X20), F17 DISCRIMINATION, F18 RAW, F21 SEQUENTIAL, F45 fast reset, F53
default execute options (audit X01) and F03 given ``Readout.probe`` (audit G1).
Audit: sources/review/unimplemented-audit.

Each test checks what LabOne Q must receive, or what qibolab must return, once the
feature works (read through :mod:`effects`). While the driver lacks it, the test is
a strict xfail (:func:`conftest.missing`) that holds only for today's refusal: a
``NotImplementedError`` with the exact message of the driver (:func:`today`). Any
other error, and any pass, turns the test red. F03 given ``Readout.probe`` runs
silently wrong today, so its xfail holds only for an assertion of the test. Where a
supported version of the same call exists, it runs as a ``control`` case with the
same assertions and must pass.

Results seam: the emulator returns placeholder data, so tests that check the values
qibolab returns replace the LabOne Q results with known data
(:func:`injected`), as test_effects_sweeps.py does.

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q); qibolab ``ExecutionParameters.results_shape``
(sources/research/driver-tests/README.md, item 2).
"""

import re
from contextlib import contextmanager

import laboneq.simple as lq
import numpy as np
import pytest
from laboneq.controller.devices.qachannel import _calc_theoretical_assignment_vec
from laboneq.dsl.experiment import (
    Case,
    Match,
    PlayPulse,
    PulseFunctional,
    Section,
)

from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.pulses import Delay, VirtualZ
from qibolab._core.pulses.modulation import demodulate
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from . import effects
from .conftest import missing
from .effects import drives_then_mz, iq, square
from .test_conformance import mz, readout_of, rx, updated
from .test_missing_infra import integration_length, start

NS = effects.NS


def today(message: str):
    """Today's refusal: a ``NotImplementedError`` with exactly ``message``."""
    return pytest.RaisesExc(NotImplementedError, match=f"^{re.escape(message)}$")


# helpers


def shape(sweepers=(), samples=-1, **options) -> tuple[int, ...]:
    """``results_shape`` of the call's execution parameters (over effects.DEFAULTS)."""
    parameters = ExecutionParameters(**(effects.DEFAULTS | options))
    return parameters.results_shape(list(sweepers), samples)


def one(ops, what):
    """The one operation in ``ops``, failing as a test if there is not exactly one."""
    assert len(ops) == 1, f"{len(ops)} operations on {what}, not 1"
    return ops[0]


def pulses(cap, channel):
    """Plays of a pulse on ``channel`` (an oscillator phase step alone is not one)."""
    return [op for op in cap.plays(channel) if op.op.pulse is not None]


def swept(cap, x) -> np.ndarray:
    """Values a LabOne Q quantity takes, failing as a test if it is not swept."""
    assert hasattr(x, "uid"), f"{x!r} is not swept"
    return np.asarray(effects.values(x))


def same_sweep(cap, a, b) -> bool:
    """Whether one LabOne Q sweep steps the parameters ``a`` and ``b`` together."""
    loops = [cap.sweep_of(a), cap.sweep_of(b)]
    uids = [{p.uid for p in loop.parameters} for loop in loops]
    return uids[0] == uids[1]


def swept_delays(cap, channel):
    """Delays on ``channel`` whose time a LabOne Q sweep drives."""
    return [op for op in cap.delays(channel) if hasattr(op.op.time, "uid")]


def frequency_loops(cap) -> dict:
    """Channel -> indices of the LabOne Q sweeps that drive its frequency."""
    loops = {}
    for n, loop in enumerate(cap.sweeps()):
        for parameter in loop.parameters:
            for driven in cap.driven_by(parameter):
                if driven.quantity == "frequency":
                    loops.setdefault(driven.channel, set()).add(n)
    return loops


class _Injected:
    """LabOne Q results whose data are ``data(shape)`` for every handle."""

    def __init__(self, results, data):
        self._results = results
        self._data = data

    def __getattr__(self, name):
        return getattr(self._results, name)

    def get_data(self, handle):
        return self._data(np.shape(self._results.get_data(handle)))


@contextmanager
def injected(platform, data):
    """Replace the data of every LabOne Q handle by ``data(shape)``."""
    session = platform.instruments["zi"].session
    run = session.run
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session, "run", lambda *a, **k: _Injected(run(*a, **k), data))
        yield


def states(shape) -> np.ndarray:
    """State 0 or 1 at each point, alternating, as LabOne Q returns them (complex)."""
    n = int(np.prod(shape))
    return (np.arange(n) % 2).reshape(shape).astype(complex)


def sections(node):
    yield node
    for child in getattr(node, "children", None) or []:
        if isinstance(child, Section):
            yield from sections(child)


# F03 amplitude sweep given Readout.probe (G1)


@pytest.mark.feature("F03")
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("readout", id="control"),
        pytest.param(
            "probe",
            id="probe",
            marks=missing(
                "amplitude sweep given Readout.probe",
                # zi_drivers e8757e58: nothing swept; faabbb45: KeyError
                raises=(AssertionError, KeyError),
            ),
        ),
    ],
)
def test_probe_amplitude_sweep_given_probe(platform, target):
    """An amplitude sweeper whose ``pulses`` hold the readout's probe pulse plays
    q0's probe at the sweeper values, as when they hold the readout.

    Audit G1: the driver looks up swept amplitudes by the id of the readout
    (zi/pulse.py:63). Up to zi_drivers e8757e58 nothing is swept and no error is
    raised: the probe plays its fixed amplitude. From faabbb45 the call fails with a
    ``KeyError`` (the probe's id): zi/sweep.py looks up the channel of each swept
    pulse among the operations of the sequences, where only the readout is.

    Pattern: qibocal resonator_punchout (resonator_spectroscopies/
    resonator_punchout.py:123, qibocal b559e4a7) sweeps the readout;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    readout = readout_of(sequence)
    values = np.array([0.05, 0.2, 0.4])
    pulse = readout if target == "readout" else readout.probe
    sweeper = Sweeper(parameter=Parameter.amplitude, values=values, pulses=[pulse])

    cap = effects.play(platform, [sequence], [[sweeper]])

    probe = one(cap.plays("q0/probe"), "q0/probe")
    amplitude = np.asarray(cap.amplitude(probe))
    assert amplitude.shape == values.shape, f"amplitude {amplitude}, not swept"
    np.testing.assert_allclose(amplitude, values, rtol=1e-12)
    assert cap.results[readout.id].shape == shape([[sweeper]])


# F05 amplitude sweep on a flux pulse


@pytest.mark.feature("F05")
@pytest.mark.parametrize(
    "channel",
    [
        pytest.param("q0/drive", id="control"),
        pytest.param(
            "q0/flux",
            id="flux",
            marks=missing(
                "pulses on a flux channel",
                raises=today("Pulse on channel 'q0/flux' is not supported."),
            ),
        ),
    ],
)
def test_flux_amplitude_sweep(platform, channel):
    """An amplitude sweeper on a flux pulse plays it on q0/flux (the HDAWG output)
    at the sweeper values, for its length, before the readout.

    Today flux pulses are refused (zi/pulse.py:52, 71-73).

    Pattern: qibocal chevron (two_qubit_interaction/chevron/chevron.py:189) and
    flux_amplitude_frequency (flux_dependence/flux_amplitude_frequency.py:231),
    qibocal b559e4a7; zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    pulse = square(duration=60, amplitude=0.2)
    sequence = PulseSequence(
        [(channel, pulse), ("q0/acquisition", Delay(duration=pulse.duration))]
    ) + mz(platform)
    values = np.array([0.1, 0.3, 0.5])
    sweeper = Sweeper(parameter=Parameter.amplitude, values=values, pulses=[pulse])

    cap = effects.play(platform, [sequence], [[sweeper]])

    played = one(pulses(cap, channel), channel)
    np.testing.assert_allclose(cap.amplitude(played), values, rtol=1e-12)
    assert cap.duration(played) == pytest.approx(60)
    probe = one(cap.plays("q0/probe"), "q0/probe")
    assert start(cap, probe) == pytest.approx(start(cap, played) + 60)
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F06, F08 duration sweeps on a pulse and the delay before the readout


@pytest.mark.parametrize(
    "channel",
    [
        pytest.param(
            "q0/drive",
            id="F06-drive",
            marks=[
                pytest.mark.feature("F06"),
                missing(
                    "duration sweep on a drive pulse and a delay",
                    raises=today("Sweeping duration is not supported."),
                ),
            ],
        ),
        pytest.param(
            "q0/flux",
            id="F08-flux",
            marks=[
                pytest.mark.feature("F08"),
                missing(
                    "duration sweep on a flux pulse and a delay",
                    raises=today("Sweeping duration is not supported."),
                ),
            ],
        ),
    ],
)
def test_duration_sweep(platform, channel):
    """One duration sweeper over a pulse and the delay on q0's readout line: at each
    step the pulse plays for the swept length and the readout starts when it ends.

    LabOne Q gets the pulse length and the delay time driven by one sweep, with the
    sweeper values. Today: "Sweeping duration is not supported." (zi/sweep.py:52-55).

    Pattern: qibocal rabi_length_signal (rabi/length_signal.py:97) and chevron
    (two_qubit_interaction/chevron/chevron.py:194), qibocal b559e4a7: one duration
    sweeper on the pulse and the delay; zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    pulse = square(duration=40, amplitude=0.2)
    delay = Delay(duration=pulse.duration)
    sequence = PulseSequence([(channel, pulse), ("q0/acquisition", delay)]) + mz(
        platform
    )
    values = np.array([16.0, 32.0, 48.0, 64.0])
    sweeper = Sweeper(
        parameter=Parameter.duration, values=values, pulses=[pulse, delay]
    )

    cap = effects.play(platform, [sequence], [[sweeper]])

    played = one(pulses(cap, channel), channel)
    np.testing.assert_allclose(swept(cap, played.op.length) / NS, values)
    assert cap.amplitude(played) == pytest.approx(0.2)
    waits = swept_delays(cap, "q0/acquisition")
    assert waits, "the delay on q0/acquisition is not swept"
    assert all(same_sweep(cap, played.op.length, w.op.time) for w in waits)
    probe = one(cap.plays("q0/probe"), "q0/probe")
    acquire = one(cap.acquires("q0/acquisition"), "q0/acquisition")
    for op in (probe, acquire):
        np.testing.assert_allclose(
            np.broadcast_to(start(cap, op), values.shape), values, err_msg=op.channel
        )
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F07 duration sweep on a delay (X23)


@pytest.mark.feature("F07")
@missing(
    "duration sweep on a delay",
    raises=today("Sweeping duration is not supported."),
)
def test_delay_sweep(platform):
    """A duration sweeper on a delay between the RX and the readout (T1): at each
    step the readout of q0 starts the swept time after the RX ends; the RX itself
    does not move.

    Audit X23. Today: "Sweeping duration is not supported." (zi/sweep.py:52-55).

    Pattern: qibocal t1 (coherence/t1.py:63) and ramsey (ramsey/acquisition.py:213),
    qibocal b559e4a7: a duration sweeper on a ``Delay`` before the readout;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    pulse = rx(platform)
    wait = Delay(duration=16)
    sequence = PulseSequence(
        [
            ("q0/drive", pulse),
            ("q0/acquisition", Delay(duration=pulse.duration)),
            ("q0/acquisition", wait),
        ]
    ) + mz(platform)
    values = np.array([16.0, 416.0, 816.0, 1216.0])
    sweeper = Sweeper(parameter=Parameter.duration, values=values, pulses=[wait])

    cap = effects.play(platform, [sequence], [[sweeper]])

    drive = one(pulses(cap, "q0/drive"), "q0/drive")
    assert start(cap, drive) == pytest.approx(0.0)
    assert cap.duration(drive) == pytest.approx(pulse.duration)
    waits = swept_delays(cap, "q0/acquisition")
    assert waits, "no delay on q0/acquisition is swept"
    for w in waits:
        np.testing.assert_allclose(swept(cap, w.op.time) / NS, values)
    expected = pulse.duration + values
    probe = one(cap.plays("q0/probe"), "q0/probe")
    acquire = one(cap.acquires("q0/acquisition"), "q0/acquisition")
    for op in (probe, acquire):
        np.testing.assert_allclose(
            np.broadcast_to(start(cap, op), values.shape), expected
        )
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F09 interpolated duration sweep


@pytest.mark.feature("F09")
@missing(
    "interpolated duration sweep on a drive pulse",
    raises=today("Sweeping duration_interpolated is not supported."),
)
def test_interpolated_duration_sweep(platform):
    """An interpolated duration sweeper on the RX (Gaussian) plays it at each swept
    length with its envelope stretched to that length, not cut or padded.

    LabOne Q gets the play length driven by the sweep with the sweeper values, and
    a pulse functional evaluated at each length (fixed samples of one length cannot
    be stretched). Today: "Sweeping duration_interpolated is not supported."
    (zi/sweep.py:52-55).

    Pattern: qibocal rabi_length_signal with ``interpolated_sweeper``
    (rabi/length_signal.py:91, rabi/utils.py:364, qibocal b559e4a7: RX, align,
    readout); zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    pulse = rx(platform)
    sequence = PulseSequence([("q0/drive", pulse)])
    sequence.align(["q0/drive", "q0/acquisition"])
    sequence += mz(platform)
    values = np.array([16.0, 32.0, 48.0, 64.0])
    sweeper = Sweeper(
        parameter=Parameter.duration_interpolated, values=values, pulses=[pulse]
    )

    cap = effects.play(platform, [sequence], [[sweeper]])

    drive = one(pulses(cap, "q0/drive"), "q0/drive")
    np.testing.assert_allclose(swept(cap, drive.op.length) / NS, values)
    assert cap.amplitude(drive) == pytest.approx(pulse.amplitude)
    assert isinstance(drive.op.pulse, PulseFunctional), (
        f"{type(drive.op.pulse).__name__}: envelope not evaluated per length"
    )
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F10 relative phase sweep


@pytest.mark.feature("F10")
@missing(
    "relative phase sweep on a drive pulse",
    raises=today("Sweeping relative_phase is not supported."),
)
def test_relative_phase_sweep(platform):
    """A relative phase sweeper on the RX plays it at each swept phase, with its
    amplitude and length kept.

    LabOne Q gets the play phase driven by the sweep with the sweeper values (rad).
    Today: "Sweeping relative_phase is not supported." (zi/sweep.py:52-55); a fixed
    ``relative_phase`` reaches the play phase (zi/pulse.py:57).

    Pattern: qibocal qubit_vz (flux_dependence/qubit_vz.py:121, qibocal b559e4a7);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    pulse = rx(platform)
    sequence = drives_then_mz(platform, {"q0": [pulse]})
    values = np.linspace(0, np.pi, 5)
    sweeper = Sweeper(parameter=Parameter.relative_phase, values=values, pulses=[pulse])

    cap = effects.play(platform, [sequence], [[sweeper]])

    drive = one(pulses(cap, "q0/drive"), "q0/drive")
    np.testing.assert_allclose(swept(cap, drive.op.phase), values, atol=1e-12)
    assert cap.amplitude(drive) == pytest.approx(pulse.amplitude)
    assert cap.duration(drive) == pytest.approx(pulse.duration)
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F11 phase sweep on a virtual Z


def phases(cap, channel) -> list:
    """Phase of each pulse played on ``channel``, frame steps included.

    The play phase plus every ``increment_oscillator_phase`` on the channel's signal
    up to that play (LabOne Q applies the increment of a play before its pulse).
    """
    found = []
    total = 0.0
    for op in cap.ops(channel, PlayPulse):
        if op.op.set_oscillator_phase is not None:
            total = effects.values(op.op.set_oscillator_phase)
        if op.op.increment_oscillator_phase is not None:
            total = total + effects.values(op.op.increment_oscillator_phase)
        if op.op.pulse is not None:
            found.append(total + cap.phase(op))
    return found


@pytest.mark.feature("F11")
@missing(
    "phase sweep on a virtual Z",
    raises=today("Sweeping phase is not supported."),
)
def test_virtual_z_phase_sweep(platform):
    """RX, a virtual Z with a swept phase, RX: the second RX plays shifted by the
    swept phase, the first one is not shifted.

    In LabOne Q, either an oscillator phase increment or the play phase of the
    second RX takes the sweeper values. Sign: a virtual Z of phase p adds p to the
    phase of the pulses after it, as in the qibolab emulator (emulator.py:400,
    ``cumulative_phase += pulse.phase``, then hamiltonians.py:230). Today: "Sweeping
    phase is not supported." (zi/sweep.py:52-55); VirtualZ is also refused (F27).

    Pattern: qibocal standard_rb_sweeper (randomized_benchmarking/
    standard_rb_sweeper.py:132) and virtual_z_phases (two_qubit_interaction/
    virtual_z_phases.py:251), qibocal b559e4a7; zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    vz = VirtualZ(phase=0.0)
    first, second = rx(platform), rx(platform)
    sequence = drives_then_mz(platform, {"q0": [first, vz, second]})
    values = np.linspace(0, np.pi, 5)
    sweeper = Sweeper(parameter=Parameter.phase, values=values, pulses=[vz])

    cap = effects.play(platform, [sequence], [[sweeper]])

    found = phases(cap, "q0/drive")
    assert len(found) == 2, f"{len(found)} pulses on q0/drive, not 2"
    np.testing.assert_allclose(np.atleast_1d(found[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.broadcast_to(found[1], values.shape), values, atol=1e-12
    )
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F12 offset sweep on a flux channel


@pytest.mark.feature("F12")
@missing(
    "offset sweep on a flux channel",
    raises=today("Sweeping offset is not supported."),
)
def test_flux_offset_sweep(platform):
    """An offset sweeper on q0/flux sets the DC offset of q0's HDAWG output to each
    sweeper value (V) in turn; q1/flux keeps its configured offset.

    LabOne Q gets the voltage offset of the q0/flux signal driven by a sweep.
    Today: "Sweeping offset is not supported." (zi/sweep.py:52-55); a fixed offset
    reaches the voltage offset (F38).

    Pattern: qibocal qubit_flux_dependence (flux_dependence/
    qubit_flux_dependence.py:149) and resonator_flux_dependence (flux_dependence/
    resonator_flux_dependence.py:129), qibocal b559e4a7: absolute offsets around the
    configured one; zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    offset = platform.config("q0/flux").offset
    values = offset + np.linspace(-0.05, 0.05, 5)
    sweeper = Sweeper(parameter=Parameter.offset, values=values, channels=["q0/flux"])

    cap = effects.play(platform, [sequence], [[sweeper]])

    signal = cap.signal_of("q0/flux")
    calibration = cap.calibration("q0/flux")
    np.testing.assert_allclose(swept(cap, calibration.voltage_offset), values)
    driven = {
        (d.channel, d.quantity)
        for loop in cap.sweeps()
        for p in loop.parameters
        for d in cap.driven_by(p)
    }
    assert ("q0/flux", "offset") in driven, f"{signal}: offset not swept"
    assert cap.offset("q1/flux") == pytest.approx(platform.config("q1/flux").offset)
    assert cap.results[readout_of(sequence).id].shape == shape([[sweeper]])


# F13 parallel sweepers with mixed timing (X20)


@pytest.mark.feature("F13")
@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(("q0/probe", "q1/probe"), id="control"),
        pytest.param(
            ("q0/drive", "q1/probe"),
            id="mixed",
            marks=missing(
                "parallel sweepers mixing near time and real time",
                raises=today(
                    "Sweeping in parallel in near time and in real time is not "
                    "supported."
                ),
            ),
        ),
    ],
)
def test_parallel_sweepers_mixed_timing(platform, channels):
    """One parallel group, one frequency sweeper per qubit: q0's drive (real time
    on the hardware oscillator) and q1's probe (near time). Both step together
    through their own values in one LabOne Q sweep, and both readouts get one axis
    of that length.

    Audit X20 (F54 is the same group on one qubit). Today: "Sweeping in parallel in
    near time and in real time is not supported." (zi/sweep.py:56-59).

    Pattern: qibocal resonator_spectroscopy (resonator_spectroscopies/
    resonator_spectroscopy.py:204, qibocal b559e4a7: one parallel group, one
    sweeper per qubit); zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = drives_then_mz(platform, {"q0": [square()], "q1": []})
    steps = np.array([0.0, 1e6, 2e6])
    group = [
        Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(ch).frequency + steps,
            channels=[ch],
        )
        for ch in channels
    ]

    cap = effects.play(platform, [sequence], [group])

    loops = frequency_loops(cap)
    first, second = channels
    assert loops.get(first) and loops.get(second), loops
    assert loops[first] == loops[second], f"stepped apart: {loops}"
    for sweeper, ch in zip(group, channels):
        np.testing.assert_allclose(cap.frequency(ch), sweeper.values, err_msg=ch)
    for _, readout in sequence.acquisitions:
        assert cap.results[readout.id].shape == shape([group])


# F17 DISCRIMINATION


@pytest.mark.feature("F17")
@missing(
    "state discrimination (DISCRIMINATION)",
    raises=today("AcquisitionType.DISCRIMINATION is not supported."),
)
def test_discrimination(platform):
    """DISCRIMINATION, single shots: LabOne Q discriminates q0 with the threshold
    and the IQ angle of the q0/acquisition config, and qibolab returns the state
    (0 or 1) LabOne Q measured at each shot.

    LabOne Q compares Re(sum of kernel x signal) with the threshold of the signal
    calibration, and its 2-state assignment vector (controller/devices/
    qachannel.py, ``_calc_theoretical_assignment_vec``) gives the state above the
    threshold: state 0 in laboneq 26.4.0 (zhinst-utils 0.7.1), state 1 in 26.10.1
    (zhinst-utils 0.7.2 swaps the states of each weight). ``sign`` is -1 or +1
    accordingly. qibolab decides state 1 when
    cos(a) I - sin(a) Q = Re((I + iQ) e^(ia)) exceeds ``threshold``, with
    ``a = iq_angle`` and I + iQ the INTEGRATION result (unit kernel). So the
    kernel must be a constant c with arg(sign c) = a and threshold / |c| =
    sign ``threshold``. The sign of arg(sign c) is accepted both ways: whether
    LabOne Q conjugates the kernel was not checked (open point). Today:
    "AcquisitionType.DISCRIMINATION is not supported." (zi/controller.py:383-384).

    Pattern: qibolab qibosoq driver ``_classify_shots`` (qibosoq/board.py:345-349,
    the qibolab threshold and iq_angle convention); qibocal t1 (coherence/t1.py:75,
    qibocal b559e4a7); zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    threshold, angle = 0.05, 0.3
    sign = 1 if _calc_theoretical_assignment_vec(1)[1] == 1 else -1
    configs = updated(
        platform, q0__acquisition={"threshold": threshold, "iq_angle": angle}
    )
    sequence = mz(platform)
    options = {
        "acquisition_type": AcquisitionType.DISCRIMINATION,
        "averaging_mode": AveragingMode.SINGLESHOT,
    }

    with injected(platform, states):
        cap = effects.play(platform, [sequence], configs=configs, **options)

    assert cap.acquisition_type == lq.AcquisitionType.DISCRIMINATION
    assert cap.averaging_mode == lq.AveragingMode.SINGLE_SHOT
    acquire = one(cap.acquires("q0/acquisition"), "q0/acquisition")
    kernel = cap.samples(acquire)
    c = kernel[0]
    np.testing.assert_allclose(kernel, c, rtol=1e-9, err_msg="kernel not constant")
    turned = sign * c
    assert min(abs(np.angle(turned) - angle), abs(np.angle(turned) + angle)) < 1e-9, (
        f"phase of {sign} x kernel {np.angle(turned):.4g} rad, not +-{angle}"
    )
    lq_threshold = np.atleast_1d(cap.calibration("q0/acquisition").threshold)
    assert lq_threshold.size == 1, f"thresholds {lq_threshold}"
    assert lq_threshold[0] / abs(c) == pytest.approx(sign * threshold)
    result = cap.results[readout_of(sequence).id]
    assert result.shape == shape(**options)
    expected = states(np.shape(cap.raw.get_data(acquire.op.handle))).real
    np.testing.assert_array_equal(result, expected)


# F18 RAW


@pytest.mark.feature("F18")
@missing(
    "raw traces (RAW)",
    raises=today("AcquisitionType.RAW is not supported."),
)
def test_raw_trace(platform):
    """RAW: LabOne Q records q0's trace for the acquisition duration (512 ns, the
    probe is 1024 ns), and qibolab returns it demodulated to baseband, as one I/Q
    pair per sample.

    Expected: 512 ns at 2 GSa/s = 1024 samples: the LabOne Q trace in order,
    demodulated at the probe frequency relative to its LO with qibolab's
    ``demodulate``. qibolab ``AcquisitionType.RAW``: "Return the waveform,
    demodulated to baseband" (execution_parameters.py); the qblox driver
    demodulates its scope trace the same way (qblox/results.py ``_scope``). The
    SHFQA trace is relative to the LO, so it still oscillates at that frequency.
    Today: "AcquisitionType.RAW is not supported." (zi/controller.py:383-384).

    Pattern: qibocal time_of_flight_readout (signal_experiments/
    time_of_flight_readout.py:102, qibocal b559e4a7); qibolab
    ``ExecutionParameters.results_shape`` (RAW: samples, 2);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    native = readout_of(mz(platform))
    duration = 512
    acquisition = native.acquisition.model_copy(update={"duration": duration})
    readout = native.model_copy(update={"acquisition": acquisition})
    sequence = PulseSequence([("q0/acquisition", readout)])

    cap = effects.play(platform, [sequence], acquisition_type=AcquisitionType.RAW)

    assert cap.acquisition_type == lq.AcquisitionType.RAW
    acquire = one(cap.acquires("q0/acquisition"), "q0/acquisition")
    assert integration_length(cap, acquire) == pytest.approx(duration)
    probe = one(cap.plays("q0/probe"), "q0/probe")
    assert cap.duration(probe) == pytest.approx(native.probe.duration)
    samples = int(duration * platform.sampling_rate)
    result = cap.results[readout.id]
    assert result.shape == shape(samples=samples, acquisition_type=AcquisitionType.RAW)
    [trace] = cap.data("q0/acquisition")
    lo = platform.config(platform.channels["q0/probe"].lo).frequency
    offset = (platform.config("q0/probe").frequency - lo) * NS  # GHz
    baseband = demodulate(iq(trace).T, offset, platform.sampling_rate).T
    np.testing.assert_allclose(result, baseband, rtol=1e-9, atol=1e-9)


# F21 SEQUENTIAL


@pytest.mark.feature("F21")
@pytest.mark.parametrize(
    "mode, labone_q",
    [
        pytest.param(AveragingMode.CYCLIC, lq.AveragingMode.CYCLIC, id="control"),
        pytest.param(
            AveragingMode.SEQUENTIAL,
            lq.AveragingMode.SEQUENTIAL,
            id="sequential",
            marks=missing(
                "sequential averaging (SEQUENTIAL)",
                raises=today("AveragingMode.SEQUENTIAL is not supported."),
            ),
        ),
    ],
)
def test_sequential_averaging(platform, mode, labone_q):
    """A real-time drive frequency sweep, averaged in the given mode: LabOne Q
    averages in that mode with the sweep inside the shot loop, and qibolab returns
    LabOne Q's average at each sweep point, in the sweeper order.

    Today: "AveragingMode.SEQUENTIAL is not supported." (zi/controller.py:385-386);
    LabOne Q has the mode.

    Pattern: qibolab ``AveragingMode.SEQUENTIAL`` (execution_parameters.py);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = drives_then_mz(platform, {"q0": [square()]})
    values = platform.config("q0/drive").frequency + np.array([0.0, 2e6, 1e6])
    sweeper = Sweeper(
        parameter=Parameter.frequency, values=values, channels=["q0/drive"]
    )

    cap = effects.play(platform, [sequence], [[sweeper]], averaging_mode=mode)

    assert cap.averaging_mode == labone_q
    [loop] = cap.sweeps()
    assert loop.real_time, "drive frequency sweep outside the shot loop"
    np.testing.assert_allclose(cap.frequency("q0/drive"), values)
    result = cap.results[readout_of(sequence).id]
    assert result.shape == shape([[sweeper]], averaging_mode=mode)
    [data] = cap.data("q0/acquisition")
    np.testing.assert_array_equal(result, iq(data))


# F45 fast reset


@pytest.mark.feature("F45")
@missing("fast reset", raises=today("Fast reset is not supported."))
def test_fast_reset(platform):
    """``fast_reset=True``: after q0 is measured, LabOne Q plays q0's RX on
    q0/drive when the state is 1 and nothing when it is 0 (active reset); the
    readout of the sequence still returns its result.

    LabOne Q gets a ``match`` on the handle of a q0 acquisition, whose
    ``case(state=1)`` plays the RX native (amplitude 0.5, 48 ns) on q0/drive and
    whose ``case(state=0)`` plays nothing there. Today: "Fast reset is not
    supported." (zi/controller.py:387-388).

    Pattern: LabOne Q DSL ``match(handle=...)``/``case(state=...)``, the construct
    of zhinst/laboneq-applications v26.4.0 tunable_transmon ``active_reset``;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = drives_then_mz(platform, {"q0": [square()]})
    native = rx(platform)

    cap = effects.play(platform, [sequence], fast_reset=True)

    handles = set(cap.handles("q0/acquisition"))
    matches = [
        node
        for top in cap.experiment.sections
        for node in sections(top)
        if isinstance(node, Match) and node.handle in handles
    ]
    assert matches, "no match on a q0 readout handle"

    def drive_plays(state):
        return [
            child
            for match in matches
            for case in match.children
            if isinstance(case, Case) and case.state == state
            for section in sections(case)
            for child in section.children
            if isinstance(child, PlayPulse)
            and child.pulse is not None
            and cap.channel_of(child.signal) == "q0/drive"
        ]

    flips = drive_plays(1)
    assert len(flips) == 1, f"{len(flips)} plays on q0/drive for state 1"
    flip = effects.Op(flips[0], flips[0].signal, "q0/drive", True)
    assert cap.amplitude(flip) == pytest.approx(native.amplitude)
    assert cap.duration(flip) == pytest.approx(native.duration)
    assert drive_plays(0) == [], "q0/drive played for state 0"
    assert cap.results[readout_of(sequence).id].shape == shape()


# F53 default execute options (X01)


@pytest.mark.feature("F53")
@missing(
    "platform.execute with the default options (DISCRIMINATION, SINGLESHOT)",
    raises=today("AcquisitionType.DISCRIMINATION is not supported."),
)
def test_default_execute_returns_states(platform):
    """``platform.execute`` with no options returns, for each shot, the state (0
    or 1) LabOne Q measured, as qibo's backend expects.

    Audit X01; test_missing_infra.py::test_default_execute_options checks what
    LabOne Q gets. Today the default DISCRIMINATION is refused
    (zi/controller.py:383).

    Pattern: qibolab ``ExecutionParameters`` defaults
    (execution_parameters.py:76-78) and the qibo backend call
    ``platform.execute(sequences, nshots=...)`` (backends.py:138,179);
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    nshots = platform.settings.nshots
    # a calibrated platform, as test_conformance.py::test_acquisition_type[F17]
    platform.parameters.configs.update(
        updated(platform, q0__acquisition={"threshold": 0.0, "iq_angle": 0.0})
    )

    zi = platform.instruments["zi"]
    with injected(platform, states), effects.capture(zi) as cap:
        results = platform.execute([sequence], nshots=nshots)

    result = results[readout_of(sequence).id]
    assert result.shape == (nshots,)
    [data] = cap.data("q0/acquisition")
    np.testing.assert_array_equal(result, data.real)
    assert set(np.unique(result)) <= {0, 1}
