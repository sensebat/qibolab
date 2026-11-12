"""Effect tests for pulse and readout features the driver lacks.

Features: F25 (separate probe and Acquisition), F27 (VirtualZ), F28 (flux pulse),
F35/F36 (exponential and SNZ envelopes on a flux line), F39 (flux filters), F41
(integration kernel), F47 (two readouts of one qubit), and the audit gaps G3 (F48,
readouts at different times, mid-circuit), G4 (F49, two probe waveforms of one
qubit), G5 (F50, chirp), G6 (F51, durations off the sample grid), G7 (F52, SHFQC).
Audit: sources/review/unimplemented-audit (README.md, capability.md, coverage.md).

Each test checks what LabOne Q must receive, or what qibolab must return, once the
feature works. While the driver lacks it, the test is a strict xfail
(:func:`conftest.missing`) that holds only for the failure seen today: the
exception type and message when the call fails, an ``AssertionError`` when it runs
silently wrong. It turns red the day the feature works (XPASS strict) or the day the
call fails another way. Where a supported form of the same call exists, it runs as a
``control`` case with the same assertions and must pass.

Expected values come from qibolab objects (``Pulse.i/q``,
``PulseSequence.to_relative_phases``, the configs), never from the driver. Run in
LabOne Q emulation only (``do_emulation``).

Code under test: sensebat/qibolab@orkun/rc-tests, driver from
ElStabilini/qibolab@zi_drivers (e8757e58) src/qibolab/_core/instruments/zi/.

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q); LabOne Q ``OutputSimulator`` port traces (qibolab
0.1 ``play_sim``, 4aaa4163); qibolab ``ExecutionParameters.results_shape``.
"""

import re

import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException

from qibolab._core.components import (
    AcquisitionChannel,
    DcChannel,
    ExponentialFilter,
    FiniteImpulseResponseFilter,
    IqChannel,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.instruments.zi import (
    ZiAcquisitionConfig,
    ZiController,
    ZiDcConfig,
    ZiInstrument,
    ZiIqConfig,
)
from qibolab._core.parameters import Parameters, Settings
from qibolab._core.platform import Platform
from qibolab._core.pulses import (
    Acquisition,
    Delay,
    Exponential,
    GaussianSquare,
    Pulse,
    Readout,
    Rectangular,
    Snz,
    VirtualZ,
)
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence

from . import effects
from .conftest import NSHOTS, QUBITS, RELAXATION, missing, natives
from .effects import square
from .test_conformance import mz, readout_of, updated
from .test_effects_pulses import Output, pulse_samples
from .test_missing_infra import _ports, integration_length, rabi, readout_on, start


def missing_because(reason, exception, match):
    """:func:`conftest.missing` that holds only for ``exception`` matching ``match``."""
    return missing(reason, raises=pytest.RaisesExc(exception, match=match))


def results_shape(**options):
    """``results_shape`` of the options :func:`effects.play` runs with."""
    return ExecutionParameters(**(effects.DEFAULTS | options)).results_shape([])


def same_angle(a, b, atol=1e-6) -> bool:
    """Whether two phases (rad) agree modulo 2 pi."""
    return abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))) <= atol


# F25 probe pulse and Acquisition as separate events


def _probe_then_acquisition(_):
    probe = square(duration=1024, amplitude=0.1)
    return PulseSequence(
        [("q0/probe", probe), ("q0/acquisition", Acquisition(duration=1024))]
    )


def _readout(_):
    probe = square(duration=1024, amplitude=0.1)
    return PulseSequence([("q0/acquisition", Readout.from_probe(probe))])


@pytest.mark.feature("F25")
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_readout, id="control"),
        pytest.param(
            _probe_then_acquisition,
            id="separate",
            marks=missing_because(
                "probe pulse and Acquisition as separate events",
                NotImplementedError,
                r"^Acquisition on channel 'q0/acquisition'",
            ),
        ),
    ],
)
def test_probe_and_acquisition_separate(platform, build):
    """F25. A probe pulse on q0/probe and an ``Acquisition`` on q0/acquisition,
    both at t = 0: the probe plays 1024 ns at amplitude 0.1 and one integration of
    1024 ns starts with it, with the acquisition id as handle and one result.

    Control: the same as a ``Readout``. Today: ``NotImplementedError: Acquisition on
    channel 'q0/acquisition' is not supported.`` (zi/pulse.py:72).

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; LabOne Q
    ``play(probe)`` + ``acquire()`` on separate signals; ElStabilini/qibolab@zi_drivers
    (e8757e58) qblox and qm drivers play ``Acquisition`` on its own.
    """
    sequence = build(platform)
    acquisition = readout_of(sequence)

    cap = effects.play(platform, [sequence])

    (probe,) = cap.plays("q0/probe")
    assert cap.amplitude(probe) == pytest.approx(0.1)
    assert cap.duration(probe) == pytest.approx(1024)
    (acquire,) = cap.acquires("q0/acquisition")
    assert acquire.op.handle == str(acquisition.id)
    assert integration_length(cap, acquire) == pytest.approx(1024)
    assert start(cap, probe) == pytest.approx(0)
    assert start(cap, acquire) == pytest.approx(0)
    assert set(cap.results) == {acquisition.id}
    assert cap.results[acquisition.id].shape == results_shape()


# F27 VirtualZ


def _drive_with_vzs(platform):
    """q0: RX, VZ(pi/2), RX, VZ(0.4), RX with relative phase 0.1; q1: RX. Then MZ."""
    natives = platform.natives.single_qubit
    rx0 = natives["q0"].RX()[0][1]
    rx1 = natives["q1"].RX()[0][1]
    sequence = PulseSequence(
        [
            ("q0/drive", rx0),
            ("q0/drive", VirtualZ(phase=np.pi / 2)),
            ("q0/drive", rx0),
            ("q0/drive", VirtualZ(phase=0.4)),
            ("q0/drive", rx0.model_copy(update={"relative_phase": 0.1})),
            ("q1/drive", rx1),
        ]
    )
    for q in ("q0", "q1"):
        sequence.append((f"{q}/acquisition", Delay(duration=3 * rx0.duration)))
    return sequence + mz(platform, "q0", "q1")


def played_phases(cap, channel) -> list[float]:
    """Phase of each pulse played on ``channel``, in order, in rad.

    Phase of the waveform handed to LabOne Q (operation phase, amplitude and
    samples, read at the sample of largest magnitude), plus the oscillator phase:
    ``increment_oscillator_phase`` of this and earlier plays on the signal, from the
    last ``set_oscillator_phase``. Covers a VirtualZ done as a pulse phase, as an
    oscillator phase step, or in the samples.
    """
    oscillator = 0.0
    phases = []
    for op in cap.plays(channel):
        if not op.real_time:
            continue
        if op.op.set_oscillator_phase is not None:
            oscillator = effects.values(op.op.set_oscillator_phase)
        if op.op.increment_oscillator_phase is not None:
            oscillator += effects.values(op.op.increment_oscillator_phase)
        if op.op.pulse is None:
            continue
        wave = np.ravel(cap.waveform(op))
        phases.append(float(np.angle(wave[np.argmax(np.abs(wave))])) + oscillator)
    return phases


@pytest.mark.feature("F27")
@pytest.mark.parametrize(
    "folded",
    [
        pytest.param(True, id="control"),
        pytest.param(
            False,
            id="virtual-z",
            marks=missing_because(
                "VirtualZ on a drive channel",
                NotImplementedError,
                r"^VirtualZ on channel 'q0/drive'",
            ),
        ),
    ],
)
def test_virtual_z_shifts_later_pulses(platform, folded):
    """F27. A ``VirtualZ`` on q0/drive adds its phase to every later pulse on q0/drive
    and to nothing else: the three RX of q0 play with phases 0, pi/2 and
    pi/2 + 0.4 + 0.1; the RX of q1 plays with phase 0.

    Expected phases: ``PulseSequence.to_relative_phases`` (qibolab's own folding of
    VirtualZ into relative phases, sequence.py:299-325). Control: that folded
    sequence, played without VirtualZ. Today: ``NotImplementedError: VirtualZ on
    channel 'q0/drive' is not supported.``

    Pattern: qibolab ``PulseSequence.to_relative_phases``; qblox driver
    ``_process_virtualz`` (qblox/sequence/experiment.py:197, phase step of later
    plays); LabOne Q ``play(increment_oscillator_phase=...)``;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = _drive_with_vzs(platform)
    played = sequence.to_relative_phases() if folded else sequence
    expected = {
        ch: [p.relative_phase for c, p in sequence.to_relative_phases() if c == ch]
        for ch in ("q0/drive", "q1/drive")
    }
    assert expected["q0/drive"] == pytest.approx([0, np.pi / 2, np.pi / 2 + 0.5])

    cap = effects.play(platform, [played])

    for channel, phases in expected.items():
        found = played_phases(cap, channel)
        assert len(found) == len(phases), f"{channel}: {found}"
        assert all(map(same_angle, found, phases)), f"{channel}: {found} != {phases}"
    for q in ("q0", "q1"):
        assert cap.results[readout_on(sequence, q).id].shape == results_shape()


# F28, F35, F36 pulses on a flux line


ENVELOPES = {
    "F28": Rectangular(),
    "F35": Exponential(tau=0.2, upsilon=0.5),
    "F36": Snz(t_idling=10),
}


def _flux_cases():
    for id, envelope in ENVELOPES.items():
        name = type(envelope).__name__.lower()
        yield pytest.param(
            "q0/drive",
            envelope,
            id=f"{id}-{name}-control",
            marks=pytest.mark.feature(id),
        )
        yield pytest.param(
            "q0/flux",
            envelope,
            id=f"{id}-{name}-flux",
            marks=[
                pytest.mark.feature(id),
                missing_because(
                    f"{name} pulse on a flux channel",
                    NotImplementedError,
                    r"^Pulse on channel 'q0/flux'",
                ),
            ],
        )


@pytest.mark.parametrize("channel,envelope", list(_flux_cases()))
def test_pulse_on_flux_line(platform, channel, envelope):
    """F28, F35, F36. A 40 ns pulse of amplitude 0.2 on q0/flux plays its samples on
    q0's flux port (``Pulse.i``; ``Pulse.q`` is 0 for these envelopes), then the
    readout starts when it ends. The DC offset of q0/flux stays the config's.

    Control: the same pulse on q0/drive. Today: ``NotImplementedError: Pulse on
    channel 'q0/flux' is not supported.`` (zi/pulse.py:72).

    Pattern: qibocal flux_gate (flux_dependence/flux_gate.py:121), chevron and
    snz_optimize (two_qubit_interaction/snz_optimize.py:169) play such pulses on the
    flux line; zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; LabOne Q
    ``OutputSimulator`` on the HDAWG output.
    """
    p = Pulse(duration=40, amplitude=0.2, envelope=envelope)
    rate = platform.sampling_rate
    assert np.allclose(p.q(rate), 0)
    sequence = PulseSequence([(channel, p), ("q0/acquisition", Delay(duration=40))])
    sequence += mz(platform)

    cap = effects.play(platform, [sequence], nshots=1)

    (op,) = cap.plays(channel)
    np.testing.assert_allclose(cap.waveform(op), pulse_samples(p, rate), atol=1e-12)
    assert cap.duration(op) == pytest.approx(40)
    (probe,) = cap.plays("q0/probe")
    assert start(cap, probe) == pytest.approx(start(cap, op) + 40)
    assert cap.offset("q0/flux") == pytest.approx(platform.config("q0/flux").offset)
    trace = Output(cap)(channel)
    played = trace.at(0, int(40 * trace.rate))
    np.testing.assert_allclose(
        played.real, pulse_samples(p, trace.rate).real, atol=1e-4
    )


# F39 filters on a flux channel


@pytest.mark.feature("F39")
@missing_because(
    "filters on a flux channel", NotImplementedError, r"^Filters on 'q0/flux'"
)
def test_flux_filters_reach_precompensation(platform):
    """F39. Filters of the q0/flux config become the LabOne Q precompensation of
    q0's flux signal (HDAWG with the PC option): one exponential filter with the
    filter's amplitude and its time constant in s, and an FIR filter whose first
    taps are the coefficients (the rest 0). q1/flux, without filters, gets no
    precompensation.

    Assumed mapping: qibolab ``ExponentialFilter`` corrects ``1 + amplitude *
    exp(-t / tau)`` with ``tau`` in samples (components/filters.py:11-22), the
    HDAWG exponential filter the same form with a time constant in s; samples are
    at the platform sampling rate. Today: ``NotImplementedError: Filters on 'q0/flux'
    are not supported.`` (zi/controller.py:250).

    Pattern: qibocal cryoscope (flux_dependence/cryoscope.py:385) sets and clears
    these filters; LabOne Q ``SignalCalibration.precompensation``
    (``ExponentialCompensation``, ``FIRCompensation``); zhinst/laboneq-applications
    v26.4.0 tests/helpers/dsl.py.
    """
    exponential = ExponentialFilter(amplitude=-0.1, tau=40)
    fir = FiniteImpulseResponseFilter(coefficients=[1.0, -0.2, 0.1])
    configs = updated(platform, q0__flux={"filters": [exponential, fir]})

    sequence = mz(platform)
    cap = effects.play(platform, [sequence], configs=configs)

    pre = cap.calibration("q0/flux").precompensation
    assert pre is not None, "no precompensation on q0/flux"
    (exp,) = pre.exponential
    assert exp.amplitude == pytest.approx(exponential.amplitude)
    seconds = exponential.tau / (platform.sampling_rate * 1e9)
    assert exp.timeconstant == pytest.approx(seconds, rel=1e-6)
    taps = np.asarray(pre.FIR.coefficients, dtype=float)
    n = len(fir.coefficients)
    np.testing.assert_allclose(taps[:n], fir.coefficients)
    np.testing.assert_allclose(taps[n:], 0)
    other = cap.calibration("q1/flux").precompensation
    assert other is None or not other.is_nonzero()
    assert cap.results[readout_of(sequence).id].shape == results_shape()


# F41 integration kernel


@pytest.mark.feature("F41")
@missing_because(
    "integration kernel in the acquisition config",
    NotImplementedError,
    r"^Readout on channel 'q0/acquisition'|[Kk]ernel",
)
def test_kernel_weights_reach_acquire(platform):
    """F41. A kernel in the q0/acquisition config becomes the integration weights of
    q0's acquisition (equal up to one complex factor); q1, without kernel, keeps
    unit weights. Both readouts return their result.

    The kernel is complex, not symmetric and of varying magnitude, so a dropped,
    conjugated, reversed or truncated kernel shows. Today: ``NotImplementedError:
    Readout on channel 'q0/acquisition' is not supported.`` (zi/pulse.py:59, 72);
    test_effects_configs_errors.py::test_kernel_refused checks that refusal.

    Pattern: qibocal calibrate_state_discrimination
    (signal_experiments/calibrate_state_discrimination.py:237) writes the kernel;
    LabOne Q ``acquire(kernel=sampled pulse)``; zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    sequence = mz(platform, "q0", "q1")
    samples = int(readout_on(sequence, "q0").duration * platform.sampling_rate)
    kernel = np.exp(1j * np.linspace(0, np.pi / 2, samples)) * np.linspace(
        1, 2, samples
    )
    configs = updated(platform, q0__acquisition={"kernel": kernel})

    cap = effects.play(platform, [sequence], configs=configs)

    for q, weights in (("q0", kernel), ("q1", np.ones(samples))):
        (acquire,) = cap.acquires(f"{q}/acquisition")
        found = cap.samples(acquire)
        assert found.shape == weights.shape, f"{q}: {found.shape}"
        ratio = found / weights
        assert np.allclose(ratio, ratio[0]), f"{q}: weights are not {q}'s kernel"
        assert cap.results[readout_on(sequence, q).id].shape == results_shape()


# F47 two readouts of one qubit


@pytest.mark.feature("F47")
@missing_because(
    "two readouts of one qubit in one sequence",
    LabOneQException,
    "multiple 'play' operations in section",
)
def test_two_readouts_of_one_qubit_in_order(platform):
    """F47. MZ, 100 ns, MZ on q0, single shots: q0's probe plays twice, the second
    one 1024 + 100 ns after the first; each readout integrates with its own handle
    and returns one result per shot.

    Today LabOne Q rejects the experiment: "There are multiple 'play' operations in
    section" (one section per sequence, zi/controller.py:410).

    Pattern: qibocal readout_characterization (readout/readout_characterization.py:
    99-101); zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    sequence.append(("q0/acquisition", Delay(duration=100)))
    sequence += mz(platform)
    first, second = (ro for _, ro in sequence.acquisitions)
    gap = first.duration + 100

    cap = effects.play(platform, [sequence], averaging_mode=AveragingMode.SINGLESHOT)

    plays = cap.plays("q0/probe")
    assert [start(cap, op) for op in plays] == pytest.approx([0, gap])
    acquires = cap.acquires("q0/acquisition")
    assert [op.op.handle for op in acquires] == [str(first.id), str(second.id)]
    assert [start(cap, op) for op in acquires] == pytest.approx([0, gap])
    shape = results_shape(averaging_mode=AveragingMode.SINGLESHOT)
    for ro in (first, second):
        assert cap.results[ro.id].shape == shape


# F48 (G3) readouts at different times, mid-circuit


PROBE = 1024
RX = 48
GAP = 200


def _sequential(platform):
    """q0 read at 0; q1 read GAP ns after q0's readout ends."""
    sequence = mz(platform, "q0")
    sequence.append(("q1/acquisition", Delay(duration=PROBE + GAP)))
    sequence += mz(platform, "q1")
    expected = {"q0/probe": [(0, PROBE)], "q1/probe": [(PROBE + GAP, 2 * PROBE + GAP)]}
    return sequence, expected


def _mid_circuit(platform):
    """q0: RX, MZ. q1: RX once q0's readout ends, then MZ."""
    natives = platform.natives.single_qubit
    sequence = PulseSequence()
    sequence += natives["q0"].RX()
    sequence.append(("q0/acquisition", Delay(duration=RX)))
    sequence += natives["q0"].MZ()
    sequence.append(("q1/drive", Delay(duration=RX + PROBE)))
    sequence += natives["q1"].RX()
    sequence.append(("q1/acquisition", Delay(duration=2 * RX + PROBE)))
    sequence += natives["q1"].MZ()
    expected = {
        "q0/drive": [(0, RX)],
        "q0/probe": [(RX, RX + PROBE)],
        "q1/drive": [(RX + PROBE, 2 * RX + PROBE)],
        "q1/probe": [(2 * RX + PROBE, 2 * RX + 2 * PROBE)],
    }
    return sequence, expected


def _together(platform):
    """Control: RX of both qubits, both read out after it (equal RX lengths)."""
    sequence = rabi(platform)
    expected = {
        "q0/drive": [(0, RX)],
        "q1/drive": [(0, RX)],
        "q0/probe": [(RX, RX + PROBE)],
        "q1/probe": [(RX, RX + PROBE)],
    }
    return sequence, expected


DIFFERENT_TIMES = missing_because(
    "readouts that start at different times in one sequence",
    LabOneQException,
    "start at the same time",
)


@pytest.mark.feature("F48")
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_together, id="control"),
        pytest.param(_sequential, id="sequential", marks=DIFFERENT_TIMES),
        pytest.param(_mid_circuit, id="mid-circuit", marks=DIFFERENT_TIMES),
    ],
)
def test_readouts_at_their_own_times(platform, build):
    """F48 (audit G3). Readouts of two qubits at different times in one sequence:
    one after the other on the shared SHFQA line, or mid-circuit (q0 is read out,
    then q1 is driven and read out). Each probe and drive plays at the time the
    sequence gives it, read on the instrument ports by LabOne Q's output simulator
    (so it holds whatever sections the driver builds), each acquisition starts with
    its probe, and each readout returns its result.

    Control: both qubits driven and read out together. Today LabOne Q rejects the
    experiment: "all play signals must start at the same time" (one section per
    sequence, zi/controller.py:410).

    Pattern: qibocal rabi on several qubits (rabi/utils.py:327) and
    readout_characterization (readout/readout_characterization.py:99-101); LabOne Q
    ``OutputSimulator``; zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence, expected = build(platform)

    cap = effects.play(platform, [sequence], nshots=1)

    output = Output(cap)
    for channel, segments in expected.items():
        if channel.endswith("probe"):
            continue
        on = merged(output(channel).segments(threshold=1e-3))
        assert on == [pytest.approx(s, abs=1.0) for s in segments], channel
    probes = {c: s for c, s in expected.items() if c.endswith("probe")}
    trace = output("q0/probe")
    union = merged(sorted(s for segments in probes.values() for s in segments))
    on = merged(trace.segments(threshold=1e-3))
    assert on == [pytest.approx(s, abs=1.0) for s in union], "probe port"
    for channel, segments in probes.items():
        for a, b in segments:
            own = tone(cap, trace, channel, a, b)
            for other, others in probes.items():
                if other != channel and not overlaps(others, a, b):
                    assert own > 5 * tone(cap, trace, other, a, b), (channel, a)
    for q in ("q0", "q1"):
        (probe,) = cap.plays(f"{q}/probe")
        (acquire,) = cap.acquires(f"{q}/acquisition")
        assert start(cap, acquire) == pytest.approx(start(cap, probe)), q
        assert cap.results[readout_on(sequence, q).id].shape == results_shape(), q


def merged(segments, gap=5.0) -> list[tuple[float, float]]:
    """Segments with gaps shorter than ``gap`` ns closed (beats of two tones)."""
    out = []
    for a, b in segments:
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlaps(segments, a, b) -> bool:
    return any(x < b and a < y for x, y in segments)


def tone(cap, trace, channel, a, b) -> float:
    """Magnitude of ``channel``'s probe tone on the probe port between ``a`` and
    ``b`` ns.

    The probes of q0 and q1 share one SHFQA output, each modulated at its own offset
    from the shared LO (here 1 MHz apart), so demodulating at one offset over a
    readout tells whose readout it is.
    """
    n = int(np.rint((a - trace.time[0]) * trace.rate))
    m = int(np.rint((b - trace.time[0]) * trace.rate))
    wave, time = trace.wave[n:m], trace.time[n:m] * effects.NS
    offset = cap.frequency(channel) - cap.lo(channel)
    return float(np.abs(np.mean(wave * np.exp(-2j * np.pi * offset * time))))


# F49 (G4) two probe waveforms for one qubit


TWO_PROBES = missing_because(
    "two probe waveforms for one qubit in one call",
    LabOneQException,
    "single pulse waveform|unique pulse",
)


@pytest.mark.feature("F49")
@pytest.mark.parametrize(
    "envelope,amplitude",
    [
        pytest.param(Rectangular(), 0.1, id="control"),
        pytest.param(
            GaussianSquare(risefall=40, sigma=10),
            0.1,
            id="two-envelopes",
            marks=TWO_PROBES,
        ),
        pytest.param(Rectangular(), 0.2, id="two-amplitudes", marks=TWO_PROBES),
    ],
)
def test_two_probe_waveforms_one_qubit(platform, envelope, amplitude):
    """F49 (audit G4). Two sequences in one call read out q0, first with its
    rectangular 1024 ns probe of amplitude 0.1, then with a 1024 ns probe of
    ``envelope`` and ``amplitude``: q0's probe plays each waveform in turn, and each
    readout returns its result.

    Control: the same probe twice. Today LabOne Q rejects a second envelope:
    "SHFQA generators can only hold a single pulse waveform" (zi/pulse.py:20-28),
    and a second amplitude: "Too many unique pulses on signal 'q0/probe'" (the
    SHFQA generator plays stored waveforms, amplitude included).
    test_missing_infra.py covers two lengths.

    Pattern: qibocal resonator_punchout and readout optimisation change the probe
    amplitude (resonator_spectroscopies/resonator_punchout.py:123,
    readout/amplitude_frequency_optimization.py:173); zhinst/laboneq-applications
    v26.4.0 tests/helpers/dsl.py.
    """
    first = mz(platform)
    probe = Pulse(duration=PROBE, amplitude=amplitude, envelope=envelope)
    second = PulseSequence([("q0/acquisition", Readout.from_probe(probe))])
    probes = [readout_of(first).probe, probe]

    cap = effects.play(platform, [first, second])

    rate = platform.sampling_rate
    plays = cap.plays("q0/probe")
    assert len(plays) == 2
    for op, p in zip(plays, probes):
        np.testing.assert_allclose(cap.waveform(op), pulse_samples(p, rate), atol=1e-12)
    for sequence in (first, second):
        assert cap.results[readout_of(sequence).id].shape == results_shape()


# F50 (G5) chirp


CHIRP = 25000
"""Chirp rate in Hz/ns: 5 MHz swept over 200 ns, a phase of pi at the end."""


@pytest.mark.feature("F50")
@missing("pulse chirp is ignored: the pulse plays without it")
def test_chirp_phase_is_quadratic(platform):
    """F50 (audit G5). A 200 ns drive pulse with ``chirp=(25000, "Hz/nsec")`` plays
    a waveform whose frequency rises linearly from the start of the pulse: its phase
    is ``pi * rate * t**2`` (t in s from the pulse start), within 0.05 rad.

    QUA chirp semantics: the frequency changes at ``rate`` while the pulse plays.
    Today the chirp is not read (zi/pulse.py:52-58) and the phase is constant.
    test_missing_infra.py checks only that some chirp is there.

    Pattern: qibolab ``Pulse.chirp`` ("QUA-style", pulses/pulse.py:78-84); QM
    driver ``qua.play(..., chirp=...)``; zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    pulse = square(duration=200).model_copy(update={"chirp": (CHIRP, "Hz/nsec")})
    sequence = PulseSequence(
        [("q0/drive", pulse), ("q0/acquisition", Delay(duration=200))]
    ) + mz(platform)

    cap = effects.play(platform, [sequence])

    (op,) = cap.plays("q0/drive")
    wave = np.ravel(cap.waveform(op))
    t = np.arange(len(wave)) / platform.sampling_rate * effects.NS
    phase = np.unwrap(np.angle(wave))
    expected = np.pi * (CHIRP / effects.NS) * t**2
    error = phase - phase[0] - expected
    assert np.abs(error).max() < 0.05, f"phase off by up to {np.abs(error).max():.3g}"
    assert np.allclose(np.abs(wave), pulse.amplitude)


# F51 (G6) durations off the sample grid


REFUSAL = r"sampl|grid|[Dd]uration"


def _off_grid(target, duration):
    marks = []
    if duration % 0.5:
        marks = missing(f"{target} of {duration} ns is rounded down to the sample grid")
    return pytest.param(target, duration, id=f"{target}-{duration}", marks=marks)


@pytest.mark.feature("F51")
@pytest.mark.parametrize(
    "target,duration",
    [
        _off_grid("drive", 40.5),
        _off_grid("drive", 40.3),
        _off_grid("readout", 1000.5),
        _off_grid("readout", 1000.3),
    ],
)
def test_off_grid_duration_refused_or_exact(platform, target, duration):
    """F51 (audit G6). A pulse whose length is not a whole number of samples (2
    GSa/s) is refused with an error naming the duration or the sample grid, or
    plays exactly its length; never rounded silently. On-grid lengths (40.5, 1000.5
    ns) are the control and must play exactly.

    Drive: the drive pulse length. Readout: the probe length and the integration
    length. Today 40.3 ns plays 80 samples (40.0 ns) and 1000.3 ns plays and
    integrates 2000 samples (``int(duration * sampling_rate)``, zi/pulse.py:27).

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests
    (refuse with a clear message); zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    p = square(duration=duration, amplitude=0.1)
    if target == "drive":
        sequence = PulseSequence(
            [("q0/drive", p), ("q0/acquisition", Delay(duration=duration))]
        ) + mz(platform)
    else:
        sequence = PulseSequence([("q0/acquisition", Readout.from_probe(p))])

    try:
        cap = effects.play(platform, [sequence])
    except (ValueError, NotImplementedError, LabOneQException) as error:
        message = f"{type(error).__name__}: {error}"
        assert re.search(REFUSAL, str(error)), message
        return

    if target == "drive":
        (op,) = cap.plays("q0/drive")
        assert cap.duration(op) == pytest.approx(duration, abs=1e-6)
    else:
        (op,) = cap.plays("q0/probe")
        assert cap.duration(op) == pytest.approx(duration, abs=1e-6)
        (acquire,) = cap.acquires("q0/acquisition")
        assert integration_length(cap, acquire) == pytest.approx(duration, abs=1e-6)


# F52 (G7) SHFQC


def _shfqc_rack():
    """Drives and the multiplexed readout of q0, q1 on one SHFQC; flux on an HDAWG."""
    instruments = {
        "shfqc": ZiInstrument(kind="SHFQC", serial="DEV12991", options="SHFQC/QC6CH"),
        "hdawg": ZiInstrument(kind="HDAWG", serial="DEV8901", options="HDAWG8/PC"),
    }
    channels, ports = {}, {}
    for i, q in enumerate(QUBITS):
        channels[f"{q}/drive"] = IqChannel(
            device="shfqc", path=str(i), lo=f"shfqc/sg{i}/lo"
        )
        ports[f"{q}/drive"] = ("shfqc", {f"SGCHANNELS/{i}/OUTPUT"})
    for i, q in enumerate(QUBITS):
        channels[f"{q}/probe"] = IqChannel(device="shfqc", path="0", lo="shfqc/qa0/lo")
        ports[f"{q}/probe"] = ("shfqc", {"QACHANNELS/0/OUTPUT"})
        channels[f"{q}/acquisition"] = AcquisitionChannel(
            device="shfqc", path="0", probe=f"{q}/probe"
        )
        ports[f"{q}/acquisition"] = ("shfqc", {"QACHANNELS/0/INPUT"})
        channels[f"{q}/flux"] = DcChannel(device="hdawg", path=str(i))
        ports[f"{q}/flux"] = ("hdawg", {f"SIGOUTS/{i}"})
    return instruments, channels, ports


def _split_rack():
    """Control: the same channels on an SHFSG and an SHFQA (supported kinds)."""
    instruments, channels, ports = _shfqc_rack()
    instruments.pop("shfqc")
    instruments["shfsg"] = ZiInstrument(
        kind="SHFSG", serial="DEV12901", options="SHFSG8"
    )
    instruments["shfqa"] = ZiInstrument(
        kind="SHFQA", serial="DEV12951", options="SHFQA4"
    )
    for id, channel in channels.items():
        if channel.device == "shfqc":
            device = "shfsg" if id.endswith("drive") else "shfqa"
            lo = getattr(channel, "lo", None)
            lo = lo and lo.replace("shfqc", device)
            update = {"device": device} | ({"lo": lo} if lo else {})
            channels[id] = channel.model_copy(update=update)
            ports[id] = (device, ports[id][1])
    return instruments, channels, ports


def _platform(instruments, channels) -> Platform:
    qubits = {q: Qubit.default(q) for q in QUBITS}
    controller = ZiController(
        address="127.0.0.1:8004",
        instruments=instruments,
        pqsc="DEV10901",
        channels=channels,
        emulation=True,
    )
    configs = {}
    for i, q in enumerate(QUBITS):
        configs[f"{q}/drive"] = ZiIqConfig(frequency=6.3e9 + i * 1e6, power_range=10)
        configs[f"{q}/probe"] = ZiIqConfig(frequency=7.0e9 + i * 1e6, power_range=5)
        configs[f"{q}/acquisition"] = ZiAcquisitionConfig(
            delay=200, smearing=0, power_range=-30
        )
        configs[f"{q}/flux"] = ZiDcConfig(offset=0.1 * (i + 1), power_range=5)
    for id, channel in channels.items():
        if getattr(channel, "lo", None) is not None:
            lo = 6.8e9 if id.endswith("probe") else 6.2e9
            configs[channel.lo] = OscillatorConfig(frequency=lo, power=0)
    parameters = Parameters(
        settings=Settings(nshots=NSHOTS, relaxation_time=RELAXATION),
        configs=configs,
        native_gates=natives(qubits),
    )
    return Platform(
        name="shfqc",
        parameters=parameters,
        instruments={"zi": controller},
        qubits=qubits,
    )


@pytest.mark.feature("F52")
@pytest.mark.parametrize(
    "rack",
    [
        pytest.param(_split_rack, id="control"),
        pytest.param(
            _shfqc_rack,
            id="SHFQC",
            marks=missing_because("SHFQC instruments", KeyError, "SHFQC"),
        ),
    ],
)
def test_shfqc_platform(rack):
    """F52 (audit G7). A two-qubit platform whose drives and multiplexed readout are
    on one SHFQC (SG channels 0, 1 and QA channel 0) and whose flux lines are on an
    HDAWG: a rabi-shaped sequence reaches those ports, each drive plays its RX with
    the frequency of its config, each probe has its frequency, each flux line its
    offset, and each readout returns its result.

    Control: the same channels on an SHFSG and an SHFQA. Assumed wiring, as for the
    supported kinds: :attr:`Channel.device` names the instrument, :attr:`Channel.path`
    the port; the probe named by an :class:`AcquisitionChannel` is a QA channel.
    Today the device setup has no SHFQC: ``KeyError: 'SHFQC'`` on connect
    (zi/controller.py:53,60,177).

    Pattern: LabOne Q device setup connections (``create_connection(to_signal,
    ports)``, SHFQC ``SGCHANNELS``/``QACHANNELS`` ports); zhinst/laboneq-applications
    v26.4.0 tests/helpers/dsl.py.
    """
    instruments, channels, ports = rack()
    platform = _platform(instruments, channels)
    zi = platform.instruments["zi"]
    zi.connect()
    try:
        sequence = rabi(platform)
        cap = effects.play(platform, [sequence])

        for id, (uid, expected) in ports.items():
            uids, found = _ports(cap.setup, id)
            assert uids == {uid}, f"{id} on {uids}"
            assert found == expected, f"{id} on {found}"
        for q in QUBITS:
            rx = platform.natives.single_qubit[q].RX()[0][1]
            (drive,) = cap.plays(f"{q}/drive")
            assert cap.amplitude(drive) == pytest.approx(rx.amplitude), q
            for kind in ("drive", "probe"):
                channel = f"{q}/{kind}"
                frequency = platform.config(channel).frequency
                assert cap.frequency(channel) == pytest.approx(frequency), channel
            offset = platform.config(f"{q}/flux").offset
            assert cap.offset(f"{q}/flux") == pytest.approx(offset), q
            assert cap.results[readout_on(sequence, q).id].shape == results_shape(), q
    finally:
        zi.disconnect()
