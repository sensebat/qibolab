"""Effects of pulses and sequences, observed at the LabOne Q boundary.

What a qibolab ``PulseSequence`` fixes, checked on what ``ZiController.play`` hands to
LabOne Q: amplitude, relative phase, envelope samples (I and Q), duration, the timing
set by ``Delay`` and ``Align`` across channels, several sequences in one call, and a
``Readout`` as a probe pulse plus an acquisition.

Two views, both public LabOne Q API present in laboneq 26.4 and 26.10:

- the DSL ``Experiment`` (:mod:`effects`): which signal plays what, with which
  amplitude, phase, samples and acquisition handle;
- LabOne Q's ``OutputSimulator`` on the compiled experiment: the samples each
  instrument port plays, in time. Times and durations come from there, so they are
  those LabOne Q will play, not those the driver claims (the driver's sampling rate is
  not used to read them).

Expected values are computed from the qibolab objects (``Pulse.i/q``,
``PulseSequence`` timing, ``ExecutionParameters.relaxation_time``), never from the
driver. Each test is also run under planted driver bugs (:mod:`mutants`) that it must
catch.

Code under test: ElStabilini/qibolab@zi_drivers (e8757e58)
src/qibolab/_core/instruments/zi/.
"""

from typing import NamedTuple

import numpy as np
import pytest
from laboneq.simple import OutputSimulator

from qibolab._core.pulses import (
    Custom,
    Delay,
    Drag,
    Exponential,
    Gaussian,
    GaussianSquare,
    Pulse,
    Readout,
    Rectangular,
    Snz,
)
from qibolab._core.sequence import PulseSequence

from . import effects
from .effects import square
from .mutants import kills

# helpers: LabOne Q output simulator


class Trace(NamedTuple):
    """Samples LabOne Q plays on one instrument port."""

    time: np.ndarray
    """Time of each sample, in ns from the start of the first shot."""
    wave: np.ndarray
    """Complex samples (I + iQ at baseband for SG ports)."""

    @property
    def rate(self) -> float:
        """Sampling rate in GSa/s."""
        return 1 / (self.time[1] - self.time[0])

    def segments(self, threshold: float = 1e-6) -> list[tuple[float, float]]:
        """(start, stop) in ns of each run of non-zero samples."""
        on = np.abs(self.wave) > threshold
        edges = np.diff(on.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        dt = self.time[1] - self.time[0]
        return [(self.time[a], self.time[b - 1] + dt) for a, b in zip(starts, stops)]

    def at(self, start: float, samples: int) -> np.ndarray:
        """``samples`` samples from time ``start`` (ns)."""
        n = int(np.rint((start - self.time[0]) * self.rate))
        return self.wave[n : n + samples]


class Output:
    """LabOne Q ``OutputSimulator`` on the experiment of a capture.

    Pattern: LabOne Q ``OutputSimulator`` ("check experiments even before they are
    executed on hardware"), tried in qibolab 0.1 ``play_sim`` (dropped in 4aaa4163);
    sample-level edge checks as in py-qua-tools ``tests/test_long_wait.py:99-131``.
    """

    def __init__(self, cap: effects.Capture, length: float = 5000):
        self.cap = cap
        self.length = length
        compiled = cap.controller.session.compile(cap.experiment)
        self.simulator = OutputSimulator(compiled)

    def __call__(self, channel) -> Trace:
        """Samples on the port of a qibolab channel (shared with others if muxed)."""
        signal = self.cap.signal_of(channel)
        path = self.cap.experiment.signals[signal].mapped_logical_signal_path
        physical = self.cap.logical_signal(path).physical_channel
        snippet = self.simulator.get_snippet(
            physical, start=0, output_length=self.length * effects.NS
        )
        return Trace(np.asarray(snippet.time) / effects.NS, np.asarray(snippet.wave))


def pulse_samples(p: Pulse, rate: float) -> np.ndarray:
    """Complex samples of a qibolab pulse, amplitude included, phase excluded."""
    return p.i(rate) + 1j * p.q(rate)


def readout(duration=960, amplitude=0.1, phase=0.0):
    """Readout with a rectangular probe; 960 ns keeps every time on a 40 ns grid."""
    return Readout.from_probe(square(duration, amplitude, phase))


def approx_segments(expected):
    return [pytest.approx(s, abs=1e-3) for s in expected]


# amplitude and phase


@pytest.mark.feature("F24")
@kills("phase_dropped", "pulse_on_wrong_channel", "drive_signal_misrouted")
def test_amplitude_and_phase_reach_their_drive(platform):
    """Each drive pulse plays its amplitude and relative phase on its own channel.

    LabOne Q gets the qibolab amplitude and ``relative_phase`` on the signal of that
    drive port, and the port plays samples of that magnitude.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the
    DSL experiment, per signal).
    """
    p0, p1 = square(amplitude=0.3, phase=0.7), square(amplitude=0.2, phase=-1.1)
    sequence = PulseSequence([("q0/drive", p0), ("q1/drive", p1)])
    sequence.append(("q0/acquisition", Delay(duration=40)))
    sequence.append(("q0/acquisition", readout()))

    cap = effects.play(platform, [sequence], nshots=1)

    for channel, p in (("q0/drive", p0), ("q1/drive", p1)):
        [op] = cap.plays(channel)
        assert cap.amplitude(op) == pytest.approx(p.amplitude)
        assert cap.phase(op) == pytest.approx(p.relative_phase)
    output = Output(cap)
    for channel, p in (("q0/drive", p0), ("q1/drive", p1)):
        trace = output(channel)
        samples = trace.at(0, int(p.duration * trace.rate))
        np.testing.assert_allclose(np.abs(samples), p.amplitude, atol=1e-4)


# envelopes


def custom(samples):
    return Custom(i_=np.linspace(0.2, 1.0, samples), q_=np.linspace(-0.5, 0.3, samples))


ENVELOPES = [
    pytest.param(lambda _: Rectangular(), id="F31-rectangular", marks=pytest.mark.feature("F31")),
    pytest.param(lambda _: Gaussian(rel_sigma=0.2), id="F32-gaussian", marks=pytest.mark.feature("F32")),
    pytest.param(lambda _: Drag(rel_sigma=0.2, beta=0.5), id="F33-drag", marks=pytest.mark.feature("F33")),
    pytest.param(lambda _: GaussianSquare(risefall=10, sigma=4), id="F34-gaussian-square", marks=pytest.mark.feature("F34")),
    pytest.param(lambda _: Exponential(tau=0.2, upsilon=0.5), id="F35-exponential", marks=pytest.mark.feature("F35")),
    pytest.param(lambda _: Snz(t_idling=10), id="F36-snz", marks=pytest.mark.feature("F36")),
]  # fmt: skip
"""Envelopes computed for any number of samples. Custom, with fixed samples, is
tested on its own: under ``sampling_rate_hdawg`` qibolab itself refuses it."""


def check_envelope(platform, envelope):
    """Play a 40 ns pulse of ``envelope`` on q0/drive and check its samples."""
    duration = 40
    rate = platform.sampling_rate
    p = Pulse(duration=duration, amplitude=0.4, envelope=envelope)
    sequence = PulseSequence([("q0/drive", p), ("q0/acquisition", Delay(duration=40))])
    sequence.append(("q0/acquisition", readout()))

    cap = effects.play(platform, [sequence], nshots=1)

    [op] = cap.plays("q0/drive")
    np.testing.assert_allclose(cap.waveform(op), pulse_samples(p, rate), atol=1e-12)
    trace = Output(cap)("q0/drive")
    expected = pulse_samples(p, trace.rate)
    played = trace.at(0, len(expected) + 8)
    assert len(played) == len(expected) + 8
    np.testing.assert_allclose(played[: len(expected)], expected, atol=1e-4)
    np.testing.assert_allclose(played[len(expected) :], 0, atol=1e-6)


@pytest.mark.parametrize("envelope", ENVELOPES)
@kills("samples_q_is_i", "sampling_rate_hdawg")
def test_envelope_samples_played(platform, envelope):
    """The drive port plays the envelope's I and Q, sampled at the rate LabOne Q plays.

    Checked twice: the samples handed to LabOne Q (x amplitude) equal
    ``Pulse.i + 1j Pulse.q`` at the driver's sampling rate, and the samples the
    instrument port plays equal them at the port's rate, from t = 0, then zeros.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (DSL values);
    LabOne Q ``OutputSimulator`` sample checks (qibolab 0.1 ``play_sim``, 4aaa4163;
    research note driver-tests/README.md C1); qibolab 0.1
    tests/test_instruments_zhinst.py ``test_zhpulse_pulse_conversion`` (each shape
    becomes 80 samples for 40 ns).
    """
    check_envelope(platform, envelope(80))


@pytest.mark.feature("F37")
@kills("samples_q_is_i", "samples_iq_swapped")
def test_custom_envelope_samples_played(platform):
    """A Custom envelope of 80 samples plays exactly those samples over 40 ns.

    Its I and Q differ and are not symmetric, so a swap, a copy or a reversal shows.

    Pattern: as ``test_envelope_samples_played``.
    """
    check_envelope(platform, custom(80))


@pytest.mark.feature("F33")
@kills("samples_q_is_i", "samples_iq_swapped")
def test_iq_not_swapped(platform):
    """I of the envelope goes to the real part, Q to the imaginary part, unchanged.

    A DRAG pulse has different I and Q; the samples handed to LabOne Q and those the
    drive port plays keep I as I and Q as Q (relative phase 0).

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; LabOne Q
    ``OutputSimulator`` sample checks (driver-tests/README.md C1).
    """
    p = Pulse(duration=40, amplitude=0.5, envelope=Drag(rel_sigma=0.2, beta=2.0))
    sequence = PulseSequence([("q0/drive", p), ("q0/acquisition", Delay(duration=40))])
    sequence.append(("q0/acquisition", readout()))

    cap = effects.play(platform, [sequence], nshots=1)

    [op] = cap.plays("q0/drive")
    rate = platform.sampling_rate
    samples = cap.waveform(op)
    np.testing.assert_allclose(samples.real, p.i(rate), atol=1e-12)
    np.testing.assert_allclose(samples.imag, p.q(rate), atol=1e-12)
    trace = Output(cap)("q0/drive")
    played = trace.at(0, len(p.i(trace.rate)))
    np.testing.assert_allclose(played.real, p.i(trace.rate), atol=1e-4)
    np.testing.assert_allclose(played.imag, p.q(trace.rate), atol=1e-4)


# duration


@pytest.mark.feature("F31")
@pytest.mark.parametrize("duration", [36, 100, 1000])
@kills("sampling_rate_hdawg")
def test_duration_played(platform, duration):
    """A pulse of ``duration`` ns lasts ``duration`` ns on the drive port.

    Also off the 16-sample grid (36 ns = 72 samples). Read from what the port plays.

    Pattern: py-qua-tools ``tests/test_long_wait.py:99-131`` (edge timing on simulated
    samples), on LabOne Q's ``OutputSimulator``.
    """
    sequence = PulseSequence([("q0/drive", square(duration))])
    sequence.append(("q0/acquisition", Delay(duration=duration)))
    sequence.append(("q0/acquisition", readout()))

    cap = effects.play(platform, [sequence], nshots=1)

    trace = Output(cap)("q0/drive")
    assert trace.segments() == approx_segments([(0, duration)])


# timing: Delay and Align


@pytest.mark.feature("F24")
@kills("delays_dropped", "probe_delay_dropped")
def test_delay_shifts_pulse_and_readout(platform):
    """A Delay postpones what follows on its channel; a readout starts as a whole.

    q0/drive: Delay 24, pulse 40 -> the pulse plays from 24 to 64 ns.
    q0/acquisition: Delay 64, Readout -> probe and acquisition both start at 64 ns.

    Pattern: py-qua-tools ``tests/test_long_wait.py:99-131`` (edges on simulated
    samples); laboneq-applications ``CompiledExperimentVerifier.assert_pulse``
    (start/end per signal), here on ``OutputSimulator`` traces.
    """
    ro = readout()
    sequence = PulseSequence(
        [
            ("q0/drive", Delay(duration=24)),
            ("q0/drive", square(40)),
            ("q0/acquisition", Delay(duration=64)),
            ("q0/acquisition", ro),
        ]
    )

    cap = effects.play(platform, [sequence], nshots=1)

    output = Output(cap)
    assert output("q0/drive").segments() == approx_segments([(24, 64)])
    assert output("q0/probe").segments() == approx_segments([(64, 64 + 960)])
    [(start, _)] = output("q0/acquisition").segments()
    assert start == pytest.approx(64)


@pytest.mark.feature("F26")
@kills("align_ignored", "delays_dropped")
def test_align_synchronises_channels(platform):
    """After Align, the aligned channels continue together from the latest one.

    q0/drive: pulse 40. q1/drive: Delay 20, pulse 60 (ends at 80).
    Align(q0/drive, q1/drive, q0/acquisition), then a second pulse on q0/drive and
    a readout of q0: both start at 80 ns.

    Pattern: qibolab ``PulseSequence.align`` / ``align_to_delays`` semantics; edges on
    LabOne Q ``OutputSimulator`` traces (py-qua-tools ``tests/test_long_wait.py``).
    """
    sequence = PulseSequence(
        [
            ("q0/drive", square(40)),
            ("q1/drive", Delay(duration=20)),
            ("q1/drive", square(60, amplitude=0.2)),
        ]
    )
    sequence.align(["q0/drive", "q1/drive", "q0/acquisition"])
    sequence.append(("q0/drive", square(40, amplitude=0.25)))
    sequence.append(("q0/acquisition", readout()))

    cap = effects.play(platform, [sequence], nshots=1)

    output = Output(cap)
    assert output("q1/drive").segments() == approx_segments([(20, 80)])
    assert output("q0/drive").segments() == approx_segments([(0, 40), (80, 120)])
    assert output("q0/probe").segments() == approx_segments([(80, 80 + 960)])
    [(start, _)] = output("q0/acquisition").segments()
    assert start == pytest.approx(80)


# several sequences


@pytest.mark.feature("F29")
@kills("sequences_reversed", "relaxation_zero")
def test_several_sequences_in_order(platform):
    """Each sequence plays, in the order given, each followed by the relaxation time.

    Sequence A drives q0 at amplitude 0.1, sequence B at 0.2; both read q0 out.
    In each shot: A's pulse at 0, A's readout at 40; B starts after A (1000 ns) and the
    relaxation time (1000 ns): B's pulse at 2000, B's readout at 2040. The
    acquisitions on q0 carry A's handle, then B's.

    Pattern: qibolab ``ExecutionParameters.estimate_duration`` (duration + one
    relaxation per sequence); zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py; edges on LabOne Q ``OutputSimulator`` traces.
    """
    sequences = []
    for amplitude in (0.1, 0.2):
        sequence = PulseSequence([("q0/drive", square(40, amplitude))])
        sequence.append(("q0/acquisition", Delay(duration=40)))
        sequence.append(("q0/acquisition", readout()))
        sequences.append(sequence)

    cap = effects.play(platform, sequences, nshots=1, relaxation_time=1000)

    plays = cap.plays("q0/drive")
    assert [cap.amplitude(op) for op in plays] == [
        pytest.approx(0.1),
        pytest.approx(0.2),
    ]
    assert cap.handles("q0/acquisition") == [
        str(ro.id) for s in sequences for _, ro in s.acquisitions
    ]
    output = Output(cap)
    drive = output("q0/drive")
    assert drive.segments() == approx_segments([(0, 40), (2000, 2040)])
    n = int(40 * drive.rate)
    np.testing.assert_allclose(np.abs(drive.at(0, n)), 0.1, atol=1e-4)
    np.testing.assert_allclose(np.abs(drive.at(2000, n)), 0.2, atol=1e-4)
    starts = [start for start, _ in output("q0/acquisition").segments()]
    assert starts == [pytest.approx(40), pytest.approx(2040)]


# readout


@pytest.mark.feature("F22")
@kills("samples_q_is_i", "sampling_rate_hdawg")
def test_readout_is_probe_and_acquisition(platform):
    """A Readout plays its probe on the probe channel and acquires on its channel.

    Probe: amplitude and samples of the probe pulse on q0/probe, for the probe's
    duration. Acquisition: one, on q0/acquisition, with the readout id as handle and
    an integration of the acquisition's duration. Both start together.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; LabOne Q
    ``OutputSimulator`` traces of the QA output and input.
    """
    probe = Pulse(
        duration=960, amplitude=0.15, envelope=GaussianSquare(risefall=40, sigma=10)
    )
    ro = Readout.from_probe(probe)
    sequence = PulseSequence([("q0/acquisition", ro)])

    cap = effects.play(platform, [sequence], nshots=1)

    rate = platform.sampling_rate
    [play] = cap.plays("q0/probe")
    np.testing.assert_allclose(
        cap.waveform(play), pulse_samples(probe, rate), atol=1e-12
    )
    [acquire] = cap.acquires("q0/acquisition")
    assert acquire.op.handle == str(ro.id)
    assert cap.duration(acquire) == pytest.approx(ro.acquisition.duration)
    assert cap.acquires() == [acquire]
    assert cap.plays() == [play]
    output = Output(cap)
    probe_port = output("q0/probe")
    np.testing.assert_allclose(
        np.abs(probe_port.at(0, int(960 * probe_port.rate))),
        np.abs(pulse_samples(probe, probe_port.rate)),
        atol=1e-4,
    )
    assert probe_port.segments() == approx_segments([(0, 960)])
    [(start, _)] = output("q0/acquisition").segments()
    assert start == pytest.approx(0)
    assert set(cap.results) == {ro.id}


@pytest.mark.feature("F23")
@kills(
    "readout_probe_swapped",
    "acquisition_handles_swapped",
    "phase_dropped",
    "probe_phase_dropped",
)
def test_multiplexed_readouts_on_their_channels(platform):
    """Readouts of two qubits: each probe and acquisition on that qubit's channels.

    q0 probes at amplitude 0.1 and phase 0.4, q1 at 0.2 and phase -0.9. Each probe
    channel plays its own amplitude and relative phase; each acquisition channel
    acquires with its own readout's handle; all start together.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (per-signal
    DSL checks).
    """
    ro0, ro1 = readout(amplitude=0.1, phase=0.4), readout(amplitude=0.2, phase=-0.9)
    sequence = PulseSequence([("q0/acquisition", ro0), ("q1/acquisition", ro1)])

    cap = effects.play(platform, [sequence], nshots=1)

    for q, ro in (("q0", ro0), ("q1", ro1)):
        [play] = cap.plays(f"{q}/probe")
        assert cap.amplitude(play) == pytest.approx(ro.probe.amplitude)
        assert cap.phase(play) == pytest.approx(ro.probe.relative_phase)
        assert cap.handles(f"{q}/acquisition") == [str(ro.id)]
    output = Output(cap)
    for q in ("q0", "q1"):
        [(start, _)] = output(f"{q}/acquisition").segments()
        assert start == pytest.approx(0)
