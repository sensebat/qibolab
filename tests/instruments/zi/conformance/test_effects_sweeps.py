"""Effects of qibolab sweepers, observed at the LabOne Q boundary.

Each test runs ``ZiController.play`` with sweepers and checks what LabOne Q is asked
to do, in qibolab terms: which channel's frequency or which pulse's amplitude takes
which values, in which order, bound to which other sweeper, and where each sweep
point lands in the qibolab results. Section names, uids and nesting are not read; a
nested grid is checked through the results only.

Features F05-F12 (flux amplitude, duration, phase, offset sweeps) are refused by
ElStabilini/qibolab@zi_drivers (e8757e58); their strict xfails in
``test_conformance.py`` cover them.

The injected faults, among them ``nested_sweepers_swapped`` and
``results_axes_not_moved`` for sweeper nesting and the results axes, are in
:mod:`mutants`.
"""

import zlib
from contextlib import contextmanager

import numpy as np
import pytest

from qibolab._core.pulses import Drag, Pulse
from qibolab._core.sweeper import Parameter, Sweeper

from . import effects
from .effects import drives_then_mz, iq, square
from .mutants import kills
from .test_conformance import mz, readout_of

# local helpers


def frequency_sweeper(platform, channel, steps):
    """Frequency sweeper around the channel's frequency, in the order of ``steps``."""
    f0 = platform.config(channel).frequency
    return Sweeper(
        parameter=Parameter.frequency,
        values=f0 + np.asarray(steps, dtype=float),
        channels=[channel],
    )


def amplitude_sweeper(pulses, values):
    return Sweeper(
        parameter=Parameter.amplitude, values=np.asarray(values), pulses=pulses
    )


def exactly(ops, count, channel):
    """The ``count`` operations on ``channel``, failing as a test if there are not."""
    assert len(ops) == count, f"{len(ops)} operations on {channel}, not {count}"
    return ops


def stepped_together(cap, parameters):
    """Whether one LabOne Q sweep runs all ``parameters``, step by step."""
    loops = [cap.sweep_of(p) for p in parameters]
    uids = [[p.uid for p in loop.parameters] for loop in loops]
    return all(u == uids[0] for u in uids)


class _Labeled:
    """LabOne Q results whose data are distinct at every point and for every handle."""

    def __init__(self, results):
        self._results = results

    def __getattr__(self, name):
        return getattr(self._results, name)

    def get_data(self, handle):
        shape = np.shape(self._results.get_data(handle))
        index = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
        offset = 1e4 * (zlib.crc32(str(handle).encode()) % 1000)
        return (offset + index) - 1j * (index + 1)


@contextmanager
def labeled(platform):
    """Replace the data LabOne Q returns by distinct labels at each point.

    The emulator returns the same number at every sweep point, so it cannot show
    where a point lands in the qibolab results. The labels can.
    """
    session = platform.instruments["zi"].session
    run = session.run
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(session, "run", lambda *a, **k: _Labeled(run(*a, **k)))
        yield


def _axis_of(names, parameter):
    """LabOne Q results axis that runs ``parameter``."""
    if not hasattr(parameter, "uid"):
        raise effects.Missing(f"{parameter!r} is not swept.")
    for m, name in enumerate(names):
        if parameter.uid == name or parameter.uid in np.atleast_1d(name):
            return m
    raise effects.Missing(f"No results axis runs '{parameter.uid}' (axes {names}).")


def assert_grid(cap, acquisition, readout, axes):
    """The qibolab result at each sweep point is LabOne Q's data at that point.

    ``axes``, one per qibolab sweeper group in order: ``(values, parameter,
    played)`` with the qibolab sweeper values, the LabOne Q sweep parameter that
    runs them, and what LabOne Q plays at each of its steps (an effect read through
    :class:`effects.Capture`, in LabOne Q sweep order). Averaged (cyclic) data.
    """
    [handle] = exactly(cap.handles(acquisition), 1, acquisition)
    raw = np.asarray(cap.raw.get_data(handle))
    names = list(cap.raw.get_axis_name(handle))
    shape = tuple(len(values) for values, _, _ in axes)
    result = cap.results[readout.id]
    assert result.shape == shape + (2,), f"results shape {result.shape}"
    expected = np.empty(shape, dtype=complex)
    for point in np.ndindex(shape):
        index = [None] * len(names)
        for k, i in enumerate(point):
            values, parameter, played = axes[k]
            [steps] = np.nonzero(
                np.isclose(np.atleast_1d(played), values[i], rtol=1e-12, atol=1e-12)
            )
            assert len(steps) == 1, (
                f"sweeper {k} value {values[i]} played at {len(steps)} LabOne Q steps"
            )
            index[_axis_of(names, parameter)] = steps[0]
        assert None not in index, f"LabOne Q axes {names} not all swept"
        expected[point] = raw[tuple(index)]
    np.testing.assert_array_equal(result, iq(expected))


# F01, F02: frequency sweeps


@pytest.mark.feature("F01")
@kills("sweep_grid_reversed", "frequency_sweep_offset")
def test_probe_frequency_sweep_values(platform):
    """A frequency sweeper on q0/probe makes LabOne Q step q0's probe and its
    demodulation (q0/acquisition) through exactly the sweeper values, in their order.
    The LO is not swept: only the oscillator is.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (sweep
    parameters on the DSL experiment); absolute vs offset frequency sweeps from the
    QM audit of qibolab (UriG-qm/qibolab qm-qa/* branches).
    """
    sequence = mz(platform, "q0", "q1")
    sweeper = frequency_sweeper(platform, "q0/probe", [1e6, -2e6, 0.0, 2e6, -1e6])

    cap = effects.play(platform, [sequence], [[sweeper]])

    np.testing.assert_allclose(cap.frequency("q0/probe"), sweeper.values, rtol=1e-12)
    np.testing.assert_allclose(
        cap.frequency("q0/acquisition"), sweeper.values, rtol=1e-12
    )
    lo = platform.config(platform.channels["q0/probe"].lo).frequency
    assert cap.lo("q0/probe") == pytest.approx(lo)
    parameter = cap.oscillator("q0/probe").frequency
    driven = {(d.channel, d.quantity) for d in cap.driven_by(parameter)}
    assert driven == {("q0/probe", "frequency"), ("q0/acquisition", "frequency")}


@pytest.mark.feature("F01")
@kills("lo_sign_flipped", "lo_not_subtracted")
def test_probe_frequency_sweep_keeps_other_probe(platform):
    """Sweeping q0/probe leaves q1/probe, on the same SHFQA port and LO, at its
    configured frequency, and leaves the drives at theirs.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (signal
    calibration on the DSL experiment); absolute vs offset frequency sweeps from the
    QM audit of qibolab (UriG-qm/qibolab qm-qa/* branches).
    """
    sequence = drives_then_mz(platform, {"q0": [square()], "q1": [square()]})
    sweeper = frequency_sweeper(platform, "q0/probe", [-2e6, -1e6, 0.0, 1e6])

    cap = effects.play(platform, [sequence], [[sweeper]])

    for channel in ("q1/probe", "q1/acquisition"):
        assert cap.frequency(channel) == pytest.approx(
            platform.config("q1/probe").frequency, rel=1e-12
        )
    for channel in ("q0/drive", "q1/drive"):
        assert cap.frequency(channel) == pytest.approx(
            platform.config(channel).frequency, rel=1e-12
        )


@pytest.mark.feature("F02")
@kills(
    "sweep_grid_reversed",
    "frequency_sweep_offset",
    "drive_signal_misrouted",
    "lo_sign_flipped",
    "lo_not_subtracted",
)
def test_drive_frequency_sweep_values(platform):
    """A frequency sweeper on q0/drive makes LabOne Q step the oscillator of the
    signal on q0's drive port through exactly the sweeper values, in their order, on
    top of an unswept LO. q1/drive and q0/probe keep their configured frequencies.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (sweep
    parameters on the DSL experiment); absolute vs offset frequency sweeps from the
    QM audit of qibolab (UriG-qm/qibolab qm-qa/* branches).
    """
    sequence = drives_then_mz(platform, {"q0": [square()], "q1": [square()]})
    sweeper = frequency_sweeper(platform, "q0/drive", [3e6, -1e6, 0.0, -3e6, 1e6])

    cap = effects.play(platform, [sequence], [[sweeper]])

    np.testing.assert_allclose(cap.frequency("q0/drive"), sweeper.values, rtol=1e-12)
    lo = platform.config(platform.channels["q0/drive"].lo).frequency
    assert cap.lo("q0/drive") == pytest.approx(lo)
    parameter = cap.oscillator("q0/drive").frequency
    driven = {(d.channel, d.quantity) for d in cap.driven_by(parameter)}
    assert driven == {("q0/drive", "frequency")}
    for channel, config in (("q1/drive", "q1/drive"), ("q0/probe", "q0/probe")):
        assert cap.frequency(channel) == pytest.approx(
            platform.config(config).frequency, rel=1e-12
        )


@pytest.mark.feature("F02")
@kills("sweep_grid_reversed", "frequency_sweep_offset", "results_q_is_i")
def test_drive_frequency_sweep_results(platform):
    """The qibolab result at step i of a drive frequency sweep is what LabOne Q
    measured while the drive was at the sweeper's value i.

    Pattern: data injected at the result seam, as in qblox-scheduler
    set_dummy_binned_acquisition_data and zhinst/laboneq-applications v26.4.0
    inject_results; zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = drives_then_mz(platform, {"q0": [square()]})
    sweeper = frequency_sweeper(platform, "q0/drive", [2e6, 0.0, -2e6, 1e6])

    with labeled(platform):
        cap = effects.play(platform, [sequence], [[sweeper]])

    axes = [
        (
            sweeper.values,
            cap.oscillator("q0/drive").frequency,
            cap.frequency("q0/drive"),
        )
    ]
    assert_grid(cap, "q0/acquisition", readout_of(sequence), axes)


# F03, F04: amplitude sweeps


@pytest.mark.feature("F03")
@kills("amplitude_sweep_ignored", "sweep_grid_reversed")
def test_probe_amplitude_sweep(platform):
    """An amplitude sweeper on q0's readout makes LabOne Q play q0's probe at
    exactly the sweeper values, in their order; q1's probe, multiplexed on the same
    port, keeps its amplitude; the integration kernels are not scaled.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (play
    amplitude on the DSL experiment).
    """
    sequence = mz(platform, "q0", "q1")
    sweeper = amplitude_sweeper([readout_of(sequence, 0)], [0.5, 0.1, 0.9, 0.3])

    cap = effects.play(platform, [sequence], [[sweeper]])

    [q0] = exactly(cap.plays("q0/probe"), 1, "q0/probe")
    np.testing.assert_allclose(cap.amplitude(q0), sweeper.values, rtol=1e-12)
    [q1] = exactly(cap.plays("q1/probe"), 1, "q1/probe")
    probe = readout_of(sequence, 1).probe
    assert cap.amplitude(q1) == pytest.approx(probe.amplitude)
    driven = {(d.channel, d.quantity) for d in cap.driven_by(q0.op.amplitude)}
    assert driven == {("q0/probe", "amplitude")}


@pytest.mark.feature("F04")
@kills(
    "amplitude_sweep_ignored",
    "sweep_grid_reversed",
    "phase_dropped",
    "samples_q_is_i",
    "pulse_on_wrong_channel",
    "drive_signal_misrouted",
)
def test_drive_amplitude_sweep(platform):
    """An amplitude sweeper on one drive pulse makes LabOne Q play that pulse's
    waveform (shape and relative phase kept) scaled by exactly the sweeper values,
    in their order. The next pulse on q0/drive and the pulse on q1/drive keep their
    amplitudes.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (play
    amplitude and phase on the DSL experiment).
    """
    swept = Pulse(
        duration=40,
        amplitude=0.5,
        envelope=Drag(rel_sigma=0.2, beta=0.5),
        relative_phase=0.7,
    )
    after = square(amplitude=0.25)
    other = square(amplitude=0.15, duration=48)
    sequence = drives_then_mz(platform, {"q0": [swept, after], "q1": [other]})
    sweeper = amplitude_sweeper([swept], [0.3, 0.9, 0.1, 0.6])

    cap = effects.play(platform, [sequence], [[sweeper]])

    first, second = exactly(cap.plays("q0/drive"), 2, "q0/drive")
    sr = platform.sampling_rate
    shape = (swept.i(sr) + 1j * swept.q(sr)) / swept.amplitude
    expected = np.multiply.outer(sweeper.values * np.exp(1j * 0.7), shape)
    np.testing.assert_allclose(cap.waveform(first), expected, atol=1e-12)
    assert cap.amplitude(second) == pytest.approx(0.25)
    [q1] = exactly(cap.plays("q1/drive"), 1, "q1/drive")
    assert cap.amplitude(q1) == pytest.approx(0.15)
    assert cap.duration(q1) == pytest.approx(48)


# F13: parallel sweepers


@pytest.mark.feature("F13")
@kills("sweep_grid_reversed", "frequency_sweep_offset", "results_q_is_i")
def test_parallel_probe_frequency_sweepers(platform):
    """Two frequency sweepers in parallel, on q0/probe and q1/probe, make LabOne Q
    step both probes together, each through its own values; each qubit's result at
    step i is measured at both probes' value i.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (sweep
    parameters on the DSL experiment); data injected at the result seam, as in
    qblox-scheduler set_dummy_binned_acquisition_data and laboneq-applications
    inject_results.
    """
    sequence = mz(platform, "q0", "q1")
    sweepers = [
        frequency_sweeper(platform, "q0/probe", [-2e6, -1e6, 0.0, 1e6]),
        frequency_sweeper(platform, "q1/probe", [3e6, 1e6, -1e6, -3e6]),
    ]

    with labeled(platform):
        cap = effects.play(platform, [sequence], [sweepers])

    parameters = [cap.oscillator(f"{q}/probe").frequency for q in ("q0", "q1")]
    for q, sweeper in zip(("q0", "q1"), sweepers):
        np.testing.assert_allclose(
            cap.frequency(f"{q}/probe"), sweeper.values, rtol=1e-12
        )
    assert stepped_together(cap, parameters), "parallel sweepers are not bound"
    for n, q in enumerate(("q0", "q1")):
        axes = [(sweepers[n].values, parameters[n], cap.frequency(f"{q}/probe"))]
        assert_grid(cap, f"{q}/acquisition", readout_of(sequence, n), axes)


@pytest.mark.feature("F13")
@kills(
    "sweep_grid_reversed",
    "frequency_sweep_offset",
    "drive_signal_misrouted",
)
def test_parallel_drive_frequency_sweepers(platform):
    """Two frequency sweepers in parallel, on q0/drive and q1/drive, make LabOne Q
    step both drives together, each through its own values.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (sweep
    parameters on the DSL experiment).
    """
    sequence = drives_then_mz(platform, {"q0": [square()], "q1": [square()]})
    sweepers = [
        frequency_sweeper(platform, "q0/drive", [-2e6, -1e6, 0.0, 1e6]),
        frequency_sweeper(platform, "q1/drive", [5e6, 3e6, 1e6, -1e6]),
    ]

    cap = effects.play(platform, [sequence], [sweepers])

    for q, sweeper in zip(("q0", "q1"), sweepers):
        np.testing.assert_allclose(
            cap.frequency(f"{q}/drive"), sweeper.values, rtol=1e-12
        )
    parameters = [cap.oscillator(f"{q}/drive").frequency for q in ("q0", "q1")]
    assert stepped_together(cap, parameters), "parallel sweepers are not bound"


# F14: nested sweepers


@pytest.mark.feature("F14")
@kills("sweep_grid_reversed", "frequency_sweep_offset", "nested_sweepers_swapped")
def test_nested_probe_sweepers_grid(platform):
    """Amplitude (outer) and frequency (inner) sweepers on q0's readout, as in
    resonator_punchout: the qibolab result at (i, j) is LabOne Q's data at the point
    where the probe plays amplitude i and frequency j.

    Pattern: data injected at the result seam, as in qblox-scheduler
    set_dummy_binned_acquisition_data and zhinst/laboneq-applications v26.4.0
    inject_results; qibolab tests/test_result_shapes.py (axis order).
    """
    sequence = mz(platform)
    amplitude = amplitude_sweeper([readout_of(sequence)], [0.1, 0.4, 0.7])
    frequency = frequency_sweeper(platform, "q0/probe", [-2e6, -1e6, 0.0, 1e6])

    with labeled(platform):
        cap = effects.play(platform, [sequence], [[amplitude], [frequency]])

    [probe] = exactly(cap.plays("q0/probe"), 1, "q0/probe")
    axes = [
        (amplitude.values, probe.op.amplitude, cap.amplitude(probe)),
        (
            frequency.values,
            cap.oscillator("q0/probe").frequency,
            cap.frequency("q0/probe"),
        ),
    ]
    assert_grid(cap, "q0/acquisition", readout_of(sequence), axes)


@pytest.mark.feature("F14")
@kills(
    "sweep_grid_reversed",
    "nested_sweepers_swapped",
    "results_axes_not_moved",
    "results_q_is_i",
)
def test_nested_drive_sweepers_grid(platform):
    """Drive frequency (outer) and drive amplitude (inner) sweepers, of equal
    length: the qibolab result at (i, j) is LabOne Q's data at the point where q0's
    drive is at frequency i and plays amplitude j, whatever loop order LabOne Q
    runs them in.

    Pattern: data injected at the result seam, as in qblox-scheduler
    set_dummy_binned_acquisition_data and zhinst/laboneq-applications v26.4.0
    inject_results; qibolab tests/test_result_shapes.py (axis order).
    """
    pulse = square(amplitude=0.5)
    sequence = drives_then_mz(platform, {"q0": [pulse]})
    frequency = frequency_sweeper(platform, "q0/drive", [-1e6, 0.0, 2e6])
    amplitude = amplitude_sweeper([pulse], [0.2, 0.8, 0.5])

    with labeled(platform):
        cap = effects.play(platform, [sequence], [[frequency], [amplitude]])

    [drive] = exactly(cap.plays("q0/drive"), 1, "q0/drive")
    axes = [
        (
            frequency.values,
            cap.oscillator("q0/drive").frequency,
            cap.frequency("q0/drive"),
        ),
        (amplitude.values, drive.op.amplitude, cap.amplitude(drive)),
    ]
    assert_grid(cap, "q0/acquisition", readout_of(sequence), axes)


# F15: one sweeper, several pulses


@pytest.mark.feature("F15")
@kills("amplitude_sweep_ignored", "sweep_grid_reversed", "results_q_is_i")
def test_one_amplitude_sweeper_two_probes(platform):
    """One amplitude sweeper over the readouts of q0 and q1, as in
    resonator_punchout: LabOne Q plays both probes at the sweeper values, stepped
    together, and each qubit's result at step i is measured at amplitude i.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (play
    amplitude on the DSL experiment); data injected at the result seam, as in
    qblox-scheduler set_dummy_binned_acquisition_data.
    """
    sequence = mz(platform, "q0", "q1")
    sweeper = amplitude_sweeper(
        [ro for _, ro in sequence.acquisitions], [0.6, 0.2, 0.8, 0.4]
    )

    with labeled(platform):
        cap = effects.play(platform, [sequence], [[sweeper]])

    probes = [cap.plays(f"{q}/probe")[0] for q in ("q0", "q1")]
    for probe in probes:
        np.testing.assert_allclose(cap.amplitude(probe), sweeper.values, rtol=1e-12)
    parameters = [probe.op.amplitude for probe in probes]
    assert stepped_together(cap, parameters), "the probes are not stepped together"
    for n, (q, probe) in enumerate(zip(("q0", "q1"), probes)):
        axes = [(sweeper.values, probe.op.amplitude, cap.amplitude(probe))]
        assert_grid(cap, f"{q}/acquisition", readout_of(sequence, n), axes)


@pytest.mark.feature("F15")
@kills(
    "amplitude_sweep_ignored",
    "sweep_grid_reversed",
    "pulse_on_wrong_channel",
    "drive_signal_misrouted",
)
def test_one_amplitude_sweeper_two_drives(platform):
    """One amplitude sweeper over a pulse on q0/drive and a longer one on q1/drive,
    as in rabi_amplitude on two qubits: each drive port plays its own pulse at the
    sweeper values, stepped together.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (play
    amplitude on the DSL experiment).
    """
    pulses = {"q0": square(amplitude=0.5, duration=40), "q1": square(64, amplitude=0.5)}
    sequence = drives_then_mz(platform, {q: [p] for q, p in pulses.items()})
    sweeper = amplitude_sweeper(list(pulses.values()), [0.3, 0.9, 0.1, 0.6])

    cap = effects.play(platform, [sequence], [[sweeper]])

    plays = {}
    for q, pulse in pulses.items():
        [plays[q]] = exactly(cap.plays(f"{q}/drive"), 1, f"{q}/drive")
        np.testing.assert_allclose(cap.amplitude(plays[q]), sweeper.values, rtol=1e-12)
        assert cap.duration(plays[q]) == pytest.approx(pulse.duration)
    parameters = [op.op.amplitude for op in plays.values()]
    assert stepped_together(cap, parameters), "the drives are not stepped together"
