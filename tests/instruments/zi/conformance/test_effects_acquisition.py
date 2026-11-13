"""Effects of acquisition and results, observed at the LabOne Q boundary.

What a ``ZiController.play`` call must make LabOne Q do, by the qibolab interface:

- ``AcquisitionType.INTEGRATION`` -> an acquisition loop in INTEGRATION mode, each
  acquisition integrating with unit weights over ``Readout.acquisition.duration``;
- ``AveragingMode.CYCLIC`` / ``SINGLESHOT`` -> the same LabOne Q averaging mode;
- ``nshots`` -> the count of the acquisition loop;
- ``relaxation_time`` -> a wait of that time after the readout, in every shot and after
  every sequence;
- ``AcquisitionConfig.delay`` -> the acquisition starts that much after the probe;
- the result of a readout is the data LabOne Q acquired on the signal of that
  readout's acquisition channel, I in ``[..., 0]`` and Q in ``[..., 1]``, with the
  axes in qibolab order (shots, then one per sweeper) and point ``i`` of a sweeper axis
  measured at ``sweeper.values[i]``.

The results are checked on the LabOne Q emulator's placeholder data: the real part is
42 + integrator index (divided by the averages), so it differs between the
acquisitions of two qubits; the imaginary part is the index of the point within one
real-time run (shots, then real-time sweeps), so it differs along those axes. Along a
near-time sweep the data repeat, and only the sweep values LabOne Q reports for the
axis (``Results.get_axis``) tell the points apart.

The injected faults are in :mod:`mutants`.
"""

import laboneq.simple as lq
import numpy as np
import pytest

from qibolab._core.execution_parameters import AcquisitionType, AveragingMode
from qibolab._core.pulses import Acquisition, Delay, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from . import effects
from .effects import Missing, iq, square
from .mutants import kills
from .test_conformance import frequency_sweeper, mz, updated

# local helpers


def acquisition_delay(cap: effects.Capture, q: str) -> float:
    """Time in ns between the start of the probe and of the acquisition of qubit ``q``.

    Difference of the port delays of the two signals.
    """
    return cap.port_delay(f"{q}/acquisition") - cap.port_delay(f"{q}/probe")


def axis_meanings(cap: effects.Capture, handle: str) -> list[dict]:
    """What each axis of the LabOne Q data of ``handle`` runs over.

    One dict per axis: ``{"shots": indices}`` for the shots axis, otherwise
    ``{(channel, quantity): values}`` for each quantity a parameter of the axis drives,
    in qibolab units (a frequency is LO + oscillator). The values are those LabOne Q
    reports with the data (``Results.get_axis``), in data order.
    """
    names = cap.raw.get_axis_name(handle)
    points = cap.raw.get_axis(handle)
    parameters = {p.uid: p for loop in cap.sweeps() for p in loop.parameters}
    meanings = []
    for name, values in zip(names, points):
        if name == cap.acquire_loop.uid:
            meanings.append({"shots": np.asarray(values)})
            continue
        names_, values_ = (
            (name, values) if isinstance(name, list) else ([name], [values])
        )
        meaning = {}
        for uid, axis in zip(names_, values_):
            if uid not in parameters:
                raise RuntimeError(f"Axis '{uid}' is no sweep parameter.")
            axis = np.asarray(axis)
            for driven in cap.driven_by(parameters[uid]):
                if driven.quantity == "frequency":
                    meaning[(driven.channel, "frequency")] = (
                        cap.lo(driven.channel) + axis
                    )
                elif driven.quantity == "amplitude":
                    swept = effects.values(driven.op.op.amplitude)
                    factor = np.asarray(cap.amplitude(driven.op)) / swept
                    meaning[(driven.channel, "amplitude")] = axis * factor
        meanings.append(meaning)
    return meanings


def expected_result(cap: effects.Capture, handle: str, axes: list) -> np.ndarray:
    """LabOne Q data of ``handle`` arranged as qibolab must return it.

    ``axes`` lists the qibolab axes in order: ``"shots"``, or ``(channel, quantity,
    values)`` for a sweeper. Point ``i`` of a sweeper axis is taken from the LabOne Q
    point where the quantity has ``values[i]``. Raises :class:`effects.Missing` if a
    value is not on the LabOne Q axis or an axis is unaccounted for.
    """
    data = np.asarray(cap.raw.get_data(handle))
    meanings = axis_meanings(cap, handle)
    order, index = [], []
    for axis in axes:
        key = "shots" if axis == "shots" else axis[:2]
        found = [n for n, meaning in enumerate(meanings) if key in meaning]
        if len(found) != 1:
            raise Missing(f"{key} runs on {len(found)} LabOne Q axes.")
        points = meanings[found[0]][key]
        if axis == "shots":
            index.append(np.arange(len(points)))
        else:
            tolerance = 1.0 if axis[1] == "frequency" else 1e-9
            picks = []
            for value in np.asarray(axis[2]):
                at = np.flatnonzero(np.abs(points - value) <= tolerance)
                if len(at) != 1:
                    raise Missing(f"{key} = {value} is at {len(at)} LabOne Q points.")
                picks.append(at[0])
            index.append(np.asarray(picks))
        order.append(found[0])
    if sorted(order) != list(range(data.ndim)):
        raise Missing(f"LabOne Q axes {meanings} are not the qibolab axes {axes}.")
    return np.transpose(data, order)[np.ix_(*index)]


def readout(acquisition=1024, probe=1024):
    return Readout(
        acquisition=Acquisition(duration=acquisition),
        probe=Pulse(duration=probe, amplitude=0.1, envelope=Rectangular()),
    )


# acquisition type and averaging


@pytest.mark.feature("F16")
@kills("acquisition_type_discrimination", "samples_q_is_i")
def test_integration_reaches_labone_q(platform):
    """``AcquisitionType.INTEGRATION`` -> LabOne Q integrates, with unit weights.

    Pattern: qibolab tests/test_result_shapes.py (acquisition type x averaging mode,
    driver independent), read at the LabOne Q boundary as in zhinst/laboneq-applications
    v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform, "q0", "q1")

    cap = effects.play(
        platform, [sequence], acquisition_type=AcquisitionType.INTEGRATION
    )

    assert cap.acquisition_type == lq.AcquisitionType.INTEGRATION
    for q in ("q0", "q1"):
        [op] = cap.acquires(f"{q}/acquisition")
        np.testing.assert_allclose(cap.samples(op), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("averaging_mode", "labone_q", "shape"),
    [
        pytest.param(
            AveragingMode.CYCLIC,
            lq.AveragingMode.CYCLIC,
            (2,),
            id="F19",
            marks=pytest.mark.feature("F19"),
        ),
        pytest.param(
            AveragingMode.SINGLESHOT,
            lq.AveragingMode.SINGLE_SHOT,
            (5, 2),
            id="F20",
            marks=pytest.mark.feature("F20"),
        ),
    ],
)
@kills("averaging_mode_swapped", "results_q_is_i")
def test_averaging_mode_reaches_labone_q(platform, averaging_mode, labone_q, shape):
    """The averaging mode reaches the acquisition loop; the result is LabOne Q's data.

    CYCLIC: one averaged I/Q pair. SINGLESHOT: one pair per shot, shot ``n`` being the
    point LabOne Q acquired in shot ``n`` (the emulator numbers them in Q).

    Pattern: qibolab tests/test_result_shapes.py (shape per averaging mode);
    emulator placeholder values (laboneq 26.4.0
    src/python/laboneq/controller/devices/zi_emulator.py:971) to check the routing.
    """
    sequence = mz(platform)
    [(_, ro)] = sequence.acquisitions

    cap = effects.play(platform, [sequence], averaging_mode=averaging_mode, nshots=5)

    assert cap.averaging_mode == labone_q
    assert cap.results[ro.id].shape == shape
    [data] = cap.data("q0/acquisition")
    np.testing.assert_array_equal(cap.results[ro.id], iq(data))


# shots and relaxation


@pytest.mark.feature("F44")
@kills("nshots_ignored")
def test_nshots_is_acquire_loop_count(platform):
    """``nshots`` -> the acquisition loop runs that many times, one result per shot.

    Pattern: zhinst/laboneq-applications v26.4.0
    tests/experiments/test_compiled_rabi.py (count of the acquisition loop checked on
    the experiment handed to LabOne Q); qibolab tests/test_result_shapes.py.
    """
    sequence = mz(platform)
    [(_, ro)] = sequence.acquisitions

    cap = effects.play(
        platform, [sequence], nshots=7, averaging_mode=AveragingMode.SINGLESHOT
    )

    assert cap.nshots == 7
    assert cap.results[ro.id].shape == (7, 2)


@pytest.mark.feature("F44")
@kills("relaxation_zero")
def test_relaxation_time_after_each_readout(platform):
    """``relaxation_time`` 5000 ns -> every acquisition is followed by a 5 us wait.

    Checked on both acquisition channels of a multiplexed readout, within one pass of
    the acquisition loop, so in every shot.

    Pattern: zhinst/laboneq-applications v26.4.0
    tests/experiments/test_compiled_rabi.py (distance between shots); QM audit of
    qibolab (UriG-qm/qibolab qm-qa/05, relaxation time reaching the program).
    """
    cap = effects.play(platform, [mz(platform, "q0", "q1")], relaxation_time=5000)

    for q in ("q0", "q1"):
        assert cap.waits_after_acquires(f"{q}/acquisition") == [pytest.approx(5000)]
        assert all(op.real_time for op in cap.acquires(f"{q}/acquisition"))


@pytest.mark.feature("F29")
@kills("relaxation_zero")
def test_relaxation_time_after_each_sequence(platform):
    """Two sequences in one call -> the relaxation time follows each of them.

    Pattern: zhinst/laboneq-applications v26.4.0
    tests/experiments/test_compiled_rabi.py (distance between shots, per point);
    qibolab ``Platform.execute`` unrolling, which puts the relaxation time between
    sequences.
    """
    sequences = [mz(platform, "q0", "q1"), mz(platform, "q0", "q1")]

    cap = effects.play(platform, sequences, relaxation_time=3000)

    for q in ("q0", "q1"):
        waits = cap.waits_after_acquires(f"{q}/acquisition")
        assert waits == [pytest.approx(3000), pytest.approx(3000)]


# acquisition window


@pytest.mark.feature("F42")
@kills("acquisition_delay_dropped")
def test_acquisition_delay_follows_config(platform):
    """``AcquisitionConfig.delay`` -> the acquisition starts that much after the probe.

    Two calls on one connection, with the delay changed by an update in between, as
    time_of_flight_readout does.

    Pattern: qibocal time_of_flight_readout
    (signal_experiments/time_of_flight_readout.py:106) updating the delay; QM audit of
    qibolab (UriG-qm/qibolab qm-qa/*, time of flight reaching the program).
    """
    for delay in (48, 120):
        configs = updated(platform, q0__acquisition={"delay": delay})

        cap = effects.play(platform, [mz(platform)], configs=configs)

        assert acquisition_delay(cap, "q0") == pytest.approx(delay)


@pytest.mark.feature("F22")
@kills("integration_length_from_probe", "samples_q_is_i")
def test_integration_length_is_acquisition_duration(platform):
    """``Readout.acquisition.duration`` 800 ns -> LabOne Q integrates 800 ns, with
    unit weights, while the probe plays its own 1024 ns.

    Pattern: qibolab tests/test_result_shapes.py RAW/INTEGRATION split (samples from
    the acquisition duration); old qibolab ZI driver tests v0.1.10
    tests/test_instruments_zhinst.py (acquire length checked on the built Experiment).
    """
    sequence = PulseSequence([("q0/acquisition", readout(acquisition=800, probe=1024))])

    cap = effects.play(platform, [sequence])

    [acquire] = cap.acquires("q0/acquisition")
    [probe] = cap.plays("q0/probe")
    assert cap.duration(acquire) == pytest.approx(800)
    np.testing.assert_allclose(cap.samples(acquire), 1.0, atol=1e-12)
    assert cap.duration(probe) == pytest.approx(1024)


# results


@pytest.mark.feature("F23")
@kills("acquisition_handles_swapped", "results_q_is_i")
def test_multiplexed_results_reach_their_readout(platform):
    """Two multiplexed readouts -> each result holds the data acquired on its own
    qubit's acquisition channel, I and Q.

    Pattern: emulator placeholder values as result routing check (laboneq 26.4.0
    src/python/laboneq/controller/devices/zi_emulator.py:971, real part = 42 +
    integrator index); QM audit of qibolab (UriG-qm/qibolab qm-qa/*, one stream per
    readout).
    """
    sequence = mz(platform, "q0", "q1")
    readouts = {ch: ro for ch, ro in sequence.acquisitions}

    cap = effects.play(
        platform, [sequence], averaging_mode=AveragingMode.SINGLESHOT, nshots=4
    )

    data = {q: cap.data(f"{q}/acquisition") for q in ("q0", "q1")}
    if np.array_equal(data["q0"], data["q1"]):
        raise RuntimeError("Same placeholder data on both qubits: no routing check.")
    for q in ("q0", "q1"):
        [raw] = data[q]
        np.testing.assert_array_equal(
            cap.results[readouts[f"{q}/acquisition"].id], iq(raw)
        )


@pytest.mark.feature("F29")
@kills("acquisition_handles_swapped", "results_q_is_i")
def test_results_of_several_sequences(platform):
    """Two sequences of a multiplexed readout -> each of the four results holds the
    data of its own qubit, acquired in its own sequence.

    LabOne Q plays the acquisitions of one signal in order, so the ``n``-th acquisition
    on a qubit's channel belongs to the ``n``-th sequence.

    Pattern: emulator placeholder values as result routing check (laboneq 26.4.0
    src/python/laboneq/controller/devices/zi_emulator.py:971); qibocal
    single_shot_classification (classification/classification.py:245) sequences.
    """
    sequences = [mz(platform, "q0", "q1"), mz(platform, "q0", "q1")]

    cap = effects.play(
        platform, sequences, averaging_mode=AveragingMode.SINGLESHOT, nshots=3
    )

    for q in ("q0", "q1"):
        channel = f"{q}/acquisition"
        data = cap.data(channel)
        assert len(data) == len(sequences)
        for sequence, raw in zip(sequences, data):
            [ro] = [ro for ch, ro in sequence.acquisitions if ch == channel]
            np.testing.assert_array_equal(cap.results[ro.id], iq(raw))


@pytest.mark.parametrize(
    "averaging_mode",
    [
        pytest.param(AveragingMode.CYCLIC, id="F14", marks=pytest.mark.feature("F14")),
        pytest.param(
            AveragingMode.SINGLESHOT, id="F46", marks=pytest.mark.feature("F46")
        ),
    ],
)
@kills(
    "results_axes_reshaped",
    "sweep_grid_reversed",
    "frequency_sweep_offset",
    "results_q_is_i",
)
def test_result_axes_follow_sweepers(platform, averaging_mode):
    """Two sweepers -> result axes (shots,) drive frequency, probe frequency, and
    point ``[n, i, j]`` is the one LabOne Q measured in shot ``n`` at
    ``drive.values[i]`` and ``probe.values[j]``.

    The drive sweep runs in real time and the probe sweep in near time, so LabOne Q
    orders its axes differently from qibolab: the driver must move them. The two
    sweeps have different lengths. The sweep values of each LabOne Q axis come from
    ``Results.get_axis``.

    Pattern: qibolab tests/test_result_shapes.py (axis per sweeper, shots first);
    emulator placeholder values (laboneq 26.4.0
    src/python/laboneq/controller/devices/zi_emulator.py:971, imaginary part = point
    index in a real-time run).
    """
    drive = frequency_sweeper(platform, "q0/drive")
    f0 = platform.config("q0/probe").frequency
    probe = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 1e6, f0 + 2e6, 1e6),
        channels=["q0/probe"],
    )
    sequence = PulseSequence(
        [("q0/drive", square()), ("q0/acquisition", Delay(duration=40))]
    )
    sequence += mz(platform)
    [(_, ro)] = sequence.acquisitions
    shots = averaging_mode is AveragingMode.SINGLESHOT

    cap = effects.play(
        platform,
        [sequence],
        [[drive], [probe]],
        averaging_mode=averaging_mode,
        nshots=2,
    )

    axes = [
        ("q0/drive", "frequency", drive.values),
        ("q0/probe", "frequency", probe.values),
    ]
    [handle] = cap.handles("q0/acquisition")
    expected = expected_result(cap, handle, ["shots", *axes] if shots else axes)
    real_time = [0, 1] if shots else [0]
    if any(np.ptp(expected.imag, axis=k).max() == 0 for k in real_time):
        raise RuntimeError("Placeholder data do not vary along a real-time axis.")
    result = cap.results[ro.id]
    assert result.shape == (*expected.shape, 2)
    np.testing.assert_array_equal(result, iq(expected))
