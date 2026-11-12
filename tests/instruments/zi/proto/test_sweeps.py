"""Sweeps added on ``orkun/proto-sweeps``, checked in LabOne Q emulation.

Each test checks what reaches the LabOne Q experiment (the operation that carries the
sweep parameter) and that the emulated run gives the qibolab result shape. Emulation
returns placeholder data, so nothing here says what the instruments output.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.dsl.experiment import Section
from laboneq.simple import Calibration, Experiment, ExperimentSignal, SweepParameter

from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.zi import pulse as zipulse
from qibolab._core.instruments.zi.controller import SAMPLING_RATE, calibration
from qibolab._core.instruments.zi.sweep import split
from qibolab._core.pulses import Delay, Gaussian, Pulse, Rectangular, VirtualZ
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from ..conftest import INSTRUMENTS
from ..test_calibration import channels, configs  # noqa: F401
from ..test_play import connected, drives, frequencies, options, readouts  # noqa: F401

AVERAGING = [AveragingMode.CYCLIC, AveragingMode.SINGLESHOT]


def operations(sequence, channels, configs, swept):  # noqa: F811
    """LabOne Q operations played for ``sequence``, by signal."""
    experiment = Experiment(signals=[ExperimentSignal(id) for id in channels])
    windows = zipulse.windows([sequence], channels)
    blocks = zipulse.layout(sequence, channels, SAMPLING_RATE, windows, swept)
    with experiment.acquire_loop_rt(count=1), experiment.section() as section:
        zipulse.play(experiment, blocks, channels, configs, SAMPLING_RATE, swept, {})
    ops = {}

    def collect(node):
        for child in node.children:
            if isinstance(child, Section):
                collect(child)
            else:
                ops.setdefault(child.signal, []).append(child)

    collect(section)
    return ops


def run(connected, configs, sequence, sweepers, averaging_mode):  # noqa: F811
    """Run in emulation and check the shape of every result."""
    opts = options(averaging_mode)
    results = connected.play(configs, [sequence], opts, sweepers)
    assert set(results) == {ro.id for _, ro in sequence.acquisitions}
    for result in results.values():
        assert result.shape == opts.results_shape(sweepers)
        assert not np.isnan(result).any()
    return results


def test_probe_amplitude_lookup(channels, configs):  # noqa: F811
    """G1: a sweeper given ``Readout.probe`` sweeps the probe amplitude."""
    sequence = readouts([0])
    readout = sequence[0][1]
    parameter = SweepParameter(values=np.array([0.1, 0.2]))
    swept = {(readout.probe.id, Parameter.amplitude): parameter}

    ops = operations(sequence, channels, configs, swept)

    assert ops["q0/probe"][0].amplitude is parameter


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_probe_amplitude(connected, configs, averaging_mode):  # noqa: F811
    """G1: punchout with the sweeper on the probe pulses, as qibocal may build it."""
    sequence = readouts([0, 1])
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        values=np.array([0.1, 0.5, 0.9]),
        pulses=[ro.probe for _, ro in sequence.acquisitions],
    )
    run(connected, configs, sequence, [[amplitude]], averaging_mode)


DURATIONS = np.array([40, 80, 120])


@pytest.mark.parametrize(
    "parameter", [Parameter.duration, Parameter.duration_interpolated]
)
def test_duration_lookup(channels, configs, parameter):  # noqa: F811
    """F06, F07, F09: the swept length of a pulse and of its delay, in s."""
    sequence = drives([0])
    (_, drive), (_, delay) = list(sequence)[:2]
    swept = {}
    duration = Sweeper(parameter=parameter, values=DURATIONS, pulses=[drive, delay])
    split([[duration]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    played = swept[drive.id, Parameter.duration]
    assert played.values == pytest.approx(DURATIONS * 1e-9)
    assert ops["q0/drive"][0].length is played
    # the delay on the acquisition channel also delays the probe
    for signal in ("q0/acquisition", "q0/probe"):
        assert ops[signal][0].time is swept[delay.id, Parameter.duration]


@pytest.mark.parametrize("averaging_mode", AVERAGING)
@pytest.mark.parametrize(
    "parameter", [Parameter.duration, Parameter.duration_interpolated]
)
def test_rabi_length(connected, configs, averaging_mode, parameter):  # noqa: F811
    """F06, F09: duration of the drive and of the delay before the readout."""
    sequence = drives([0, 1])
    duration = Sweeper(
        parameter=parameter,
        values=DURATIONS,
        pulses=[op for _, op in list(sequence)[:4]],
    )
    run(connected, configs, sequence, [[duration]], averaging_mode)


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_delay_duration(connected, configs, averaging_mode):  # noqa: F811
    """F07, X23: T1, the delay between a pi pulse and the readout is swept."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
    delay = Delay(duration=40)
    sequence = PulseSequence(
        [("q0/drive", drive), ("q0/acquisition", Delay(duration=40))]
    )
    sequence.append(("q0/acquisition", delay))
    sequence |= readouts([0])
    duration = Sweeper(parameter=Parameter.duration, values=DURATIONS, pulses=[delay])
    run(connected, configs, sequence, [[duration]], averaging_mode)


def test_fractional_duration_refused(connected, configs):  # noqa: F811
    """G6: swept durations must be whole samples, as fixed ones (orkun/proto-pulses).

    On orkun/proto-sweeps a fractional duration played with the functional envelope;
    the merge keeps the refusal of orkun/proto-pulses for fixed and swept ones.
    """
    sequence = drives([0])
    (_, drive), (_, delay) = list(sequence)[:2]
    duration = Sweeper(
        parameter=Parameter.duration, values=np.array([40, 41.3]), pulses=[drive, delay]
    )
    with pytest.raises(ValueError, match="whole number of samples"):
        connected.play(configs, [sequence], options(AveragingMode.CYCLIC), [[duration]])


def test_delay_duration_off_readout_grid_refused(connected, configs):  # noqa: F811
    """A swept delay before a readout changes by multiples of 8 ns, or is refused.

    Found in review: with 40, 44 and 60 ns LabOne Q moved the readout onto the 8 ns
    readout grid (4 ns late) without an error (output simulator).
    """
    delay = Delay(duration=40)
    sequence = PulseSequence(
        [("q0/drive", Pulse(duration=40, amplitude=0.5, envelope=Rectangular()))]
    )
    sequence.append(("q0/acquisition", Delay(duration=40)))
    sequence.append(("q0/acquisition", delay))
    sequence |= readouts([0])
    duration = Sweeper(
        parameter=Parameter.duration, values=np.array([40, 44, 60]), pulses=[delay]
    )
    with pytest.raises(ValueError, match="readout grid"):
        connected.play(configs, [sequence], options(AveragingMode.CYCLIC), [[duration]])


PHASES = np.linspace(0, 2 * np.pi, 5)


def test_relative_phase_lookup(channels, configs):  # noqa: F811
    """F10: the swept phase of the drive pulse, in rad, in real time."""
    sequence = drives([0])
    drive = sequence[0][1]
    swept = {}
    phase = Sweeper(parameter=Parameter.relative_phase, values=PHASES, pulses=[drive])
    _, real, _ = split([[phase]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    assert real == [[swept[drive.id, Parameter.relative_phase]]]
    assert ops["q0/drive"][0].phase is real[0][0]
    assert real[0][0].values == pytest.approx(PHASES)


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_relative_phase(connected, configs, averaging_mode):  # noqa: F811
    """F10: relative phase of the drive pulses of two qubits."""
    sequence = drives([0, 1])
    phase = Sweeper(
        parameter=Parameter.relative_phase,
        values=PHASES,
        pulses=[sequence[0][1], sequence[2][1]],
    )
    run(connected, configs, sequence, [[phase]], averaging_mode)


def ramsey_vz(qubits) -> tuple[PulseSequence, list[VirtualZ]]:
    """Two drive pulses with a virtual Z between them, then the readout."""
    sequence = PulseSequence()
    vzs = []
    for q in qubits:
        drive = Pulse(duration=40, amplitude=0.5, envelope=Gaussian(rel_sigma=0.2))
        vz = VirtualZ(phase=0.1)
        vzs.append(vz)
        sequence.extend(
            [
                (f"q{q}/drive", drive),
                (f"q{q}/drive", vz),
                (f"q{q}/drive", drive.new()),
                (f"q{q}/acquisition", Delay(duration=80)),
            ]
        )
    return sequence | readouts(qubits), vzs


def test_virtual_z_phase_lookup(channels, configs):  # noqa: F811
    """F11: the swept phase of a virtual Z increments the drive oscillator."""
    sequence, (vz,) = ramsey_vz([0])
    swept = {}
    phase = Sweeper(parameter=Parameter.phase, values=PHASES, pulses=[vz])
    _, real, _ = split([[phase]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    played = ops["q0/drive"]
    assert len(played) == 3
    assert played[1].pulse is None
    assert played[1].increment_oscillator_phase is real[0][0]
    assert real[0][0].values == pytest.approx(PHASES)
    # without a sweep, the fixed phase is added, with its sign
    ops = operations(sequence, channels, configs, {})
    assert ops["q0/drive"][1].increment_oscillator_phase == 0.1


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_virtual_z_phase(connected, configs, averaging_mode):  # noqa: F811
    """F11: phase of the virtual Z of two qubits."""
    sequence, vzs = ramsey_vz([0, 1])
    phase = Sweeper(parameter=Parameter.phase, values=PHASES, pulses=vzs)
    run(connected, configs, sequence, [[phase]], averaging_mode)


OFFSETS = np.array([-0.2, 0.0, 0.2])


def test_offset_split(channels, configs):  # noqa: F811
    """F12: the offset of a flux channel is swept in near time, absolute values."""
    ids = ["q0/flux", "q1/flux"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    offset = Sweeper(parameter=Parameter.offset, values=OFFSETS, channels=ids)

    near, real, axes = split([[offset]], channels, configs, calib, {})

    assert real == [] and axes == [0, None]
    assert [calib[id].voltage_offset for id in ids] == near[0]
    assert near[0][0].values == pytest.approx(OFFSETS)
    with pytest.raises(ValueError, match="q0/drive"):
        offset = Sweeper(
            parameter=Parameter.offset, values=OFFSETS, channels=["q0/drive"]
        )
        split([[offset]], channels, configs, calib, {})


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_offset(connected, configs, averaging_mode):  # noqa: F811
    """F12: flux offset of two qubits, as a flux spectroscopy, with the probe frequency."""
    sequence = readouts([0, 1])
    offset = Sweeper(
        parameter=Parameter.offset, values=OFFSETS, channels=["q0/flux", "q1/flux"]
    )
    probes = frequencies(configs, ["q0/probe", "q1/probe"], points=4)
    run(connected, configs, sequence, [[offset], probes], averaging_mode)


def flux_pulses(qubits) -> PulseSequence:
    """Flux pulse on each qubit, then a delayed readout."""
    sequence = PulseSequence()
    for q in qubits:
        flux = Pulse(duration=40, amplitude=0.3, envelope=Rectangular())
        sequence.extend(
            [(f"q{q}/flux", flux), (f"q{q}/acquisition", Delay(duration=40))]
        )
    return sequence | readouts(qubits)


def test_flux_duration_lookup(channels, configs):  # noqa: F811
    """F08: the swept length of a flux pulse, on the HDAWG signal."""
    sequence = flux_pulses([0])
    flux = sequence[0][1]
    swept = {}
    duration = Sweeper(parameter=Parameter.duration, values=DURATIONS, pulses=[flux])
    split([[duration]], channels, configs, Calibration(), swept)

    ops = operations(sequence, channels, configs, swept)

    assert ops["q0/flux"][0].length is swept[flux.id, Parameter.duration]
    assert ops["q0/flux"][0].amplitude == 0.3


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_flux_duration(connected, configs, averaging_mode):  # noqa: F811
    """F08: duration of the flux pulses of two qubits and of the readout delays."""
    sequence = flux_pulses([0, 1])
    duration = Sweeper(
        parameter=Parameter.duration,
        values=DURATIONS,
        pulses=[op for _, op in list(sequence)[:4]],
    )
    run(connected, configs, sequence, [[duration]], averaging_mode)


def test_mixed_split(channels, configs):  # noqa: F811
    """F13: a group with a real-time and a near-time sweep runs in near time."""
    ids = ["q0/drive", "q0/probe", "q0/acquisition"]
    calib = calibration(ids, channels, INSTRUMENTS, configs)
    parallel = frequencies(configs, ["q0/drive", "q0/probe"])

    near, real, axes = split([parallel], channels, configs, calib, {})

    assert real == [] and axes == [0, None]
    assert near[0] == [calib[id].oscillator.frequency for id in ids[:2]]


def mixed(configs, sequence):  # noqa: F811
    """Parallel groups, each with a real-time sweep next to a near-time one."""
    pulses = [op for _, op in sequence]
    drive, vz, delay = pulses[0], pulses[1], pulses[3]
    probe = sequence.acquisitions[0][1]
    n = 3
    return [
        frequencies(configs, ["q0/drive", "q0/probe"], points=n),
        [
            Sweeper(
                parameter=Parameter.duration,
                values=DURATIONS[:n],
                pulses=[pulses[2], delay],
            ),
            Sweeper(
                parameter=Parameter.amplitude,
                values=0.1 * np.arange(1, n + 1),
                pulses=[probe],
            ),
        ],
        [
            Sweeper(
                parameter=Parameter.relative_phase, values=PHASES[:n], pulses=[drive]
            ),
            Sweeper(parameter=Parameter.phase, values=PHASES[:n], pulses=[vz]),
            Sweeper(
                parameter=Parameter.offset, values=OFFSETS[:n], channels=["q0/flux"]
            ),
        ],
    ]


@pytest.mark.parametrize("averaging_mode", AVERAGING)
def test_mixed(connected, configs, averaging_mode):  # noqa: F811
    """F13, G2, X20: every real-time parameter also runs in a near-time group.

    Drive frequency, duration of a pulse and of a delay, relative phase and virtual Z
    phase, each next to a near-time sweep (probe frequency, probe amplitude, offset).
    """
    sequence, _ = ramsey_vz([0])
    run(connected, configs, sequence, mixed(configs, sequence), averaging_mode)


@pytest.mark.parametrize("id", ["q0/flux", "q0/acquisition"])
def test_frequency_not_iq(channels, configs, id):  # noqa: F811
    """X06: a frequency sweep on a channel without oscillator names the channel."""
    calib = calibration([id], channels, INSTRUMENTS, configs)
    frequency = Sweeper(
        parameter=Parameter.frequency, values=np.array([1.0, 2.0]), channels=[id]
    )
    with pytest.raises(ValueError, match=id):
        split([[frequency]], channels, configs, calib, {})


@pytest.mark.parametrize("parameter", [Parameter.duration, Parameter.relative_phase])
@pytest.mark.parametrize("probe", [False, True])
def test_probe_other_sweeps(channels, configs, parameter, probe):  # noqa: F811
    """Probe duration and phase sweeps are refused, not ignored.

    A real-time length sweep on a probe is refused by LabOne Q: an SHFQA generator
    holds one waveform.
    """
    sequence = readouts([0])
    readout = sequence[0][1]
    pulse = readout.probe if probe else readout
    swept = {}
    sweeper = Sweeper(parameter=parameter, values=np.array([1.0, 2.0]), pulses=[pulse])
    split([[sweeper]], channels, configs, Calibration(), swept)
    with pytest.raises(NotImplementedError, match="q0/acquisition"):
        operations(sequence, channels, configs, swept)
