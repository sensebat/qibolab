"""Elisa's checkpoints CP1..CP5 (D27, D28): the exact qibocal acquisition calls.

Each test rebuilds the sequence and sweepers of the qibocal ``_acquisition`` function
(paths relative to ``qibocal/src/qibocal/protocols``) and calls ``platform.execute``
with the same arguments. Parameters are small numbers, to keep emulation fast. The
results are read the way the protocol reads them.
"""

import numpy as np
import pytest

from qibolab._core.execution_parameters import AcquisitionType, AveragingMode
from qibolab._core.pulses import Acquisition, Delay, Pulse, Readout, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import NSHOTS, QUBITS, RELAXATION, broken, refused

REFUSED: dict[str, str] = {}
"""Checkpoints the driver refuses, with its exact ``NotImplementedError`` message."""

LO_OFF_GRID = broken(
    # laboneq 26.4; it names q0/drive or q1/drive (they share the LO), not always the same
    "to 6.201 GHz. Only integer multiples of 200 MHz are accepted.",
    # laboneq 26.10
    "(SHFSG) Local oscillator frequency 6201000000 Hz is not multiple of 100000000 Hz.",
    reason="LO off the 200 MHz grid (known gap)",
)
"""The wide CP3 batches move the drive LO to f0 + offset, which LabOne Q refuses.

q0 and q1 share one drive LO; the update of q1 (f0 = 6.301 GHz) comes last and puts
it at 6.201 GHz.
"""


def checkpoint(id):
    def decorate(test):
        test = pytest.mark.checkpoint(id)(test)
        if id in REFUSED:
            test = refused(REFUSED[id])(test)
        return test

    return decorate


def check(results, readouts, shape):
    """What the protocol reads: ``results[ro.id]``, then I and Q on the last axis."""
    for ro in readouts:
        result = results[ro.id]
        assert result.shape == shape
        assert np.isfinite(result).all()
        i, q = result[..., 0], result[..., 1]
        assert i.shape == q.shape


def frequency_range(center, width=4e6, step=1e6):
    return (center - width / 2, center + width / 2, step)


@checkpoint("CP1")
def test_resonator_spectroscopy(platform):
    """resonator_spectroscopies/resonator_spectroscopy.py:188-228."""
    sequence = PulseSequence()
    ro_pulses = {}
    for q in QUBITS:
        readout = Readout(
            acquisition=Acquisition(duration=1000),
            probe=Pulse(duration=1000, amplitude=0.1, envelope=Rectangular()),
        )
        ro_pulses[q] = readout
        sequence.append((platform.qubits[q].acquisition, readout))
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            range=frequency_range(platform.config(platform.qubits[q].probe).frequency),
            channels=[platform.qubits[q].probe],
        )
        for q in QUBITS
    ]

    results = platform.execute(
        [sequence],
        [sweepers],
        nshots=NSHOTS,
        relaxation_time=RELAXATION,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    check(results, ro_pulses.values(), (len(sweepers[0]), 2))


@checkpoint("CP2")
def test_resonator_punchout(platform):
    """resonator_spectroscopies/resonator_punchout.py:99-142."""
    delta_frequency_range = np.arange(-2e6, 2e6, 1e6)
    ro_pulses = {}
    freq_sweepers = {}
    sequence = PulseSequence()
    for qubit in QUBITS:
        natives = platform.natives.single_qubit[qubit]
        ro_channel, ro_pulse = natives.MZ()[0]
        ro_pulses[qubit] = ro_pulse.model_copy(
            update={"probe": ro_pulse.probe.model_copy(update={"amplitude": 1.0})}
        )
        sequence.append((ro_channel, ro_pulses[qubit]))
        probe = platform.qubits[qubit].probe
        f0 = platform.config(probe).frequency
        freq_sweepers[qubit] = Sweeper(
            parameter=Parameter.frequency,
            values=f0 + delta_frequency_range,
            channels=[probe],
        )
    amp_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[ro_pulses[qubit] for qubit in QUBITS],
    )

    results = platform.execute(
        [sequence],
        [[amp_sweeper], [freq_sweepers[q] for q in QUBITS]],
        nshots=NSHOTS,
        relaxation_time=RELAXATION,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    shape = (len(amp_sweeper), len(delta_frequency_range), 2)
    check(results, ro_pulses.values(), shape)


def calculate_batches(freq_width, max_if_bandwidth=300e6):
    """Copy of qubit_spectroscopy._calculate_batches (qubit_spectroscopy.py:113-124)."""
    batch_starts = np.arange(-freq_width / 2, freq_width / 2, 2 * max_if_bandwidth)
    batch_ends = np.append(batch_starts[1:], freq_width / 2)
    batch_limits = np.stack((batch_starts, batch_ends))
    lo_offsets = batch_limits.sum(axis=0) / 2 if len(batch_starts) > 1 else [0]
    return np.vstack((batch_limits, lo_offsets)).T


@checkpoint("CP3")
@pytest.mark.parametrize(
    "width,step",
    [
        pytest.param(4e6, 1e6, id="narrow"),
        pytest.param(800e6, 100e6, id="wide", marks=LO_OFF_GRID),
    ],
)
def test_qubit_spectroscopy(platform, width, step):
    """qubit_spectroscopies/qubit_spectroscopy.py:137-226.

    The wide span is cut in two batches, each with the drive LO moved by ``updates``.
    """
    drive_channels = {}
    lo_channels = {}
    f0 = {}
    for qubit in QUBITS:
        drive_channel = platform.qubits[qubit].drive
        drive_channels[qubit] = drive_channel
        f0[qubit] = platform.config(drive_channel).frequency
        lo_channels[qubit] = platform.channels[drive_channel].lo

    for start, end, lo_offset in calculate_batches(width):
        sequence = PulseSequence()
        freq_ranges = {}
        ro_pulses = {}
        sweepers = []
        for qubit in QUBITS:
            natives = platform.natives.single_qubit[qubit]
            qd_channel = drive_channels[qubit]
            [(ro_channel, ro_pulse)] = natives.MZ()
            qd_pulse = Pulse(amplitude=0.1, duration=2000, envelope=Rectangular())
            ro_pulses[qubit] = ro_pulse
            sequence.append((qd_channel, qd_pulse))
            sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence.append((ro_channel, ro_pulse))
            batch_range = (f0[qubit] + start, f0[qubit] + end, step)
            freq_ranges[qubit] = batch_range
            sweepers.append(
                Sweeper(
                    parameter=Parameter.frequency,
                    range=batch_range,
                    channels=[qd_channel],
                )
            )
        batch_updates = []
        for qubit in QUBITS:
            update_dict = {}
            update_dict[drive_channels[qubit]] = {"frequency": f0[qubit] + lo_offset}
            if lo_offset != 0 and lo_channels[qubit] is not None:
                update_dict[lo_channels[qubit]] = {"frequency": f0[qubit] + lo_offset}
            batch_updates.append(update_dict)

        results = platform.execute(
            [sequence],
            [sweepers],
            updates=batch_updates,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
            nshots=NSHOTS,
            relaxation_time=RELAXATION,
        )

        check(results, ro_pulses.values(), (len(sweepers[0]), 2))


@checkpoint("CP4")
@pytest.mark.parametrize("rx90", [False, True], ids=["RX", "RX90"])
def test_rabi_amplitude_signal(platform, rx90):
    """rabi/amplitude_signal.py:88-111 with rabi/utils.py:298-329.

    With ``rx90`` the same pulse is played twice and the sweeper changes both.
    """
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for q in QUBITS:
        natives = platform.natives.single_qubit[q]
        qd_channel, qd_pulse = natives.RX90()[0] if rx90 else natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]
        qd_pulses[q] = qd_pulse
        ro_pulses[q] = ro_pulse
        if rx90:
            sequence.append((qd_channel, qd_pulses[q]))
        sequence.append((qd_channel, qd_pulses[q]))
        sequence.append((ro_channel, Delay(duration=qd_pulse.duration)))
        sequence.append((ro_channel, ro_pulse))
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.0, 1.0, 0.1),
        pulses=[qd_pulses[qubit] for qubit in QUBITS],
    )

    results = platform.execute(
        [sequence],
        [[sweeper]],
        updates=[
            {
                platform.qubits[q].probe: {
                    "frequency": platform.config(platform.qubits[q].probe).frequency
                }
            }
            for q in QUBITS
        ],
        nshots=NSHOTS,
        relaxation_time=RELAXATION,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    check(results, ro_pulses.values(), (len(sweeper), 2))


@checkpoint("CP5")
def test_single_shot_classification_and_dispersive_shift(platform):
    """classification/classification.py:210-245, dispersive_shift/dispersive_shift.py:114-138."""
    native = platform.natives.single_qubit
    sequences, all_ro_pulses = [], []
    for state in [0, 1]:
        ro_pulses = {}
        sequence = PulseSequence()
        for q in QUBITS:
            ro_sequence = native[q].MZ()
            ro_pulses[q] = ro_sequence[0][1]
            sequence += ro_sequence
        if state == 1:
            rx_sequence = PulseSequence()
            for q in QUBITS:
                rx_sequence += native[q].RX()
            sequence = rx_sequence | sequence
        sequences.append(sequence)
        all_ro_pulses.append(ro_pulses)

    results = platform.execute(
        sequences,
        nshots=NSHOTS,
        relaxation_time=RELAXATION,
        acquisition_type=AcquisitionType.INTEGRATION,
    )

    for ro_pulses in all_ro_pulses:
        check(results, ro_pulses.values(), (NSHOTS, 2))

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    for qubit in QUBITS:
        natives = platform.natives.single_qubit[qubit]
        sequence_0 += natives.MZ()
        sequence_1 += natives.RX() | natives.MZ()
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            range=frequency_range(platform.config(platform.qubits[q].probe).frequency),
            channels=[platform.qubits[q].probe],
        )
        for q in QUBITS
    ]

    results = platform.execute(
        [sequence_0, sequence_1],
        [sweepers],
        nshots=NSHOTS,
        relaxation_time=RELAXATION,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in QUBITS:
        for sequence in [sequence_0, sequence_1]:
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            check(results, [ro_pulse], (len(sweepers[0]), 2))
