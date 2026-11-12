"""One test per feature of :data:`features.FEATURES`, through ``ZiController.play``.

Each test is the smallest call shaped like the qibocal protocols that need the
feature. It checks only the qibolab contract, never values (emulation returns
placeholders): no exception, one result per acquisition, shape from
``ExecutionParameters.results_shape``, real and finite numbers.
"""

import numpy as np
import pytest

from qibolab._core.components import ExponentialFilter, FiniteImpulseResponseFilter
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.pulses import (
    Acquisition,
    Custom,
    Delay,
    Drag,
    Exponential,
    Gaussian,
    GaussianSquare,
    Pulse,
    Rectangular,
    Snz,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from .conftest import NSHOTS, RELAXATION, broken, refused

NOT_IMPLEMENTED = {
    "F05": "Pulse on channel 'q0/flux' is not supported.",
    "F06": "Sweeping duration is not supported.",
    "F07": "Sweeping duration is not supported.",
    "F08": "Sweeping duration is not supported.",
    "F09": "Sweeping duration_interpolated is not supported.",
    "F10": "Sweeping relative_phase is not supported.",
    "F11": "Sweeping phase is not supported.",
    "F12": "Sweeping offset is not supported.",
    "F17": "AcquisitionType.DISCRIMINATION is not supported.",
    "F18": "AcquisitionType.RAW is not supported.",
    "F21": "AveragingMode.SEQUENTIAL is not supported.",
    "F25": "Acquisition on channel 'q0/acquisition' is not supported.",
    "F27": "VirtualZ on channel 'q0/drive' is not supported.",
    "F28": "Pulse on channel 'q0/flux' is not supported.",
    "F39": "Filters on 'q0/flux' are not supported.",
    "F41": "Readout on channel 'q0/acquisition' is not supported.",
    "F45": "Fast reset is not supported.",
}
"""Features the driver refuses, with its exact ``NotImplementedError`` message.

Found by running. A refusal with another message is a failure, not an xfail.
"""

BROKEN = {
    "F47": "There are multiple 'play' operations in section",
}
"""Features the driver accepts but LabOne Q rejects when compiling, found by running."""


def marks(id):
    marks = [pytest.mark.feature(id)]
    if id in NOT_IMPLEMENTED:
        marks.append(refused(NOT_IMPLEMENTED[id]))
    if id in BROKEN:
        marks.append(broken(BROKEN[id]))
    return marks


def feature(id):
    def decorate(test):
        for mark in marks(id):
            test = mark(test)
        return test

    return decorate


def case(id, *values):
    return pytest.param(*values, id=id, marks=marks(id))


def play(platform, sequences, sweepers=(), configs=None, samples=-1, **options):
    """Call ``ZiController.play`` and check the qibolab contract on the results."""
    sweepers = list(sweepers)
    options = ExecutionParameters(
        **{
            "nshots": NSHOTS,
            "relaxation_time": RELAXATION,
            "acquisition_type": AcquisitionType.INTEGRATION,
            "averaging_mode": AveragingMode.CYCLIC,
        }
        | options
    )
    configs = configs if configs is not None else platform.parameters.configs.copy()
    controller = platform.instruments["zi"]

    results = controller.play(configs, sequences, options, sweepers)

    ids = {acq.id for seq in sequences for _, acq in seq.acquisitions}
    assert set(results) == ids
    for result in results.values():
        assert result.shape == options.results_shape(sweepers, samples)
        assert np.isrealobj(result)
        assert np.isfinite(result).all()
    return results


def mz(platform, *qubits):
    """Readout of the given qubits, from the MZ natives."""
    sequence = PulseSequence()
    for q in qubits or ("q0",):
        sequence += platform.natives.single_qubit[q].MZ()
    return sequence


def rx(platform, q="q0"):
    [(_, pulse)] = platform.natives.single_qubit[q].RX()
    return pulse


def then_readout(platform, channel, *pulses, q="q0"):
    """Pulses on one channel, a delay on the readout line, then MZ (rabi/utils.py:324)."""
    sequence = PulseSequence([(channel, pulse) for pulse in pulses])
    delay = Delay(duration=sum(pulse.duration for pulse in pulses))
    sequence.append((platform.qubits[q].acquisition, delay))
    sequence += mz(platform, q)
    return sequence, delay


def square(duration=40, amplitude=0.1):
    return Pulse(duration=duration, amplitude=amplitude, envelope=Rectangular())


def readout_of(sequence, n=0):
    return sequence.acquisitions[n][1]


def frequency_sweeper(platform, channel):
    f0 = platform.config(channel).frequency
    return Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 2e6, f0 + 2e6, 1e6),
        channels=[channel],
    )


def updated(platform, **updates):
    """Configs with some fields changed, as ``ExecutionParameters.updates`` does."""
    configs = platform.parameters.configs.copy()
    for name, update in updates.items():
        name = name.replace("__", "/")
        configs[name] = configs[name].model_copy(update=update)
    return configs


# sweep


@pytest.mark.parametrize("kind", [case("F01", "probe"), case("F02", "drive")])
def test_frequency_sweep(platform, kind):
    if kind == "probe":
        sequence = mz(platform)
    else:
        sequence, _ = then_readout(platform, "q0/drive", square())
    sweeper = frequency_sweeper(platform, f"q0/{kind}")
    play(platform, [sequence], [[sweeper]])


@feature("F03")
def test_amplitude_sweep_probe(platform):
    sequence = mz(platform)
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[readout_of(sequence)],
    )
    play(platform, [sequence], [[sweeper]])


@feature("F04")
def test_amplitude_sweep_drive(platform):
    pulse = rx(platform)
    sequence, _ = then_readout(platform, "q0/drive", pulse)
    sweeper = Sweeper(
        parameter=Parameter.amplitude, range=(0.1, 0.9, 0.2), pulses=[pulse]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F05")
def test_amplitude_sweep_flux(platform):
    pulse = square(amplitude=0.2)
    sequence, _ = then_readout(platform, "q0/flux", pulse)
    sweeper = Sweeper(
        parameter=Parameter.amplitude, range=(0.1, 0.5, 0.1), pulses=[pulse]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F06")
def test_duration_sweep_drive(platform):
    pulse = rx(platform)
    sequence, delay = then_readout(platform, "q0/drive", pulse)
    sweeper = Sweeper(
        parameter=Parameter.duration, range=(16, 96, 16), pulses=[pulse, delay]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F07")
def test_duration_sweep_delay(platform):
    pulse = rx(platform)
    sequence, _ = then_readout(platform, "q0/drive", pulse)
    wait = Delay(duration=16)
    sequence.insert(-1, ("q0/acquisition", wait))
    sweeper = Sweeper(
        parameter=Parameter.duration, range=(16, 1616, 400), pulses=[wait]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F08")
def test_duration_sweep_flux(platform):
    pulse = square(amplitude=0.2)
    sequence, delay = then_readout(platform, "q0/flux", pulse)
    sweeper = Sweeper(
        parameter=Parameter.duration, range=(16, 96, 16), pulses=[pulse, delay]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F09")
def test_duration_interpolated_sweep(platform):
    pulse = rx(platform)
    sequence = PulseSequence([("q0/drive", pulse)])
    sequence.align(["q0/drive", "q0/acquisition"])
    sequence += mz(platform)
    sweeper = Sweeper(
        parameter=Parameter.duration_interpolated, range=(16, 96, 16), pulses=[pulse]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F10")
def test_relative_phase_sweep(platform):
    pulse = rx(platform)
    sequence, _ = then_readout(platform, "q0/drive", pulse)
    sweeper = Sweeper(
        parameter=Parameter.relative_phase,
        values=np.linspace(0, np.pi, 5),
        pulses=[pulse],
    )
    play(platform, [sequence], [[sweeper]])


@feature("F11")
def test_virtual_z_phase_sweep(platform):
    vz = VirtualZ(phase=0)
    sequence, _ = then_readout(platform, "q0/drive", rx(platform), vz, rx(platform))
    sweeper = Sweeper(
        parameter=Parameter.phase, values=np.linspace(0, np.pi, 5), pulses=[vz]
    )
    play(platform, [sequence], [[sweeper]])


@feature("F12")
def test_offset_sweep(platform):
    offset0 = platform.config("q0/flux").offset
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=offset0 + np.linspace(-0.05, 0.05, 5),
        channels=["q0/flux"],
    )
    play(platform, [mz(platform)], [[sweeper]])


@feature("F13")
def test_parallel_sweepers(platform):
    sequence = mz(platform, "q0", "q1")
    sweepers = [frequency_sweeper(platform, f"{q}/probe") for q in ("q0", "q1")]
    play(platform, [sequence], [sweepers])


@feature("F14")
def test_nested_sweepers(platform):
    sequence = mz(platform)
    amplitude = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[readout_of(sequence)],
    )
    frequency = frequency_sweeper(platform, "q0/probe")
    play(platform, [sequence], [[amplitude], [frequency]])


@feature("F15")
def test_one_sweeper_several_qubits(platform):
    sequence = mz(platform, "q0", "q1")
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        range=(0.1, 1.0, 0.3),
        pulses=[ro for _, ro in sequence.acquisitions],
    )
    play(platform, [sequence], [[sweeper]])


# acquisition


@pytest.mark.parametrize(
    "acquisition_type",
    [
        case("F16", AcquisitionType.INTEGRATION),
        case("F17", AcquisitionType.DISCRIMINATION),
        case("F18", AcquisitionType.RAW),
    ],
)
def test_acquisition_type(platform, acquisition_type):
    sequence = mz(platform)
    configs = updated(platform, q0__acquisition={"threshold": 0.0, "iq_angle": 0.0})
    samples = int(readout_of(sequence).duration * platform.sampling_rate)
    play(
        platform,
        [sequence],
        configs=configs,
        samples=samples,
        acquisition_type=acquisition_type,
    )


# averaging


@pytest.mark.parametrize(
    "averaging_mode",
    [
        case("F19", AveragingMode.CYCLIC),
        case("F20", AveragingMode.SINGLESHOT),
        case("F21", AveragingMode.SEQUENTIAL),
    ],
)
def test_averaging_mode(platform, averaging_mode):
    play(platform, [mz(platform)], averaging_mode=averaging_mode)


@feature("F46")
def test_single_shots_with_sweeper(platform):
    sweeper = frequency_sweeper(platform, "q0/probe")
    play(platform, [mz(platform)], [[sweeper]], averaging_mode=AveragingMode.SINGLESHOT)


# sequence


@feature("F22")
def test_readout(platform):
    play(platform, [mz(platform)])


@feature("F23")
def test_multiplexed_readout(platform):
    play(platform, [mz(platform, "q0", "q1")])


@feature("F24")
def test_drive_then_readout(platform):
    sequence, _ = then_readout(platform, "q0/drive", square())
    play(platform, [sequence])


@feature("F25")
def test_probe_and_acquisition(platform):
    sequence = PulseSequence(
        [
            ("q0/probe", square(duration=1024)),
            ("q0/acquisition", Acquisition(duration=1024)),
        ]
    )
    play(platform, [sequence])


@feature("F26")
def test_align(platform):
    sequence = PulseSequence([("q0/drive", rx(platform))])
    sequence.align(["q0/drive", "q0/acquisition"])
    sequence += mz(platform)
    play(platform, [sequence])


@feature("F27")
def test_virtual_z(platform):
    vz = VirtualZ(phase=np.pi / 2)
    sequence, _ = then_readout(platform, "q0/drive", rx(platform), vz, rx(platform))
    play(platform, [sequence])


@feature("F28")
def test_flux_pulse(platform):
    sequence, _ = then_readout(platform, "q0/flux", square(amplitude=0.2))
    play(platform, [sequence])


@feature("F47")
def test_two_readouts_of_one_qubit(platform):
    """MZ, a delay, MZ again (readout/readout_characterization.py:99-101)."""
    sequence = mz(platform)
    sequence.append(("q0/acquisition", Delay(duration=100)))
    sequence += mz(platform)
    play(platform, [sequence], averaging_mode=AveragingMode.SINGLESHOT)


def ground_and_excited(platform):
    """The two sequences of single_shot_classification and dispersive_shift."""
    natives = platform.natives.single_qubit
    excite = natives["q0"].RX() + natives["q1"].RX()
    return [mz(platform, "q0", "q1"), excite | mz(platform, "q0", "q1")]


@feature("F29")
def test_several_sequences(platform):
    play(
        platform,
        ground_and_excited(platform),
        averaging_mode=AveragingMode.SINGLESHOT,
    )


@feature("F30")
def test_several_sequences_with_sweeper(platform):
    sweepers = [frequency_sweeper(platform, f"{q}/probe") for q in ("q0", "q1")]
    play(platform, ground_and_excited(platform), [sweepers])


# envelope


def custom(samples):
    return Custom(i_=np.ones(samples), q_=np.zeros(samples))


@pytest.mark.parametrize(
    "envelope",
    [
        case("F31", lambda _: Rectangular()),
        case("F32", lambda _: Gaussian(rel_sigma=0.2)),
        case("F33", lambda _: Drag(rel_sigma=0.2, beta=0.1)),
        case("F34", lambda _: GaussianSquare(risefall=10, sigma=4)),
        case("F35", lambda _: Exponential(tau=0.2, upsilon=0.5)),
        case("F36", lambda _: Snz(t_idling=10)),
        case("F37", custom),
    ],
)
def test_envelope(platform, envelope):
    duration = 40
    samples = int(duration * platform.sampling_rate)
    pulse = Pulse(duration=duration, amplitude=0.1, envelope=envelope(samples))
    sequence, _ = then_readout(platform, "q0/drive", pulse)
    play(platform, [sequence])


# config


@feature("F38")
def test_flux_offset(platform):
    configs = updated(platform, q0__flux={"offset": 0.3})
    play(platform, [mz(platform)], configs=configs)


@feature("F39")
def test_flux_filters(platform):
    filters = [
        ExponentialFilter(amplitude=-0.1, tau=40),
        FiniteImpulseResponseFilter(coefficients=[1.0, -0.2, 0.1]),
    ]
    configs = updated(platform, q0__flux={"filters": filters})
    play(platform, [mz(platform)], configs=configs)


@feature("F40")
def test_lo_update(platform):
    lo = platform.instruments["zi"].channels["q0/drive"].lo
    f0 = platform.config("q0/drive").frequency
    configs = updated(platform, q0__drive={"frequency": f0 + 100e6})
    configs[lo] = configs[lo].model_copy(update={"frequency": f0 + 100e6})
    sequence, _ = then_readout(platform, "q0/drive", square())
    play(platform, [sequence], configs=configs)


@feature("F41")
def test_kernel(platform):
    sequence = mz(platform)
    samples = int(readout_of(sequence).duration * platform.sampling_rate)
    kernel = np.ones(samples, dtype=complex)
    configs = updated(platform, q0__acquisition={"kernel": kernel})
    play(platform, [sequence], configs=configs)


@feature("F42")
def test_acquisition_delay(platform):
    configs = updated(platform, q0__acquisition={"delay": 50})
    play(platform, [mz(platform)], configs=configs)


@feature("F43")
def test_new_configs_each_call(platform):
    f0 = platform.config("q0/probe").frequency
    for frequency in (f0, f0 + 1e6):
        configs = updated(platform, q0__probe={"frequency": frequency})
        play(platform, [mz(platform)], configs=configs)


# execution


@feature("F44")
def test_nshots_relaxation(platform):
    """Single shots, so that ``nshots`` shows in the result shape."""
    play(
        platform,
        [mz(platform)],
        nshots=3,
        relaxation_time=20_000,
        averaging_mode=AveragingMode.SINGLESHOT,
    )


@feature("F45")
def test_fast_reset(platform):
    play(platform, [mz(platform)], fast_reset=True)
