"""Pulses added on orkun/proto-pulses, run in LabOne Q emulation.

Emulation compiles the experiment and returns placeholder data. Timing and samples
are checked on the compiled output with the LabOne Q output simulator.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.simulator.output_simulator import OutputSimulator

from qibolab._core.components import (
    DcChannel,
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.zi import ZiDcConfig
from qibolab._core.instruments.zi.controller import precompensation
from qibolab._core.pulses import (
    Custom,
    Delay,
    Exponential,
    Pulse,
    Readout,
    Rectangular,
    Snz,
    VirtualZ,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

from ..test_calibration import channels, configs  # noqa: F401

OPTIONS = ExecutionParameters(
    nshots=3,
    relaxation_time=1000,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.CYCLIC,
)

PROBE = Pulse(duration=1000, amplitude=0.1, envelope=Rectangular())

NS = 1e-9


@pytest.fixture
def connected(controller, channels):  # noqa: F811
    controller.channels = channels
    controller.connect()
    yield controller
    controller.disconnect()


@pytest.fixture
def compiled(connected):
    """Compiled experiment of each call to ``connected.play``."""
    out = []
    compile = connected.session.compile

    def spy(experiment, *args, **kwargs):
        out.append(compile(experiment, *args, **kwargs))
        return out[-1]

    connected.session.compile = spy
    return out


def wave(controller, compiled, id, length=2e-6):
    """Simulated output of channel ``id`` in the first ``length`` s, times in ns."""
    channel = controller.session.device_setup.logical_signal_by_uid(id)
    snippet = OutputSimulator(compiled).get_snippet(
        channel.physical_channel, start=0, output_length=length
    )
    return np.round(snippet.time / NS, 3), snippet.wave


def segments(time, wave) -> list[tuple[float, float]]:
    """Start and end in ns of each run of non-zero output."""
    on = np.concatenate([[False], np.abs(wave) > 1e-6, [False]])
    edges = np.flatnonzero(np.diff(on.astype(int)))
    step = time[1] - time[0]
    return [(time[a], time[b - 1] + step) for a, b in zip(edges[::2], edges[1::2])]


def readout(qubit: int = 0, probe: Pulse = PROBE) -> tuple[str, Readout]:
    return (f"q{qubit}/acquisition", Readout.from_probe(probe.new()))


def run(controller, configs, sequences, sweepers=()):  # noqa: F811
    """Play ``sequences``, checking one result of the expected shape per readout."""
    results = controller.play(configs, sequences, OPTIONS, list(sweepers))
    ids = {ro.id for seq in sequences for _, ro in seq.acquisitions}
    assert set(results) == ids
    for result in results.values():
        assert result.shape == OPTIONS.results_shape(list(sweepers))
    return results


@pytest.mark.parametrize(
    "op",
    [
        ("q0/drive", Pulse(duration=41.3, amplitude=0.5, envelope=Rectangular())),
        ("q0/acquisition", Delay(duration=10.25)),
        readout(probe=PROBE.model_copy(update={"duration": 999.9})),
    ],
)
def test_sub_sample_duration_refused(connected, configs, op):  # noqa: F811
    with pytest.raises(ValueError, match="whole number of samples"):
        connected.play(configs, [PulseSequence([op, readout()])], OPTIONS, [])


def test_half_ns_duration(connected, configs):  # noqa: F811
    """41.5 ns is 83 samples at 2 GSps."""
    drive = Pulse(duration=41.5, amplitude=0.5, envelope=Rectangular())
    run(connected, configs, [PulseSequence([("q0/drive", drive), readout()])])


@pytest.mark.parametrize(
    "envelope",
    [
        Rectangular(),
        Exponential(tau=0.2, upsilon=0.5),
        Snz(t_idling=40),
        Custom(i_=np.linspace(0, 1, 200), q_=np.zeros(200)),
    ],
)
def test_flux_pulse(connected, compiled, configs, envelope):  # noqa: F811
    """The flux pulse plays its real samples on the HDAWG, 40 ns after the start."""
    flux = Pulse(duration=100, amplitude=0.4, envelope=envelope)
    sequence = PulseSequence(
        [
            ("q0/flux", Delay(duration=40)),
            ("q0/flux", flux),
            ("q0/acquisition", Delay(duration=140)),
            readout(),
        ]
    )
    run(connected, configs, [sequence])
    time, out = wave(connected, compiled[-1], "q0/flux")
    start = np.searchsorted(time, 40)
    assert np.all(out[:start] == 0)
    assert out[start : start + 200] == pytest.approx(0.4 * envelope.i(200), abs=1e-6)
    assert np.all(out[start + 200 : start + 2000] == 0)


def test_flux_amplitude_sweep(connected, configs):  # noqa: F811
    flux = Pulse(duration=100, amplitude=0.4, envelope=Rectangular())
    sequence = PulseSequence(
        [("q0/flux", flux), ("q0/acquisition", Delay(duration=100)), readout()]
    )
    sweeper = Sweeper(
        parameter=Parameter.amplitude, values=np.array([0.1, 0.2, 0.3]), pulses=[flux]
    )
    run(connected, configs, [sequence], [[sweeper]])


def test_coupler_pulse(controller, channels, configs):  # noqa: F811
    """A coupler flux line is a DC channel on a free HDAWG port."""
    channels["c01/flux"] = DcChannel(device="hdawg", path="7")
    configs["c01/flux"] = ZiDcConfig(offset=0.0, power_range=5)
    controller.channels = channels
    controller.connect()
    try:
        flux = Pulse(duration=60, amplitude=0.3, envelope=Rectangular())
        sequence = PulseSequence(
            [("c01/flux", flux), ("q0/acquisition", Delay(duration=60)), readout()]
        )
        run(controller, configs, [sequence])
    finally:
        controller.disconnect()


def test_flux_phase_refused(connected, configs):  # noqa: F811
    flux = Pulse(
        duration=100, amplitude=0.4, envelope=Rectangular(), relative_phase=0.1
    )
    with pytest.raises(ValueError, match="no phase"):
        run(connected, configs, [PulseSequence([("q0/flux", flux), readout()])])


def drive_output(controller, configs, compiled, ops):  # noqa: F811
    """Drive output of ``q0`` for the operations ``ops`` on its drive channel."""
    duration = sum(op.duration for op in ops)
    sequence = PulseSequence(
        [("q0/drive", op) for op in ops]
        + [("q0/acquisition", Delay(duration=duration)), readout()]
    )
    run(controller, configs, [sequence])
    return wave(controller, compiled[-1], "q0/drive")


@pytest.mark.parametrize("phase", [np.pi / 2, -0.3])
def test_virtual_z(connected, compiled, configs, phase):  # noqa: F811
    """A VirtualZ adds its phase to the following pulses.

    It is checked against the relative phase of the pulse (X24, already supported):
    VirtualZ(phase) then a pulse plays as the pulse with ``relative_phase=phase``,
    which is the qibolab meaning in qm (``frame_rotation_2pi``) and qblox
    (``set_ph_delta``).
    """
    pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    shifted = pulse.model_copy(update={"relative_phase": phase})
    time, expected = drive_output(connected, configs, compiled, [pulse, shifted])
    _, rotated = drive_output(
        connected, configs, compiled, [pulse.new(), VirtualZ(phase=phase), pulse.new()]
    )
    assert rotated[:160] == pytest.approx(expected[:160], abs=1e-6)
    assert np.abs(rotated[80:160] - rotated[:80]).max() > 0.1
    assert segments(time, rotated)[0] == (0, 80)  # no time taken


def test_virtual_z_alone(connected, configs):  # noqa: F811
    sequence = PulseSequence([("q0/drive", VirtualZ(phase=0.1)), readout()])
    run(connected, configs, [sequence])


@pytest.mark.parametrize("chirp", [(1_000_000, "Hz/nsec"), (1_000_000, "GHz/sec")])
def test_chirp(connected, compiled, configs, chirp):  # noqa: F811
    """A chirp of 1 MHz/ns adds the phase pi * 1e-3 * t**2 (t in ns) to the samples."""
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    time, plain = drive_output(connected, configs, compiled, [pulse])
    chirped = pulse.model_copy(update={"chirp": chirp})
    _, out = drive_output(connected, configs, compiled, [chirped])
    t = time[:200]
    assert out[:200] == pytest.approx(
        plain[:200] * np.exp(1j * np.pi * 1e-3 * t**2), abs=1e-5
    )


def test_chirp_refused(connected, configs):  # noqa: F811
    pulse = Pulse(duration=100, amplitude=0.5, envelope=Rectangular())
    for chirp, error in [
        (([1, 2], "Hz/nsec"), NotImplementedError),
        (([1, 2], [50, 50], "Hz/nsec"), NotImplementedError),
        ((1, "Hz/day"), ValueError),
    ]:
        chirped = pulse.model_copy(update={"chirp": chirp})
        with pytest.raises(error, match="chirp"):
            run(connected, configs, [PulseSequence([("q0/drive", chirped)])])
    flux = pulse.model_copy(update={"chirp": (1, "Hz/nsec")})
    with pytest.raises(ValueError, match="chirp"):
        run(connected, configs, [PulseSequence([("q0/flux", flux)])])


def test_precompensation():
    """Exponential filters keep their order, FIR filters are convolved into one."""
    filters = [
        ExponentialFilter(amplitude=-0.1, tau=40),
        FiniteImpulseResponseFilter(coefficients=[1.0, 0.5]),
        ExponentialFilter(amplitude=0.2, tau=2000),
        FiniteImpulseResponseFilter(coefficients=[1.0, -0.25]),
    ]
    comp = precompensation(filters, sampling_rate=2.0)
    assert [e.timeconstant for e in comp.exponential] == pytest.approx([20e-9, 1e-6])
    assert [e.amplitude for e in comp.exponential] == pytest.approx([-0.1, 0.2])
    assert comp.FIR.coefficients == pytest.approx([1.0, 0.25, -0.125])
    assert precompensation([], sampling_rate=2.0) is None


def test_flux_filters_compile(connected, compiled, configs):  # noqa: F811
    """The precompensation compiles and runs with the HDAWG ``PC`` option."""
    filters = [
        ExponentialFilter(amplitude=-0.1, tau=40),
        FiniteImpulseResponseFilter(coefficients=[1.0, 0.1]),
    ]
    configs["q0/flux"] = configs["q0/flux"].model_copy(update={"filters": filters})
    flux = Pulse(duration=100, amplitude=0.4, envelope=Rectangular())
    run(connected, configs, [PulseSequence([("q0/flux", flux), readout()])])


@pytest.mark.parametrize(
    ("filters", "limit"),
    [
        ([ExponentialFilter(amplitude=-0.1, tau=40)] * 9, "8 exponential"),
        ([FiniteImpulseResponseFilter(coefficients=[0.1] * 41)], "40 coefficients"),
    ],
)
def test_flux_filters_limits(connected, configs, filters, limit):  # noqa: F811
    """LabOne Q refuses more than the HDAWG holds."""
    configs["q0/flux"] = configs["q0/flux"].model_copy(update={"filters": filters})
    flux = Pulse(duration=100, amplitude=0.4, envelope=Rectangular())
    with pytest.raises(Exception, match=limit):
        run(connected, configs, [PulseSequence([("q0/flux", flux), readout()])])


def probe_segments(controller, compiled, length=2.2e-6):
    """Start and end in ns of each pulse on the QA line of all the probes."""
    return segments(*wave(controller, compiled[-1], "q0/probe", length))


def test_readout_after_delay(connected, compiled, configs):  # noqa: F811
    """A delay on the acquisition channel delays the probe as well."""
    sequence = PulseSequence([("q0/acquisition", Delay(duration=140)), readout()])
    run(connected, configs, [sequence])
    assert probe_segments(connected, compiled) == [(140, 1140)]


def test_readouts_at_different_times(connected, compiled, configs):  # noqa: F811
    """Two qubits of one QA line, read out one after the other (audit G3/X03)."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence(
        [
            ("q0/drive", drive),
            ("q0/acquisition", Delay(duration=40)),
            readout(0),
            ("q1/drive", drive.new()),
            ("q1/acquisition", Delay(duration=1104)),
            readout(1),
        ]
    )
    run(connected, configs, [sequence])
    assert probe_segments(connected, compiled) == [(40, 1040), (1104, 2104)]


def test_overlapping_readouts_different_start(connected, configs):  # noqa: F811
    """Readouts on one QA line that overlap must start together: LabOne Q refuses."""
    sequence = PulseSequence(
        [readout(0), ("q1/acquisition", Delay(duration=48)), readout(1)]
    )
    with pytest.raises(ValueError, match="must start together"):
        run(connected, configs, [sequence])


def test_two_readouts_of_one_qubit(connected, compiled, configs):  # noqa: F811
    """Mid-circuit measurement: readout, drive, readout again (F47)."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence(
        [
            readout(0),
            ("q0/drive", Delay(duration=1000)),
            ("q0/drive", drive),
            ("q0/acquisition", Delay(duration=40)),
            readout(0),
        ]
    )
    results = run(connected, configs, [sequence])
    assert len(results) == 2
    assert probe_segments(connected, compiled) == [(0, 1000), (1040, 2040)]
    time, out = wave(connected, compiled[-1], "q0/drive", 2.5e-6)
    assert segments(time, out) == [(1000, 1040)]


def test_readouts_sub_grid(connected, compiled, configs):  # noqa: F811
    """A readout 4 ns off the 8 ns grid of the readout sections starts on time."""
    sequence = PulseSequence([("q0/acquisition", Delay(duration=1004)), readout()])
    run(connected, configs, [sequence])
    assert probe_segments(connected, compiled) == [(1004, 2004)]


def test_readouts_sub_grid_mismatch(connected, configs):  # noqa: F811
    """Readouts on one QA line start at the same time modulo 8 ns."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    sequence = PulseSequence(
        [
            readout(0),
            ("q0/drive", Delay(duration=1000)),
            ("q0/drive", drive),
            ("q0/acquisition", Delay(duration=44)),
            readout(0),
        ]
    )
    with pytest.raises(ValueError, match="modulo 8.0 ns"):
        run(connected, configs, [sequence])
    first = PulseSequence([readout(0)])
    second = PulseSequence([("q1/acquisition", Delay(duration=4)), readout(1)])
    with pytest.raises(ValueError, match="modulo 8.0 ns"):
        run(connected, configs, [first, second])


def test_readouts_too_close(connected, configs):  # noqa: F811
    """The section of a readout ends on the grid after it: the next one waits."""
    probe = PROBE.model_copy(update={"duration": 1001})
    sequence = PulseSequence([readout(probe=probe), readout()])
    with pytest.raises(ValueError, match="too close"):
        run(connected, configs, [sequence])


def test_sequences_follow_relaxation(connected, compiled, configs):  # noqa: F811
    """The second sequence starts after the first one and the relaxation time."""
    drive = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
    first = PulseSequence(
        [("q0/drive", drive), ("q0/acquisition", Delay(duration=40)), readout(0)]
    )
    second = PulseSequence(
        [("q1/drive", drive.new()), ("q1/acquisition", Delay(duration=40)), readout(1)]
    )
    run(connected, configs, [first, second])
    # first: 40 + 1000 ns, then 1000 ns relaxation
    assert probe_segments(connected, compiled, 3.5e-6) == [(40, 1040), (2080, 3080)]
    time, out = wave(connected, compiled[-1], "q1/drive", 3.5e-6)
    assert segments(time, out) == [(2040, 2080)]
