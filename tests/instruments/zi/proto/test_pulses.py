"""Pulses added on orkun/proto-pulses, run in LabOne Q emulation.

Emulation compiles the experiment and returns placeholder data. Timing and samples
are checked on the compiled output with the LabOne Q output simulator.
"""

import numpy as np
import pytest

pytest.importorskip("laboneq")

from laboneq.simulator.output_simulator import OutputSimulator

from qibolab._core.components import DcChannel
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.zi import ZiDcConfig
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
    assert not rotated[80:160] == pytest.approx(rotated[:80], abs=1e-3)
    assert segments(time, rotated)[0] == (0, 80)  # no time taken


def test_virtual_z_alone(connected, configs):  # noqa: F811
    sequence = PulseSequence([("q0/drive", VirtualZ(phase=0.1)), readout()])
    run(connected, configs, [sequence])
