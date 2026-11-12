"""Effects of channel configs, and the errors users get for common mistakes.

Configs. What a config passed to ``ZiController.play`` makes LabOne Q do, read from
the experiment handed to LabOne Q (see :mod:`effects`): the frequency of an
``IqConfig`` is LO + oscillator on the channel's port, with the LO from the
``OscillatorConfig`` the channel names; ``power_range`` is the range of the port;
the ``DcConfig`` offset reaches the HDAWG port of the flux channel, used or not;
the acquisition delay delays the integration relative to the probe.

Errors. Each mistake is made through the qibolab interface. Where the current
message is clear, the test matches it. Where it is not, the test asserts the message
a clear error would have and is a strict xfail that holds only while the call fails
with the current exception type; it turns red once the message is fixed.

Features the driver may refuse (acquisition types, averaging modes, fast reset,
sweep parameters, kernels): the test passes if the call is refused with a
``NotImplementedError`` that names the feature, or if it runs and has its effect
(:func:`refused_or_works`). It never passes on a call that runs silently wrong, and
it stays green the day the feature is implemented. The strict xfails that wait for
the feature itself are elsewhere (test_conformance.py and the effect files).

Every test that must pass carries ``@kills``: it also runs under planted driver bugs
(:mod:`mutants`) and must fail under each of them.

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q); zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2
error tests (``pytest.raises(..., match=...)`` on the message the user sees);
mutation testing (mutmut, cosmic-ray). Sources:
sources/research/driver-tests/README.md (M6) and
sources/review/zi-driver-tests-inventory.md section 4.
"""

import re

import laboneq.simple as lq
import numpy as np
import pytest
from laboneq.dsl.experiment import Case, Match, PlayPulse, Section

from qibolab._core.components import (
    AcquisitionConfig,
    DcConfig,
    IqConfig,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.sweeper import Parameter, Sweeper

from . import effects
from .conftest import unclear
from .effects import drives_then_mz, square
from .mutants import kills
from .test_conformance import mz, readout_of, updated

# local helpers


def expect_clear(call, kind, *patterns):
    """``call`` fails with ``kind`` and a message matching all ``patterns``.

    Otherwise an ``AssertionError`` that starts with the error raised.
    """
    with pytest.raises(Exception) as info:
        call()
    error = info.value
    message = f"{type(error).__name__}: {error}"
    assert isinstance(error, kind), message
    assert all(re.search(p, str(error)) for p in patterns), message


# scenarios


BOTH = {"q0": [square()], "q1": [square()]}
"""A drive pulse on each qubit, then the readout of both at once."""

IQ = ["q0/drive", "q1/drive", "q0/probe", "q1/probe"]


# configs: frequency, LO


@pytest.mark.feature("F40")
@kills("lo_not_subtracted", "lo_sign_flipped", "drive_signal_misrouted")
def test_iq_frequency_is_lo_plus_oscillator(platform):
    """The ``IqConfig`` frequency of each IQ channel is LO + oscillator on its port.

    The LO is the ``OscillatorConfig`` the channel names (shared by the two drives
    and by the two probes); the acquisition demodulates at its probe's frequency.
    Frequencies are taken from the configs passed to ``play``, not the platform's.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    configs = updated(
        platform,
        q0__drive={"frequency": 6.35e9},
        q1__drive={"frequency": 6.12e9},
        q0__probe={"frequency": 6.95e9},
        q1__probe={"frequency": 7.03e9},
    )
    channels = platform.instruments["zi"].channels

    cap = effects.play(platform, [drives_then_mz(platform, BOTH)], configs=configs)

    for id in IQ:
        assert cap.lo(id) == pytest.approx(configs[channels[id].lo].frequency), id
        assert cap.frequency(id) == pytest.approx(configs[id].frequency), id
    for q in ("q0", "q1"):
        probe = configs[f"{q}/probe"].frequency
        assert cap.frequency(f"{q}/acquisition") == pytest.approx(probe), q


@pytest.mark.feature("F40")
@kills("lo_not_subtracted", "lo_sign_flipped")
def test_lo_update_keeps_channel_frequency(platform):
    """Moving an LO in its ``OscillatorConfig`` moves the LO of every channel that
    names it, and keeps each channel at its own frequency.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; qibocal
    qubit_spectroscopy LO batching (qubit_spectroscopies/qubit_spectroscopy.py:212).
    """
    channels = platform.instruments["zi"].channels
    drive_lo, probe_lo = channels["q0/drive"].lo, channels["q0/probe"].lo
    configs = platform.parameters.configs.copy()
    configs[drive_lo] = configs[drive_lo].model_copy(update={"frequency": 6.0e9})
    configs[probe_lo] = configs[probe_lo].model_copy(update={"frequency": 7.2e9})

    cap = effects.play(platform, [drives_then_mz(platform, BOTH)], configs=configs)

    for id in IQ:
        lo = 6.0e9 if id.endswith("drive") else 7.2e9
        assert cap.lo(id) == pytest.approx(lo), id
        assert cap.frequency(id) == pytest.approx(configs[id].frequency), id


@pytest.mark.feature("F43")
@kills("calibration_cached", "lo_sign_flipped")
def test_each_call_uses_its_own_configs(platform):
    """Two calls on one connection: each runs with the frequency of its own configs.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; qibocal
    qubit_spectroscopy batches (qubit_spectroscopies/qubit_spectroscopy.py:218).
    """
    f0 = platform.config("q0/probe").frequency
    for frequency in (f0, f0 + 3e6):
        configs = updated(platform, q0__probe={"frequency": frequency})
        cap = effects.play(platform, [mz(platform)], configs=configs)
        assert cap.frequency("q0/probe") == pytest.approx(frequency)
        assert cap.frequency("q0/acquisition") == pytest.approx(frequency)


# configs: power range


@pytest.mark.checkpoint("CP1")
@kills(
    "power_range_dropped",
    "drive_signal_misrouted",
    "flux_signal_misrouted",
    "unused_dc_channels_dropped",
)
def test_power_range_sets_port_range(platform):
    """``power_range`` of each config is the range of the channel's port: dBm for the
    SHFSG and SHFQA outputs and the SHFQA input, V for the HDAWG outputs.

    Probes on one SHFQA channel share one range, as do their acquisitions.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    ranges = {
        "q0/drive": 10,
        "q1/drive": 0,
        "q0/probe": -5,
        "q1/probe": -5,
        "q0/acquisition": -20,
        "q1/acquisition": -20,
        "q0/flux": 5,
        "q1/flux": 0.8,
    }
    configs = updated(
        platform,
        **{id.replace("/", "__"): {"power_range": r} for id, r in ranges.items()},
    )

    cap = effects.play(platform, [drives_then_mz(platform, BOTH)], configs=configs)

    for id, expected in ranges.items():
        assert cap.calibration(id).range == pytest.approx(expected), id


# configs: DC offset, acquisition delay


@pytest.mark.feature("F38")
@kills("dc_offset_dropped", "unused_dc_channels_dropped", "flux_signal_misrouted")
def test_dc_offset_reaches_hdawg_port(platform):
    """The ``DcConfig`` offset of each flux channel is the offset of its HDAWG port,
    also when the sequences play nothing on it.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; qibocal
    qubit_flux_dependence (flux_dependence/qubit_flux_dependence.py:146).
    """
    configs = updated(platform, q0__flux={"offset": 0.3}, q1__flux={"offset": -0.2})

    cap = effects.play(platform, [mz(platform)], configs=configs)

    assert cap.offset("q0/flux") == pytest.approx(0.3)
    assert cap.offset("q1/flux") == pytest.approx(-0.2)


@pytest.mark.feature("F42")
@kills("acquisition_delay_dropped")
def test_acquisition_delay_delays_integration(platform):
    """The acquisition ``delay`` starts the integration that many ns after the probe.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; qibocal
    time_of_flight_readout (signal_experiments/time_of_flight_readout.py:106).
    """
    configs = updated(platform, q0__acquisition={"delay": 56})

    cap = effects.play(platform, [mz(platform)], configs=configs)

    relative = cap.port_delay("q0/acquisition") - cap.port_delay("q0/probe")
    assert relative == pytest.approx(56)


# errors: refused by name, or working


def refused_or_works(call, names, works):
    """``call`` is refused by name, or it runs and has its effect.

    Refused: a ``NotImplementedError`` whose message matches one of ``names`` (a
    message that does not name the feature is an ``AssertionError`` starting with
    ``NotImplementedError:``). Runs: ``works`` is called with the :class:`Capture`
    and must assert the effect. Never silently wrong; a test written this way stays
    green the day the feature is implemented.
    """
    try:
        cap = call()
    except NotImplementedError as error:
        message = f"NotImplementedError: {error}"
        assert any(re.search(name, str(error)) for name in names), message
        return
    works(cap)


def _shape(sweepers=(), **options):
    """``results_shape`` of the call's execution parameters."""
    parameters = ExecutionParameters(**(effects.DEFAULTS | options))
    return parameters.results_shape(list(sweepers))


@pytest.mark.parametrize(
    "acquisition_type",
    [
        pytest.param(
            AcquisitionType.DISCRIMINATION,
            id="F17-discrimination",
            marks=pytest.mark.feature("F17"),
        ),
        pytest.param(
            AcquisitionType.RAW,
            id="F18-raw",
            marks=pytest.mark.feature("F18"),
        ),
    ],
)
@kills("acquisition_type_unchecked", "refusal_messages_vague")
def test_unsupported_acquisition_type(platform, acquisition_type):
    """DISCRIMINATION and RAW are refused by name, or they reach LabOne Q.

    Working: the acquisition loop has the LabOne Q acquisition type of the same
    name, and each readout returns the qibolab shape (``()`` per point for
    DISCRIMINATION, ``(samples, 2)`` for RAW).

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py; qibolab
    ``ExecutionParameters.results_shape`` (sources/research/driver-tests/README.md,
    item 2).
    """
    sequence = mz(platform)
    name = acquisition_type.name

    def works(cap):
        assert cap.acquisition_type == getattr(lq.AcquisitionType, name)
        result = cap.results[readout_of(sequence).id]
        if acquisition_type is AcquisitionType.RAW:
            assert result.ndim == 2 and result.shape[-1] == 2, result.shape
        else:
            assert result.shape == _shape(acquisition_type=acquisition_type)

    refused_or_works(
        lambda: effects.play(platform, [sequence], acquisition_type=acquisition_type),
        [name],
        works,
    )


@pytest.mark.feature("F21")
@kills("averaging_mode_unchecked", "refusal_messages_vague")
def test_unsupported_averaging_mode(platform):
    """SEQUENTIAL averaging is refused by name, or it reaches LabOne Q.

    Working: the acquisition loop averages SEQUENTIAL and each readout returns the
    qibolab shape.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    mode = AveragingMode.SEQUENTIAL

    def works(cap):
        assert cap.averaging_mode == lq.AveragingMode.SEQUENTIAL
        result = cap.results[readout_of(sequence).id]
        assert result.shape == _shape(averaging_mode=mode)

    refused_or_works(
        lambda: effects.play(platform, [sequence], averaging_mode=mode),
        ["SEQUENTIAL"],
        works,
    )


def _sections(node):
    yield node
    for child in getattr(node, "children", None) or []:
        if isinstance(child, Section):
            yield from _sections(child)


@pytest.mark.feature("F45")
@kills("fast_reset_ignored", "refusal_messages_vague")
def test_fast_reset_refused(platform):
    """``fast_reset`` is refused by name, or it plays a conditional reset.

    Working: LabOne Q gets a ``match`` on the handle of the q0 readout, whose
    ``case(state=1)`` plays on q0/drive (active reset).

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    LabOne Q DSL ``match(handle=...)``/``case(state=...)``, the construct of
    zhinst/laboneq-applications v26.4.0 tunable_transmon ``active_reset``.
    """
    sequence = drives_then_mz(platform, {"q0": [square()]})

    def works(cap):
        handles = set(cap.handles("q0/acquisition"))
        matches = [
            node
            for top in cap.experiment.sections
            for node in _sections(top)
            if isinstance(node, Match) and node.handle in handles
        ]
        assert matches, "no match on the q0 readout handle"
        flips = [
            child
            for match in matches
            for case in match.children
            if isinstance(case, Case) and case.state == 1
            for section in _sections(case)
            for child in section.children
            if isinstance(child, PlayPulse)
            and cap.channel_of(child.signal) == "q0/drive"
        ]
        assert flips, "no play on q0/drive for state 1"

    refused_or_works(
        lambda: effects.play(platform, [sequence], fast_reset=True),
        ["[Ff]ast reset", "fast_reset"],
        works,
    )


def _drive_sweeper(platform, parameter):
    pulse = square()
    sequence = drives_then_mz(platform, {"q0": [pulse]})
    values = np.array([16.0, 32.0] if parameter is Parameter.duration else [0.1, 0.2])
    return [sequence], Sweeper(parameter=parameter, values=values, pulses=[pulse])


def _offset_sweeper(platform, parameter):
    values = np.array([0.1, 0.2])
    sweeper = Sweeper(parameter=parameter, values=values, channels=["q0/flux"])
    return [mz(platform)], sweeper


SWEPT = {
    Parameter.duration: ("q0/drive", "length", 1e-9),
    Parameter.relative_phase: ("q0/drive", "phase", 1.0),
    Parameter.offset: ("q0/flux", "offset", 1.0),
}
"""Swept parameter: channel, LabOne Q quantity it drives, LabOne Q unit per qibolab
unit."""


def _swept_values(cap, channel, quantity):
    """Values of ``quantity`` on ``channel`` driven by any LabOne Q sweep."""
    found = []
    for loop in cap.sweeps():
        for parameter in loop.parameters:
            for driven in cap.driven_by(parameter):
                if driven.channel != channel or driven.quantity != quantity:
                    continue
                if driven.op is not None:
                    x = getattr(driven.op.op, quantity)
                else:
                    x = cap.calibration(channel).voltage_offset
                found.append(np.asarray(effects.values(x)))
    return found


@pytest.mark.parametrize(
    "build, parameter",
    [
        pytest.param(
            _drive_sweeper,
            Parameter.duration,
            id="F06-duration",
            marks=pytest.mark.feature("F06"),
        ),
        pytest.param(
            _drive_sweeper,
            Parameter.relative_phase,
            id="F10-relative_phase",
            marks=pytest.mark.feature("F10"),
        ),
        pytest.param(
            _offset_sweeper,
            Parameter.offset,
            id="F12-offset",
            marks=pytest.mark.feature("F12"),
        ),
    ],
)
@kills("unsupported_sweepers_dropped", "refusal_messages_vague")
def test_unsupported_sweep_parameter(platform, build, parameter):
    """A sweep of duration, relative phase or offset is refused by name, or swept.

    Working: a LabOne Q sweep drives the pulse length or phase on q0/drive, or the
    voltage offset of q0/flux, with the sweeper values (in LabOne Q units), and the
    results have the sweep axis.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py.
    """
    sequences, sweeper = build(platform, parameter)
    channel, quantity, unit = SWEPT[parameter]

    def works(cap):
        found = _swept_values(cap, channel, quantity)
        expected = sweeper.values * unit
        assert any(
            v.shape == expected.shape and np.allclose(v, expected) for v in found
        ), f"{quantity} of {channel} swept over {found}, not {expected}"
        result = cap.results[readout_of(sequences[0]).id]
        assert result.shape == _shape([[sweeper]])

    refused_or_works(
        lambda: effects.play(platform, sequences, [[sweeper]]),
        [parameter.name],
        works,
    )


# errors: mistakes with an unclear message (strict xfail until fixed)


@pytest.mark.parametrize(
    "channel, sequence",
    [
        pytest.param("q0/drive", "drive", marks=pytest.mark.feature("F43")),
        pytest.param("q0/probe", "mz", marks=pytest.mark.feature("F43")),
        pytest.param("q0/acquisition", "mz", marks=pytest.mark.feature("F43")),
        pytest.param("q0/flux", "mz", marks=pytest.mark.feature("F38")),
    ],
)
@unclear(KeyError, "'No config for channel <id>.'")
def test_missing_channel_config(platform, channel, sequence):
    """A channel used by the call without a config in ``configs``.

    Today: ``KeyError: '<id>'``.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    sequences = [
        drives_then_mz(platform, {"q0": [square()]})
        if sequence == "drive"
        else mz(platform)
    ]
    configs = platform.parameters.configs.copy()
    del configs[channel]

    expect_clear(
        lambda: effects.play(platform, sequences, configs=configs),
        ValueError,
        re.escape(channel),
        "config",
    )


@pytest.mark.parametrize("channel", ["q0/drive", "q0/probe"])
@pytest.mark.feature("F40")
@unclear(KeyError, "'LO <name> of channel <id> has no OscillatorConfig.'")
def test_missing_lo_config(platform, channel):
    """The LO a channel names has no ``OscillatorConfig`` in ``configs``.

    Today: ``KeyError: '<lo name>'``, without the channel.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    lo = platform.instruments["zi"].channels[channel].lo
    configs = platform.parameters.configs.copy()
    del configs[lo]

    expect_clear(
        lambda: effects.play(
            platform, [drives_then_mz(platform, {"q0": [square()]})], configs=configs
        ),
        ValueError,
        re.escape(lo),
        re.escape(channel),
    )


@pytest.mark.parametrize(
    "channel, config, expected",
    [
        pytest.param(
            "q0/drive",
            IqConfig(frequency=6.3e9),
            "ZiIqConfig",
            marks=pytest.mark.feature("F40"),
        ),
        pytest.param(
            "q0/probe",
            IqConfig(frequency=7.0e9),
            "ZiIqConfig",
            marks=pytest.mark.feature("F40"),
        ),
        pytest.param(
            "q0/acquisition",
            AcquisitionConfig(delay=200, smearing=0),
            "ZiAcquisitionConfig",
            marks=pytest.mark.feature("F42"),
        ),
        pytest.param(
            "q0/flux",
            DcConfig(offset=0.1),
            "ZiDcConfig",
            marks=pytest.mark.feature("F38"),
        ),
    ],
)
@unclear(AttributeError, "'Channel <id> needs a <ZiConfig>, got <Config>.'")
def test_wrong_config_kind(platform, channel, config, expected):
    """A plain qibolab config where the driver needs its own kind (with
    ``power_range``).

    Today: ``AttributeError: '<Config>' object has no attribute 'power_range'``,
    without the channel or the kind to use.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    configs = platform.parameters.configs.copy()
    configs[channel] = config

    expect_clear(
        lambda: effects.play(
            platform, [drives_then_mz(platform, {"q0": [square()]})], configs=configs
        ),
        (TypeError, ValueError),
        re.escape(channel),
        expected,
    )


@pytest.mark.checkpoint("CP1")
@unclear(AssertionError, "'Not connected: call connect() before play().'")
def test_play_before_connect(platform):
    """``play`` on a controller that is not connected.

    Today: a bare ``AssertionError`` from the ``session`` property.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    zi = platform.instruments["zi"]
    platform.disconnect()
    options = ExecutionParameters(**effects.DEFAULTS)

    expect_clear(
        lambda: zi.play(
            platform.parameters.configs.copy(), [mz(platform)], options, []
        ),
        RuntimeError,
        "connect",
    )


@pytest.mark.feature("F41")
@kills(
    "kernel_ignored",
    "refusal_messages_vague",
    driver_marks=unclear(NotImplementedError, "a message naming the kernel"),
)
def test_kernel_refused(platform):
    """An integration kernel in the acquisition config is refused by name, or used.

    Working: the q0 acquisition integrates with the kernel samples (up to a factor).
    Today: ``NotImplementedError: Readout on channel 'q0/acquisition' is not
    supported.``, which does not name the kernel.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6; zhinst/laboneq-applications v26.4.0
    tests/helpers/dsl.py.
    """
    sequence = mz(platform)
    samples = int(readout_of(sequence).duration * platform.sampling_rate)
    kernel = np.exp(1j * np.linspace(0, np.pi / 2, samples)) * np.linspace(
        1, 2, samples
    )
    configs = updated(platform, q0__acquisition={"kernel": kernel})

    def works(cap):
        (acquire,) = cap.acquires("q0/acquisition")
        weights = cap.samples(acquire)
        assert weights.shape == kernel.shape, weights.shape
        ratio = weights / kernel
        assert np.allclose(ratio, ratio[0]), "integration weights are not the kernel"

    refused_or_works(
        lambda: effects.play(platform, [sequence], configs=configs),
        ["[Kk]ernel"],
        works,
    )


@pytest.mark.feature("F01")
@unclear(AttributeError, "'Sweep the frequency of the probe channel q0/probe.'")
def test_frequency_sweep_on_acquisition_channel(platform):
    """A frequency sweeper on the acquisition channel instead of the probe channel.

    Today: ``AttributeError: 'AcquisitionChannel' object has no attribute 'lo'``.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    f0 = platform.config("q0/probe").frequency
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        range=(f0 - 1e6, f0 + 1e6, 1e6),
        channels=["q0/acquisition"],
    )

    expect_clear(
        lambda: effects.play(platform, [mz(platform)], [[sweeper]]),
        (ValueError, NotImplementedError),
        re.escape("q0/acquisition"),
        re.escape("q0/probe"),
    )
