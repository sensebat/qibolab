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
from contextlib import contextmanager

import numpy as np
import pytest

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
from .effects import drives_then_mz, square
from .mutants import kills
from .test_conformance import mz, readout_of, updated

# local helpers


@contextmanager
def raises(error, match):
    """``pytest.raises``, with "did not raise" as an ``AssertionError``.

    So that ``@kills`` counts a planted bug that removes a check as caught.
    """
    try:
        with pytest.raises(error, match=match) as info:
            yield info
    except pytest.fail.Exception as failed:
        raise AssertionError(str(failed)) from None


def unclear(current: type, clear: str):
    """Strict xfail while the call fails with ``current``, whatever its message.

    ``clear`` says what a clear message would say.
    """
    return pytest.mark.xfail(
        raises=pytest.RaisesExc(AssertionError, match=f"^{current.__name__}: "),
        strict=True,
        reason=f"unclear error message: {current.__name__} instead of {clear}",
    )


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


# errors: refusals with a clear message


@pytest.mark.parametrize(
    "options, message",
    [
        pytest.param(
            {"acquisition_type": AcquisitionType.DISCRIMINATION},
            "AcquisitionType.DISCRIMINATION is not supported.",
            id="F17-discrimination",
            marks=pytest.mark.feature("F17"),
        ),
        pytest.param(
            {"acquisition_type": AcquisitionType.RAW},
            "AcquisitionType.RAW is not supported.",
            id="F18-raw",
            marks=pytest.mark.feature("F18"),
        ),
    ],
)
@kills("acquisition_type_unchecked", "refusal_messages_vague")
def test_unsupported_acquisition_type(platform, options, message):
    """An acquisition type the driver does not implement is refused by name.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests.
    """
    with raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        effects.play(platform, [mz(platform)], **options)


@pytest.mark.parametrize(
    "options, message",
    [
        pytest.param(
            {"averaging_mode": AveragingMode.SEQUENTIAL},
            "AveragingMode.SEQUENTIAL is not supported.",
            id="F21-sequential",
            marks=pytest.mark.feature("F21"),
        ),
    ],
)
@kills("averaging_mode_unchecked", "refusal_messages_vague")
def test_unsupported_averaging_mode(platform, options, message):
    """An averaging mode the driver does not implement is refused by name.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests.
    """
    with raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        effects.play(platform, [mz(platform)], **options)


@pytest.mark.feature("F45")
@kills("refusal_messages_vague")
def test_fast_reset_refused(platform):
    """``fast_reset`` is refused by name.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests.
    """
    with raises(NotImplementedError, match=r"^Fast reset is not supported\.$"):
        effects.play(platform, [mz(platform)], fast_reset=True)


def _drive_sweeper(platform, parameter):
    pulse = square()
    sequence = drives_then_mz(platform, {"q0": [pulse]})
    values = np.array([16.0, 32.0] if parameter is Parameter.duration else [0.1, 0.2])
    return [sequence], Sweeper(parameter=parameter, values=values, pulses=[pulse])


def _offset_sweeper(platform, parameter):
    values = np.array([0.1, 0.2])
    sweeper = Sweeper(parameter=parameter, values=values, channels=["q0/flux"])
    return [mz(platform)], sweeper


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
@kills("refusal_messages_vague")
def test_unsupported_sweep_parameter(platform, build, parameter):
    """A sweep of a parameter the driver does not implement is refused by name.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests.
    """
    sequences, sweeper = build(platform, parameter)
    message = f"Sweeping {parameter.name} is not supported."
    with raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        effects.play(platform, sequences, [[sweeper]])


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
@unclear(NotImplementedError, "a message naming the kernel")
def test_kernel_refused(platform):
    """An integration kernel in the acquisition config.

    Today: ``NotImplementedError: Readout on channel 'q0/acquisition' is not
    supported.``, which does not name the kernel.

    Pattern: zhinst-qcodes 639ed104f4 and qblox-instruments 1.4.2 error tests;
    sources/research/driver-tests/README.md M6.
    """
    sequence = mz(platform)
    samples = int(readout_of(sequence).duration * platform.sampling_rate)
    configs = updated(
        platform, q0__acquisition={"kernel": np.ones(samples, dtype=complex)}
    )

    expect_clear(
        lambda: effects.play(platform, [sequence], configs=configs),
        NotImplementedError,
        "[Kk]ernel",
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
