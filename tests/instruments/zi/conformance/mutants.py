"""Planted driver bugs, to show that each effect test can fail.

Each mutant is a function of a ``pytest.MonkeyPatch`` that patches the driver modules
of ElStabilini/qibolab@zi_drivers (e8757e58) for one test; the source is never
edited. :func:`kills` turns an effect test into one test on the driver, which must
pass, and one per named mutant, which must fail with an ``AssertionError``.

The first five are the mutants that all ``play()`` tests of
ElStabilini/qibolab@zi_drivers (e8757e58) tests/instruments/zi/test_play.py miss.
TEETH.md lists, for each mutant, the tests of the whole ZI suite that it turns red.

Pattern: mutation testing as in mutmut / cosmic-ray, applied by monkeypatch.
"""

import inspect
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import laboneq.simple as lq
import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException
from laboneq.simple import pulse_library

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.execution_parameters import AcquisitionType, AveragingMode
from qibolab._core.instruments.zi import controller, pulse, results, sweep
from qibolab._core.pulses import Align, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter

__all__ = ["MUTANTS", "Mutant", "apply", "kills"]


class Mutant(NamedTuple):
    description: str
    patch: Callable[[pytest.MonkeyPatch], None]
    laboneq_may_refuse: bool = False
    """LabOne Q may reject the mutated experiment itself (version dependent)."""


MUTANTS: dict[str, Mutant] = {}


def mutant(description: str, laboneq_may_refuse: bool = False):
    def register(patch):
        MUTANTS[patch.__name__] = Mutant(description, patch, laboneq_may_refuse)
        return patch

    return register


# ways to patch the driver


class _Experiment:
    """Experiment proxy whose ``play``/``acquire``/``delay`` calls can be rewritten."""

    def __init__(self, experiment, play=None, acquire=None, keep_delay=None):
        self._experiment = experiment
        self._play = play
        self._acquire = acquire
        self._keep_delay = keep_delay

    def __getattr__(self, name):
        return getattr(self._experiment, name)

    def play(self, signal, **kwargs):
        if self._play is not None:
            signal, kwargs = self._play(self._experiment, signal, kwargs)
        return self._experiment.play(signal, **kwargs)

    def acquire(self, signal, **kwargs):
        if self._acquire is not None:
            signal, kwargs = self._acquire(self._experiment, signal, kwargs)
        return self._experiment.acquire(signal, **kwargs)

    def delay(self, signal, **kwargs):
        if self._keep_delay is not None and not self._keep_delay(signal):
            return None
        return self._experiment.delay(signal, **kwargs)


def _wrap_pulse_play(mp, play=None, acquire=None, sequence=None, keep_delay=None):
    """Patch ``pulse.play`` to rewrite its LabOne Q calls (and its sequence).

    The relaxation delays, added by ``ZiController.play`` itself, are not affected.
    """
    original = pulse.play

    def patched(experiment, seq, channels, configs, sampling_rate, amplitudes):
        if sequence is not None:
            seq = sequence(seq, channels)
        proxy = _Experiment(experiment, play, acquire, keep_delay)
        return original(proxy, seq, channels, configs, sampling_rate, amplitudes)

    mp.setattr(pulse, "play", patched)


def _wrap_controller_play(mp, update=None, sequences=None, refusal=None):
    """Patch ``ZiController.play``: change its options or sequences, or reword its
    refusals."""
    original = controller.ZiController.play

    def patched(self, configs, seqs, options, sweepers):
        if update is not None:
            options = options.model_copy(update=update(options))
        if sequences is not None:
            seqs = sequences(seqs)
        try:
            return original(self, configs, seqs, options, sweepers)
        except NotImplementedError:
            if refusal is None:
                raise
            raise NotImplementedError(refusal) from None

    mp.setattr(controller.ZiController, "play", patched)


def _calibration(mp, change):
    """Patch ``calibration`` to apply ``change(channel, item)`` to each signal."""
    original = controller.calibration

    def calibration(ids, channels, instruments, configs):
        calib = original(ids, channels, instruments, configs)
        for id in ids:
            change(channels[id], calib[id])
        return calib

    mp.setattr(controller, "calibration", calibration)


def _misroute(mp, kind, suffix=""):
    """Map the signals of channels of ``kind`` to the next such channel's logical
    signal."""
    original = controller.ZiController._experiment

    def patched(self, ids, calib):
        experiment = original(self, ids, calib)
        same = sorted(
            id
            for id, ch in self.channels.items()
            if isinstance(ch, kind) and id.endswith(suffix)
        )
        for id in ids:
            if id in same:
                other = same[(same.index(id) + 1) % len(same)]
                experiment.map_signal(
                    id, self.session.device_setup.logical_signal_by_uid(other)
                )
        return experiment

    mp.setattr(controller.ZiController, "_experiment", patched)


def _oscillators(mp, frequency):
    """Patch ``calibration`` to set each oscillator to ``frequency(config, lo)``."""
    original = controller.calibration

    def calibration(ids, channels, instruments, configs):
        calib = original(ids, channels, instruments, configs)
        seen = set()
        for channel_id in ids:
            if isinstance(channels[channel_id], AcquisitionChannel):
                continue
            oscillator = calib[channel_id].oscillator
            if oscillator is None or id(oscillator) in seen:
                continue
            seen.add(id(oscillator))
            lo = configs[channels[channel_id].lo].frequency
            oscillator.frequency = frequency(configs[channel_id].frequency, lo)
        return calib

    mp.setattr(controller, "calibration", calibration)


def _samples(mp, change):
    """Patch ``pulse.waveform`` to apply ``change`` to the complex samples."""
    original = pulse.waveform

    @cache
    def waveform(envelope, duration, sampling_rate):
        samples = np.asarray(original(envelope, duration, sampling_rate).samples)
        return pulse_library.sampled_pulse(change(samples))

    mp.setattr(pulse, "waveform", waveform)


# the five mutants all test_play.py tests miss


@mutant("swept amplitudes are not used when playing")
def amplitude_sweep_ignored(mp):
    original = pulse.play

    def patched(experiment, seq, channels, configs, sampling_rate, amplitudes):
        return original(experiment, seq, channels, configs, sampling_rate, {})

    mp.setattr(pulse, "play", patched)


@mutant("relaxation time is 0")
def relaxation_zero(mp):
    _wrap_controller_play(mp, lambda options: {"relaxation_time": 0})


@mutant("relative phase of pulses is dropped")
def phase_dropped(mp):
    def play(experiment, signal, kwargs):
        kwargs.pop("phase", None)
        return signal, kwargs

    _wrap_pulse_play(mp, play=play)


@mutant("results: Q is replaced by I")
def results_q_is_i(mp):
    original = results.parse

    def parse(data, options, axes):
        result = original(data, options, axes)
        result[..., 1] = result[..., 0]
        return result

    mp.setattr(results, "parse", parse)


@mutant("sweep values are reversed")
def sweep_grid_reversed(mp):
    original = sweep._group

    def group(*args):
        parameters, real_time = original(*args)
        for parameter in parameters:
            parameter.values = parameter.values[::-1]
        return parameters, real_time

    mp.setattr(sweep, "_group", group)


# routing: signals and channels


@mutant("pulses on a drive channel are played on the drive of another qubit")
def pulse_on_wrong_channel(mp):
    def play(experiment, signal, kwargs):
        drives = sorted(
            s for s in experiment.signals if s.endswith("/drive") and s != signal
        )
        if signal.endswith("/drive") and drives:
            signal = drives[0]
        return signal, kwargs

    _wrap_pulse_play(mp, play=play)


@mutant("drive signals are mapped to the logical signal of the next qubit's drive")
def drive_signal_misrouted(mp):
    _misroute(mp, IqChannel, "/drive")


@mutant("flux signals are mapped to the logical signal of the next qubit's flux")
def flux_signal_misrouted(mp):
    _misroute(mp, DcChannel)


@mutant("the probe pulses of the first two readouts are swapped between qubits")
def readout_probe_swapped(mp):
    def play(experiment, signal, kwargs):
        probes = sorted(s for s in experiment.signals if s.endswith("/probe"))[:2]
        if signal in probes and len(probes) == 2:
            signal = probes[1 - probes.index(signal)]
        return signal, kwargs

    _wrap_pulse_play(mp, play=play)


@mutant("DC channels unused by the sequences are left out of the experiment")
def unused_dc_channels_dropped(mp):
    original = controller.signals

    def signals(channels, sequences):
        used = {id for sequence in sequences for id in sequence.channels}
        return [
            id
            for id in original(channels, sequences)
            if not isinstance(channels[id], DcChannel) or id in used
        ]

    mp.setattr(controller, "signals", signals)


# sweeps


@mutant("frequency sweep values are offset by one step")
def frequency_sweep_offset(mp):
    original = sweep._group

    def group(sweeper_group, channels, configs, calibration, amplitudes):
        parameters, real_time = original(
            sweeper_group, channels, configs, calibration, amplitudes
        )
        frequency = [s for s in sweeper_group if s.parameter is Parameter.frequency]
        if frequency:
            for parameter in parameters:
                if all(parameter is not a for a in amplitudes.values()):
                    step = parameter.values[1] - parameter.values[0]
                    parameter.values = parameter.values + step
        return parameters, real_time

    mp.setattr(sweep, "_group", group)


@mutant("nested sweepers are run in reverse order (grid transposed)")
def nested_sweepers_swapped(mp):
    original = sweep.split

    def split(sweepers, *args):
        return original(list(sweepers)[::-1], *args)

    mp.setattr(sweep, "split", split)


# frequencies and channel configs


@mutant(
    "the LO frequency is not subtracted from the oscillator frequency",
    laboneq_may_refuse=True,
)
def lo_not_subtracted(mp):
    """The oscillator gets the full frequency, GHz: laboneq 26.10 refuses it when
    compiling (above Nyquist), laboneq 26.4 compiles it."""
    _oscillators(mp, lambda frequency, lo: frequency)


@mutant("the oscillator frequency is LO - frequency instead of frequency - LO")
def lo_sign_flipped(mp):
    """Stays within the oscillator range, so LabOne Q accepts it on every version."""
    _oscillators(mp, lambda frequency, lo: lo - frequency)


@mutant("power_range is not passed to LabOne Q")
def power_range_dropped(mp):
    def change(channel, item):
        item.range = None

    _calibration(mp, change)


@mutant("the DC offset of a flux channel is not applied (0 V)")
def dc_offset_dropped(mp):
    def change(channel, item):
        if isinstance(channel, DcChannel):
            item.voltage_offset = 0.0

    _calibration(mp, change)


@mutant("the acquisition delay is not applied")
def acquisition_delay_dropped(mp):
    def change(channel, item):
        if isinstance(channel, AcquisitionChannel):
            item.port_delay = None

    _calibration(mp, change)


@mutant("the calibration of the first call is reused by later calls")
def calibration_cached(mp):
    original = controller.calibration
    cached = {}

    def calibration(ids, channels, instruments, configs):
        key = tuple(ids)
        if key not in cached:
            cached[key] = original(ids, channels, instruments, configs)
        return cached[key]

    mp.setattr(controller, "calibration", calibration)


# pulses and sequences


@mutant("pulse samples: Q is replaced by I")
def samples_q_is_i(mp):
    _samples(mp, lambda samples: samples.real + 1j * samples.real)


@mutant("pulse samples: I and Q are swapped")
def samples_iq_swapped(mp):
    _samples(mp, lambda samples: samples.imag + 1j * samples.real)


@mutant("samples are computed at the HDAWG rate (2.4 GSa/s) on SHF instruments")
def sampling_rate_hdawg(mp):
    mp.setattr(
        controller.ZiController,
        "sampling_rate",
        property(lambda self: controller.HDAWG_SAMPLING_RATE),
    )


@mutant("delays of a sequence are not played")
def delays_dropped(mp):
    _wrap_pulse_play(mp, keep_delay=lambda signal: False)


@mutant("a delay on an acquisition channel does not delay its probe channel")
def probe_delay_dropped(mp):
    _wrap_pulse_play(mp, keep_delay=lambda signal: not signal.endswith("/probe"))


@mutant("Align is ignored")
def align_ignored(mp):
    def sequence(seq, channels):
        return PulseSequence([(ch, p) for ch, p in seq if not isinstance(p, Align)])

    _wrap_pulse_play(mp, sequence=sequence)


@mutant("several sequences are played in reverse order")
def sequences_reversed(mp):
    _wrap_controller_play(mp, sequences=lambda sequences: sequences[::-1])


@mutant("the integration length is the probe duration, not the acquisition one")
def integration_length_from_probe(mp):
    def probe_long(op):
        if not isinstance(op, Readout):
            return op
        acquisition = op.acquisition.model_copy(update={"duration": op.probe.duration})
        return op.model_copy(update={"acquisition": acquisition})

    _wrap_pulse_play(
        mp,
        sequence=lambda seq, channels: PulseSequence(
            [(ch, probe_long(op)) for ch, op in seq]
        ),
    )


# execution parameters


@mutant("nshots is ignored: 1000 shots always")
def nshots_ignored(mp):
    _wrap_controller_play(mp, lambda options: {"nshots": 1000})


@mutant("CYCLIC and SINGLESHOT averaging are swapped")
def averaging_mode_swapped(mp):
    swap = {
        AveragingMode.CYCLIC: AveragingMode.SINGLESHOT,
        AveragingMode.SINGLESHOT: AveragingMode.CYCLIC,
    }
    _wrap_controller_play(
        mp,
        lambda options: {
            "averaging_mode": swap.get(options.averaging_mode, options.averaging_mode)
        },
    )


class _Discriminating:
    """``laboneq.simple`` whose ``AcquisitionType.INTEGRATION`` is DISCRIMINATION."""

    class AcquisitionType:
        INTEGRATION = lq.AcquisitionType.DISCRIMINATION

    def __getattr__(self, name):
        return getattr(lq, name)


@mutant("LabOne Q acquires in DISCRIMINATION mode instead of INTEGRATION")
def acquisition_type_discrimination(mp):
    mp.setattr(controller, "lq", _Discriminating())


@mutant("any acquisition type runs as INTEGRATION")
def acquisition_type_unchecked(mp):
    _wrap_controller_play(
        mp, lambda options: {"acquisition_type": AcquisitionType.INTEGRATION}
    )


@mutant("any averaging mode runs as CYCLIC")
def averaging_mode_unchecked(mp):
    _wrap_controller_play(mp, lambda options: {"averaging_mode": AveragingMode.CYCLIC})


@mutant("refusals do not say what is not supported")
def refusal_messages_vague(mp):
    _wrap_controller_play(mp, refusal="Not supported.")


# results


@mutant("results are reshaped to the qibolab shape instead of moving the axes")
def results_axes_reshaped(mp):
    original = results.parse

    def parse(data, options, axes):
        shape = original(data, options, axes).shape[:-1]
        data = np.asarray(data).reshape(shape)
        return np.stack([np.real(data), np.imag(data)], axis=-1)

    mp.setattr(results, "parse", parse)


@mutant("results keep the LabOne Q axis order (near time, shots, real time)")
def results_axes_not_moved(mp):
    def parse(data, options, axes):
        return np.stack([np.real(data), np.imag(data)], axis=-1)

    mp.setattr(results, "parse", parse)


@mutant("acquisition handles are swapped between the first two readouts")
def acquisition_handles_swapped(mp):
    def sequence(seq, channels):
        handles = [str(ro.id) for _, ro in seq.acquisitions][:2]
        swap.clear()
        if len(handles) == 2:
            swap.update({handles[0]: handles[1], handles[1]: handles[0]})
        return seq

    swap = {}

    def acquire(experiment, signal, kwargs):
        kwargs["handle"] = swap.get(kwargs["handle"], kwargs["handle"])
        return signal, kwargs

    _wrap_pulse_play(mp, acquire=acquire, sequence=sequence)


def apply(mp: pytest.MonkeyPatch, name: str) -> None:
    """Plant the mutant ``name`` for the rest of the test."""
    MUTANTS[name].patch(mp)


def _first_line(error: Exception) -> str:
    lines = [line for line in str(error).splitlines() if line.strip()]
    return lines[0] if lines else type(error).__name__


def kills(*names: str):
    """Run the test on the driver (must pass) and under each mutant (must fail).

    The test gets one more parameter, ``driver`` or ``mutant-<name>``. Under a
    mutant, the test must raise ``AssertionError`` (an assertion of the test, or
    :class:`effects.Missing`), whose first line is printed. A ``LabOneQException``
    also counts for a mutant registered with ``laboneq_may_refuse``. Any other
    exception is an error, and passing means the mutant survived.
    """
    for name in names:
        assert name in MUTANTS, f"Unknown mutant '{name}'."

    def decorate(test):
        signature = inspect.signature(test)
        extra = [
            inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY)
            for n in ("mutant", "monkeypatch")
            if n not in signature.parameters
        ]

        def wrapper(*args, mutant, monkeypatch, **kwargs):
            if "monkeypatch" in signature.parameters:
                kwargs["monkeypatch"] = monkeypatch
            if mutant is None:
                return test(*args, **kwargs)
            apply(monkeypatch, mutant)
            try:
                test(*args, **kwargs)
            except AssertionError as error:
                print(f"mutant '{mutant}' caught: {_first_line(error)}")
                return
            except LabOneQException as error:
                if not MUTANTS[mutant].laboneq_may_refuse:
                    raise
                print(f"mutant '{mutant}' refused by LabOne Q: {_first_line(error)}")
                return
            pytest.fail(f"mutant '{mutant}' survived", pytrace=False)

        wrapper.__name__ = test.__name__
        wrapper.__qualname__ = test.__qualname__
        wrapper.__doc__ = test.__doc__
        wrapper.__module__ = test.__module__
        wrapper.__dict__.update(test.__dict__)
        wrapper.__signature__ = signature.replace(
            parameters=[*signature.parameters.values(), *extra]
        )
        cases = [pytest.param(None, id="driver")] + [
            pytest.param(n, id=f"mutant-{n}") for n in names
        ]
        return pytest.mark.parametrize("mutant", cases)(wrapper)

    return decorate
