"""Planted driver bugs, to show that each effect test can fail.

Each mutant is a function of a ``pytest.MonkeyPatch`` that patches the driver modules
of ElStabilini/qibolab@zi_drivers (e8757e58) for one test; the source is never
edited. :func:`kills` turns an effect test into one test on the driver, which must
pass, and one per named mutant, which must fail with an ``AssertionError``.

The first five are the mutants that all ``play()`` tests of
ElStabilini/qibolab@zi_drivers (e8757e58) tests/instruments/zi/test_play.py miss.

Pattern: mutation testing as in mutmut / cosmic-ray, applied by monkeypatch.
"""

import inspect
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import numpy as np
import pytest
from laboneq.core.exceptions import LabOneQException
from laboneq.simple import pulse_library

from qibolab._core.components import AcquisitionChannel, IqChannel
from qibolab._core.instruments.zi import controller, pulse, results, sweep
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


class _Experiment:
    """Experiment proxy whose ``play``/``acquire`` calls can be rewritten."""

    def __init__(self, experiment, play=None, acquire=None):
        self._experiment = experiment
        self._play = play
        self._acquire = acquire

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


def _wrap_pulse_play(mp, play=None, acquire=None, sequence=None):
    """Patch ``pulse.play`` to rewrite its LabOne Q calls (and its sequence)."""
    original = pulse.play

    def patched(experiment, seq, channels, configs, sampling_rate, amplitudes):
        if sequence is not None:
            seq = sequence(seq, channels)
        proxy = _Experiment(experiment, play, acquire)
        return original(proxy, seq, channels, configs, sampling_rate, amplitudes)

    mp.setattr(pulse, "play", patched)


def _wrap_controller_play(mp, update):
    """Patch ``ZiController.play`` to change the execution parameters it gets."""
    original = controller.ZiController.play

    def patched(self, configs, sequences, options, sweepers):
        options = options.model_copy(update=update(options))
        return original(self, configs, sequences, options, sweepers)

    mp.setattr(controller.ZiController, "play", patched)


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


# further mutants


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
    original = controller.ZiController._experiment

    def patched(self, ids, calib):
        experiment = original(self, ids, calib)
        drives = sorted(
            id
            for id, ch in self.channels.items()
            if isinstance(ch, IqChannel) and id.endswith("/drive")
        )
        for id in ids:
            if id in drives:
                other = drives[(drives.index(id) + 1) % len(drives)]
                experiment.map_signal(
                    id, self.session.device_setup.logical_signal_by_uid(other)
                )
        return experiment

    mp.setattr(controller.ZiController, "_experiment", patched)


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


@mutant("nshots is ignored: 1000 shots always")
def nshots_ignored(mp):
    _wrap_controller_play(mp, lambda options: {"nshots": 1000})


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


@mutant("pulse samples: Q is replaced by I")
def samples_q_is_i(mp):
    original = pulse.waveform

    @cache
    def waveform(envelope, duration, sampling_rate):
        samples = np.asarray(original(envelope, duration, sampling_rate).samples)
        return pulse_library.sampled_pulse(samples.real + 1j * samples.real)

    mp.setattr(pulse, "waveform", waveform)


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
