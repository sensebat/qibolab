"""CP1-CP5 reference files: what the qibocal calls hand to LabOne Q, kept on disk.

Each test runs the exact qibocal calls of one checkpoint (the functions of
test_checkpoints.py) on the invented rack, records every experiment the driver hands
to LabOne Q and the results it returns, writes them in qibolab terms and compares
them with ``references/<laboneq major.minor>/cpN.json``.

What a reference holds, per ``ZiController.play`` call:

- ``execution``: shots, averaging mode and acquisition type of the acquisition loop;
- ``channels``: per qibolab channel reached by an experiment signal, its frequency
  (LO + oscillator, all values when swept), LO, DC offset, power range, the waits
  after its acquisitions, and its operations in the order LabOne Q plays them
  (amplitude, phase, duration, a summary of the samples; consecutive delays summed);
- ``sweeps``: outer first, real time or not, and per parameter its values and what
  it drives (``<channel>.frequency``, ``<channel>[<op>].amplitude``, ...);
- ``results``: the qibolab results, per readout.

Readouts are named ``<sequence>:<channel>:<n-th on the channel>``. What the reference
leaves out, as not fixed by qibolab: uids, section names and nesting, the order of
signals, of sweep parameters within one sweep and of operations on different signals,
and the bookkeeping fields that differ between laboneq versions. Numbers are compared
to 9 significant digits.

When to regenerate (``--update-references``, see conftest_references.py): after a
driver change that is meant to change what LabOne Q gets, or when a laboneq version
is added; review the diff of the JSON files like code. Never regenerate to make a
failing test pass without reading the diff. A laboneq version without references is
skipped.

Each test runs under the injected faults of mutants.py that change its reference. Two
bugs change none: ``phase_dropped`` (no checkpoint sets a pulse phase) and
``pulse_on_wrong_channel`` (it swaps q0 and q1 drive pulses, which are equal in every
checkpoint). The frequency faults leave CP1 and CP2 alone where the swept frequency
replaces the static one. The checkpoint's own result checks are switched off while
recording, so a caught bug is caught by the reference.

Pattern: quantify-scheduler golden files (``REGENERATE_REF_FILES``,
tests/scheduler/backends/test_zhinst_backend.py at quantify-os/quantify-scheduler
dcd2405f81) and qblox-scheduler Q1ASM baselines with a regenerate flag
(qblox/qblox-scheduler d9c77ed9); expected-program comparison of
qiboteam/qibolab PR #1039; effects read as in zhinst/laboneq-applications v26.4.0
tests/helpers/dsl.py.
"""

import json
import math
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from laboneq.dsl.experiment import Acquire, PlayPulse

from qibolab._core.instruments.zi import ZiController

from . import effects
from . import test_checkpoints as cps
from .mutants import kills
from .test_checkpoints import checkpoint

LABONEQ = ".".join(version("laboneq").split(".")[:2])
"""laboneq major.minor: one reference folder per version."""

REFERENCES = Path(__file__).parent / "references" / LABONEQ

DIGITS = 9
"""Significant digits kept and compared."""


# numbers


def number(x):
    """A float rounded to :data:`DIGITS`, ``-0`` and tiny values as 0."""
    x = float(x)
    if abs(x) < 1e-12:
        return 0.0
    return float(f"{x:.{DIGITS}g}")


def plain(x):
    """JSON-ready copy: arrays to lists, floats rounded, enums by name."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, bool | str) or x is None:
        return x
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return number(x)
    if isinstance(x, complex | np.complexfloating):
        return [number(x.real), number(x.imag)]
    if hasattr(x, "name"):
        return x.name
    raise TypeError(f"Cannot write {type(x).__name__} in a reference.")


def summary(samples: np.ndarray) -> dict:
    """Samples in a few numbers: count, sum of I and of Q, largest magnitude."""
    return {
        "n": len(samples),
        "sum": [samples.real.sum(), samples.imag.sum()],
        "max": np.abs(samples).max() if len(samples) else 0.0,
    }


# recording


@contextmanager
def recording() -> Iterator[list[dict]]:
    """Record each ``ZiController.play`` call made while the block runs.

    Each entry has the sequences, the experiments compiled, and the results.
    """
    calls = []
    original = ZiController.play

    def spy(self, configs, sequences, options, sweepers):
        with effects.capture(self) as cap:
            cap.results = original(self, configs, sequences, options, sweepers)
        calls.append({"sequences": sequences, "capture": cap})
        return cap.results

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ZiController, "play", spy)
        # the checkpoint's own shape checks are not what this file tests
        mp.setattr(cps, "check", lambda *args, **kwargs: None)
        yield calls


def readout_names(sequences) -> dict[str, str]:
    """``<sequence>:<channel>:<n>`` per acquisition id (as a string)."""
    names = {}
    for n, sequence in enumerate(sequences):
        count = {}
        for channel, acquisition in sequence.acquisitions:
            k = count.get(channel, 0)
            count[channel] = k + 1
            names.setdefault(str(acquisition.id), f"{n}:{channel}:{k}")
    return names


# writing one experiment in qibolab terms


def optional(read: Callable, *args):
    """``read(*args)``, or ``None`` when the channel has no such value."""
    try:
        return read(*args)
    except (effects.Missing, AttributeError, TypeError):
        return None


def operation(cap: effects.Capture, op: effects.Op, names: dict) -> dict:
    if isinstance(op.op, PlayPulse):
        entry = {
            "amplitude": cap.amplitude(op),
            "phase": cap.phase(op),
            "duration": cap.duration(op),
            "samples": summary(cap.samples(op)),
        }
        for field in ("increment_oscillator_phase", "set_oscillator_phase"):
            value = effects.values(getattr(op.op, field, None))
            if value is not None:
                entry[field] = value
        return {"play": entry}
    if isinstance(op.op, Acquire):
        return {
            "acquire": {
                "readout": names.get(op.op.handle, "unknown"),
                "duration": cap.duration(op),
                "kernel": summary(cap.samples(op)),
            }
        }
    return {"delay": effects.values(op.op.time) / effects.NS}


def operations(cap: effects.Capture, channel: str, names: dict, labels: dict):
    """Operations on the channel, consecutive delays summed, zero delays dropped.

    ``labels`` gets ``<channel>[<index>]`` for each LabOne Q operation.
    """
    entries, position = [], {}
    for op in cap.ops(channel):
        if not op.real_time:
            raise RuntimeError(f"Operation outside the acquisition loop on {channel}.")
        entry = operation(cap, op, names)
        if "delay" in entry and entries and "delay" in entries[-1]:
            entries[-1]["delay"] = entries[-1]["delay"] + entry["delay"]
        else:
            entries.append(entry)
        position[id(op.op)] = len(entries) - 1
    kept, index = [], {}
    for n, entry in enumerate(entries):
        if "delay" in entry and np.all(np.asarray(entry["delay"]) == 0):
            continue
        index[n] = len(kept)
        kept.append(entry)
    for key, n in position.items():
        labels[key] = f"{channel}[{index.get(n, 'dropped')}]"
    return kept


def describe(cap: effects.Capture, names: dict) -> dict:
    """One experiment in qibolab terms (see the module doc)."""
    signals = list(cap.experiment.signals)
    reached = sorted({cap.channel_of(s) for s in signals} - {None})
    labels = {}
    channels = {}
    for channel in reached:
        calib = optional(cap.calibration, channel)
        entry = {
            "frequency": optional(cap.frequency, channel),
            "lo": optional(cap.lo, channel),
            "offset": optional(cap.offset, channel),
            "range": effects.values(getattr(calib, "range", None)),
            "ops": operations(cap, channel, names, labels),
        }
        waits = cap.waits_after_acquires(channel)
        if waits:
            entry["waits_after_acquires"] = waits
        channels[channel] = entry

    def drives(parameter) -> list[str]:
        found = set()
        for driven in cap.driven_by(parameter):
            where = labels[id(driven.op.op)] if driven.op else driven.channel
            found.add(f"{where}.{driven.quantity}")
        return sorted(found)

    sweeps = []
    for loop in cap.sweeps():
        parameters = [
            {"values": np.asarray(p.values), "drives": drives(p)}
            for p in loop.parameters
        ]
        parameters.sort(key=lambda p: (p["drives"], list(p["values"])))
        sweeps.append({"real_time": loop.real_time, "parameters": parameters})

    return {
        "execution": {
            "nshots": cap.nshots,
            "averaging_mode": cap.averaging_mode,
            "acquisition_type": cap.acquisition_type,
        },
        "unmapped_signals": sum(cap.channel_of(s) is None for s in signals),
        "channels": channels,
        "sweeps": sweeps,
    }


def reference(calls: list[dict]) -> list[dict]:
    """The reference of the ``play`` calls of one checkpoint case."""
    out = []
    for call in calls:
        cap = call["capture"]
        names = readout_names(call["sequences"])
        experiments = []
        for experiment in cap.experiments:
            one = effects.Capture(cap.controller, experiments=[experiment])
            experiments.append(describe(one, names))
        results = {
            names.get(str(id), "unknown"): np.asarray(result)
            for id, result in cap.results.items()
        }
        out.append(
            {"experiments": experiments, "results": dict(sorted(results.items()))}
        )
    return plain(out)


# comparing


def differences(actual, expected, path="") -> Iterator[str]:
    """Paths where two references differ, numbers within :data:`DIGITS`."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in expected:
                yield f"{path}/{key}: not in the reference"
            elif key not in actual:
                yield f"{path}/{key}: missing"
            else:
                yield from differences(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            yield f"{path}: {len(actual)} items, reference {len(expected)}"
            return
        for n, (a, e) in enumerate(zip(actual, expected)):
            yield from differences(a, e, f"{path}/{n}")
    elif isinstance(expected, float | int) and isinstance(actual, float | int):
        if not math.isclose(actual, expected, rel_tol=10**-DIGITS, abs_tol=1e-12):
            yield f"{path}: {actual} != {expected}"
    elif actual != expected:
        yield f"{path}: {actual!r} != {expected!r}"


def dump(data) -> str:
    """JSON with lists of numbers kept on one line."""
    text = json.dumps(data, indent=1, sort_keys=True)
    return re.sub(
        r"\[\s+([-0-9.e+,\s]+?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        text,
    )


def compare(request, platform, name: str, cases: dict[str, Callable]) -> None:
    """Run each case, then compare with (or write) ``references/<version>/<name>.json``."""
    path = REFERENCES / f"{name}.json"
    update = request.config.getoption("update_references", default=False)
    mutated = request.node.callspec.params.get("mutant") is not None
    if update and mutated:
        pytest.skip("references are written from the driver only")
    if not update and not path.exists():
        pytest.skip(f"no reference for laboneq {LABONEQ}: run --update-references")

    actual = {}
    for case, call in cases.items():
        with recording() as calls:
            call()
        actual[case] = reference(calls)

    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump(actual) + "\n")
    expected = json.loads(path.read_text())
    found = list(differences(actual, expected))
    shown = "; ".join(found[:5])
    assert not found, (
        f"{name} differs from references/{LABONEQ}/{name}.json at "
        f"{len(found)} places: {shown}"
    )


# the checkpoints


@checkpoint("CP1")
@kills(
    "relaxation_zero",
    "results_q_is_i",
    "sweep_grid_reversed",
    "frequency_sweep_offset",
    "acquisition_handles_swapped",
    "nshots_ignored",
    "samples_q_is_i",
)
def test_cp1_reference(platform, request):
    """CP1 resonator spectroscopy: probe frequency sweep on q0 and q1.

    Pattern: quantify-scheduler golden files; qblox-scheduler baselines.
    """
    compare(
        request,
        platform,
        "cp1",
        {"default": lambda: cps.test_resonator_spectroscopy(platform)},
    )


@checkpoint("CP2")
@kills(
    "amplitude_sweep_ignored",
    "relaxation_zero",
    "results_q_is_i",
    "sweep_grid_reversed",
    "frequency_sweep_offset",
    "acquisition_handles_swapped",
    "nshots_ignored",
    "samples_q_is_i",
)
def test_cp2_reference(platform, request):
    """CP2 resonator punchout: probe amplitude (outer) x frequency (inner).

    Pattern: quantify-scheduler golden files; qblox-scheduler baselines.
    """
    compare(
        request,
        platform,
        "cp2",
        {"default": lambda: cps.test_resonator_punchout(platform)},
    )


@checkpoint("CP3")
@kills(
    "relaxation_zero",
    "results_q_is_i",
    "sweep_grid_reversed",
    "drive_signal_misrouted",
    "frequency_sweep_offset",
    "acquisition_handles_swapped",
    "nshots_ignored",
    "lo_not_subtracted",
    "lo_sign_flipped",
    "samples_q_is_i",
)
def test_cp3_reference(platform, request):
    """CP3 qubit spectroscopy, narrow span: drive frequency sweep in one batch.

    The wide span is refused by LabOne Q (LO off grid, see test_checkpoints.py) and
    has no reference.

    Pattern: quantify-scheduler golden files; qblox-scheduler baselines.
    """
    compare(
        request,
        platform,
        "cp3",
        {"narrow": lambda: cps.test_qubit_spectroscopy(platform, 4e6, 1e6)},
    )


@checkpoint("CP4")
@kills(
    "amplitude_sweep_ignored",
    "relaxation_zero",
    "results_q_is_i",
    "sweep_grid_reversed",
    "drive_signal_misrouted",
    "acquisition_handles_swapped",
    "nshots_ignored",
    "lo_not_subtracted",
    "lo_sign_flipped",
    "samples_q_is_i",
)
def test_cp4_reference(platform, request):
    """CP4 Rabi amplitude: drive amplitude sweep, with RX and with RX90 twice.

    Pattern: quantify-scheduler golden files; qblox-scheduler baselines.
    """
    compare(
        request,
        platform,
        "cp4",
        {
            "RX": lambda: cps.test_rabi_amplitude_signal(platform, False),
            "RX90": lambda: cps.test_rabi_amplitude_signal(platform, True),
        },
    )


@checkpoint("CP5")
@kills(
    "relaxation_zero",
    "results_q_is_i",
    "sweep_grid_reversed",
    "drive_signal_misrouted",
    "frequency_sweep_offset",
    "acquisition_handles_swapped",
    "nshots_ignored",
    "lo_not_subtracted",
    "lo_sign_flipped",
    "samples_q_is_i",
)
def test_cp5_reference(platform, request):
    """CP5 single-shot classification and dispersive shift: two sequences, 0 and 1.

    Pattern: quantify-scheduler golden files; qblox-scheduler baselines.
    """
    compare(
        request,
        platform,
        "cp5",
        {
            "default": lambda: cps.test_single_shot_classification_and_dispersive_shift(
                platform
            )
        },
    )
