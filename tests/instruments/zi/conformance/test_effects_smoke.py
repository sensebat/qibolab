"""Smoke test of :mod:`effects` and :mod:`mutants`.

One effect, observed at the LabOne Q boundary: a pulse of amplitude 0.3 on
``q0/drive`` makes LabOne Q play amplitude 0.3 on the signal that reaches the port of
``q0/drive``, and nothing else there. It runs on the driver (must pass) and under two
planted bugs that send the pulse elsewhere (must fail).

Also checks that the ``effects`` field of :data:`features.FEATURES` matches the
``feature`` markers of the effect tests (the files of ``conftest.EFFECT_FILES``).

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q, by signal); mutation testing (mutmut, cosmic-ray).
"""

import importlib
from pathlib import Path

import pytest

from qibolab._core.pulses import Delay
from qibolab._core.sequence import PulseSequence

from . import effects
from .conftest import EFFECT_FILES
from .effects import square
from .features import FEATURES
from .mutants import kills
from .test_conformance import mz


@pytest.mark.feature("F24")
@kills("pulse_on_wrong_channel", "drive_signal_misrouted")
def test_drive_amplitude_reaches_drive_port(platform):
    """A pulse of amplitude 0.3 on q0/drive plays amplitude 0.3 on q0's drive port.

    Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the
    DSL experiment, per signal).
    """
    sequence = PulseSequence(
        [
            ("q0/drive", square(amplitude=0.3)),
            ("q1/drive", square(amplitude=0.2)),
            ("q0/acquisition", Delay(duration=40)),
        ]
    )
    sequence += mz(platform, "q0")

    cap = effects.play(platform, [sequence])

    plays = cap.plays("q0/drive")
    assert [cap.amplitude(op) for op in plays] == [pytest.approx(0.3)]
    assert cap.duration(plays[0]) == pytest.approx(40)


def _features(test) -> set[str]:
    """Feature ids a test carries, on the function or on its parametrized cases."""
    ids = set()
    for mark in getattr(test, "pytestmark", []):
        if mark.name == "feature":
            ids.add(mark.args[0])
        if mark.name == "parametrize":
            for case in mark.args[1]:
                for inner in getattr(case, "marks", ()):
                    if inner.name == "feature":
                        ids.add(inner.args[0])
    return ids


def test_features_list_their_effect_tests():
    """``FEATURES[...]["effects"]`` lists exactly the effect tests marked with the
    feature, so the report shows which features have effect tests.

    Pattern: none harvested; a bookkeeping check of features.py, not an effect test.
    """
    marked = {}
    here = Path(__file__).parent
    paths = {path for pattern in EFFECT_FILES for path in here.glob(pattern)}
    for path in sorted(paths):
        module = importlib.import_module(f"{__package__}.{path.stem}")
        for name, test in vars(module).items():
            if name.startswith("test_") and callable(test):
                for id in _features(test):
                    marked.setdefault(id, set()).add(f"{path.name}::{name}")
    listed = {f["id"]: set(f["effects"]) for f in FEATURES}
    assert set(marked) <= set(listed)
    assert {id: tests for id, tests in listed.items() if tests} == marked
