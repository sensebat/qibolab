"""Smoke test of :mod:`effects` and :mod:`mutants`.

One effect, observed at the LabOne Q boundary: a pulse of amplitude 0.3 on
``q0/drive`` makes LabOne Q play amplitude 0.3 on the signal that reaches the port of
``q0/drive``, and nothing else there. It runs on the driver (must pass) and under two
planted bugs that send the pulse elsewhere (must fail).

Pattern: zhinst/laboneq-applications v26.4.0 tests/helpers/dsl.py (assert on the DSL
experiment handed to LabOne Q, by signal); mutation testing (mutmut, cosmic-ray).
"""

import pytest

from qibolab._core.pulses import Delay, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence

from . import effects
from .mutants import kills
from .test_conformance import mz


def square(amplitude):
    return Pulse(duration=40, amplitude=amplitude, envelope=Rectangular())


@kills("pulse_on_wrong_channel", "drive_signal_misrouted")
def test_drive_amplitude_reaches_drive_port(platform):
    sequence = PulseSequence(
        [
            ("q0/drive", square(0.3)),
            ("q1/drive", square(0.2)),
            ("q0/acquisition", Delay(duration=40)),
        ]
    )
    sequence += mz(platform, "q0")

    cap = effects.play(platform, [sequence])

    plays = cap.plays("q0/drive")
    assert [cap.amplitude(op) for op in plays] == [pytest.approx(0.3)]
    assert cap.duration(plays[0]) == pytest.approx(40)
