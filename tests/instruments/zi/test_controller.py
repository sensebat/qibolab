"""Tests ``controller.py``."""

import pytest

pytest.importorskip("laboneq")

from laboneq.core.exceptions import LabOneQException
from pydantic import ValidationError

from qibolab._core.components import AcquisitionChannel, DcChannel, IqChannel
from qibolab._core.instruments.zi import ZiController, ZiInstrument
from qibolab._core.instruments.zi.controller import PQSC_UID, device_setup

from .conftest import INSTRUMENTS, NQUBITS, PQSC


def _recipe(scheduled):
    """Device initializations: ``recipe`` in laboneq 26.4, ``artifacts`` in 26.10."""
    return getattr(scheduled, "recipe", None) or scheduled.artifacts


def _connections(setup):
    """Map logical signal paths to ``(instrument, port)``."""
    return {
        conn.remote_path.split("/", 2)[-1]: (instrument.uid, conn.local_port)
        for instrument in setup.instruments
        for conn in instrument.connections
    }


def test_device_setup(controller):
    setup = device_setup(controller.address, INSTRUMENTS, PQSC, controller.channels)

    assert {i.uid for i in setup.instruments} == set(INSTRUMENTS) | {PQSC_UID}
    connections = _connections(setup)
    assert len(connections) == 4 * NQUBITS
    for i in range(NQUBITS):
        assert connections[f"q{i}/drive"] == ("shfsg", f"SGCHANNELS/{i}/OUTPUT")
        assert connections[f"q{i}/flux"] == ("hdawg", f"SIGOUTS/{i}")
        assert connections[f"q{i}/probe"] == ("shfqa", "QACHANNELS/0/OUTPUT")
        assert connections[f"q{i}/acquisition"] == ("shfqa", "QACHANNELS/0/INPUT")


def test_device_setup_without_pqsc():
    channels = {"q0/drive": IqChannel(device="shfsg", path="0")}
    setup = device_setup("127.0.0.1:8004", INSTRUMENTS, None, channels)
    assert PQSC_UID not in {i.uid for i in setup.instruments}


def test_device_setup_extra_drive_on_same_port(channels):
    channels["q0/drive12"] = channels["q0/drive"]
    setup = device_setup("127.0.0.1:8004", INSTRUMENTS, PQSC, channels)
    connections = _connections(setup)
    assert connections["q0/drive12"] == connections["q0/drive"]


@pytest.mark.parametrize(
    "id,channel",
    [
        # not a ``group/name`` id
        ("drive0", IqChannel(device="shfsg", path="0")),
        ("q0/drive/(1, 2)", IqChannel(device="shfsg", path="0")),
        # unknown instrument
        ("q0/drive", IqChannel(device="missing", path="0")),
        # channel type not supported by the instrument
        ("q0/drive", IqChannel(device="hdawg", path="0")),
        ("q0/flux", DcChannel(device="shfsg", path="0")),
        ("q0/acquisition", AcquisitionChannel(device="shfsg", path="0")),
        # port not on the instrument (SHFQA4: ports 0-3)
        ("q0/probe", IqChannel(device="shfqa", path="4")),
        ("q0/drive", IqChannel(device="shfsg", path="-1")),
    ],
)
def test_device_setup_invalid_channel(id, channel):
    with pytest.raises(ValueError):
        device_setup("127.0.0.1:8004", INSTRUMENTS, PQSC, {id: channel})


@pytest.mark.parametrize(
    "kind,options,nports",
    [
        ("SHFSG", "SHFSG8", 8),
        ("SHFSG", "SHFSG4", 4),
        ("SHFSG", "SHFSG4/RTR", 8),
        ("SHFSG", "SHFSG2", 2),
        ("SHFSG", None, 8),
        ("SHFQA", "SHFQA2", 2),
        ("SHFQA", "SHFQA4/16W", 4),
        ("SHFQA", None, 4),
        ("HDAWG", "HDAWG4/PC", 4),
        ("HDAWG", "HDAWG8/PC", 8),
        ("HDAWG", "PC", 8),
    ],
)
def test_instrument_nports(kind, options, nports):
    assert ZiInstrument(kind=kind, serial="DEV0", options=options).nports == nports


def test_pqsc_uid_reserved():
    instruments = INSTRUMENTS | {PQSC_UID: INSTRUMENTS["hdawg"]}
    with pytest.raises(LabOneQException):
        device_setup("127.0.0.1:8004", instruments, PQSC, {})


def test_unsupported_instrument():
    with pytest.raises(ValidationError):
        ZiController(
            address="127.0.0.1:8004",
            instruments={"qa": {"kind": "UHFQA", "serial": "DEV2001"}},
        )


def test_instrument_from_dict():
    controller = ZiController(
        address="127.0.0.1:8004",
        instruments={"sg": {"kind": "SHFSG", "serial": "DEV12001"}},
    )
    assert controller.instruments["sg"] == ZiInstrument(kind="SHFSG", serial="DEV12001")


def test_connect_disconnect(controller):
    assert not controller.is_connected
    controller.connect()
    assert controller.is_connected
    session = controller.session
    controller.connect()
    assert controller.session is session

    controller.disconnect()
    assert not controller.is_connected
    controller.disconnect()


def test_disconnect_zeroes_dc_offsets(controller, monkeypatch):
    controller.keep_dc_offsets_on = False
    controller.connect()

    compiled = []
    run = controller.session.run

    def spy(experiment, *args, **kwargs):
        compiled.append(experiment)
        return run(experiment, *args, **kwargs)

    monkeypatch.setattr(controller.session, "run", spy)
    controller.disconnect()

    assert not controller.is_connected
    assert len(compiled) == 1
    (hdawg,) = [
        init
        for init in _recipe(compiled[0].scheduled_experiment).initializations
        if init.device_uid == "hdawg"
    ]
    assert {o.channel: o.offset for o in hdawg.outputs} == dict.fromkeys(
        range(NQUBITS), 0.0
    )


def test_disconnect_keeps_dc_offsets(controller, monkeypatch):
    controller.connect()
    runs = []
    monkeypatch.setattr(controller.session, "run", lambda *a, **k: runs.append(a))
    controller.disconnect()
    assert runs == []


def test_sampling_rate(controller):
    assert controller.sampling_rate == 2.0
    hdawgs = {"hdawg": INSTRUMENTS["hdawg"]}
    assert controller.model_copy(update={"instruments": hdawgs}).sampling_rate == 2.4
