"""Helpers to write the channels of a Zurich Instruments platform."""

from qibolab._core.components import AcquisitionChannel, Channel, DcChannel, IqChannel
from qibolab._core.identifier import ChannelId, QubitId
from qibolab._core.qubits import Qubit

from .controller import InstrumentKind, ZiInstrument, _check_port

__all__ = ["Cluster", "infer_los", "map_ports"]

Cluster = dict[str, tuple[ZiInstrument, dict[int, list[QubitId]]]]
"""Wiring: instrument name → (instrument, port number → connected elements).

Pass ``{name: instrument for name, (instrument, _) in cluster.items()}`` as the
controller's ``instruments``.
"""

_CHANNELS: dict[InstrumentKind, dict[str, type[Channel]]] = {
    "SHFSG": {"drive": IqChannel},
    "SHFQA": {"probe": IqChannel, "acquisition": AcquisitionChannel},
    "HDAWG": {"flux": DcChannel},
}
"""Channel types on one port of each instrument kind.

They are keyed by the :class:`Qubit` attribute holding their id.
"""


def _check_kind(name: str, instrument: ZiInstrument) -> None:
    """Check that the ports of an instrument can be described by a :data:`Cluster`.

    An SHFQC has an SG port and a QA port with the same number, so its channels are
    written directly.
    """
    if instrument.kind not in _CHANNELS:
        raise ValueError(
            f"Instrument '{name}' of kind '{instrument.kind}' cannot be described by "
            "port numbers only; write its channels directly."
        )


def _element_channels(
    element: QubitId, el: Qubit, name: str, kind: InstrumentKind, port: int
) -> dict[ChannelId, Channel]:
    """Channels of one element on one port, by channel id."""
    channels = {}
    for attr, cls in _CHANNELS[kind].items():
        id = getattr(el, attr)
        if id is None:
            raise ValueError(
                f"Element '{element}' has no {attr} channel, "
                f"but it is connected to port {port} of '{name}'."
            )
        channel = cls(device=name, path=str(port))
        if isinstance(channel, AcquisitionChannel):
            channel = channel.model_copy(update={"probe": el.probe})
        channels[id] = channel
        if attr == "drive":
            for extra in el.drive_extra.values():
                channels[extra] = channel
    return channels


def map_ports(
    cluster: Cluster,
    qubits: dict[QubitId, Qubit],
    couplers: dict[QubitId, Qubit] | None = None,
) -> dict[ChannelId, Channel]:
    """Create the channels from a compact description of the wiring.

    ``cluster`` maps each instrument name to the instrument and its used ports
    (0-based), and each port to the elements connected to it. Elements are keys of
    ``qubits`` or ``couplers``; several elements on one port are multiplexed.

    The channel types follow from the instrument kind:

    - SHFSG port: a drive :class:`IqChannel`, plus one for each id in the element's
      ``drive_extra`` (e.g. the ef drive), all on the same port;
    - SHFQA port (one QA channel): a probe :class:`IqChannel` and an
      :class:`AcquisitionChannel`, whose ``probe`` is filled;
    - HDAWG port: a flux :class:`DcChannel`.

    Channel ids are taken from the elements (e.g. ``q1/drive`` for
    :meth:`Qubit.default`), so ``drive_extra`` must be set before calling this.
    LO names (see :func:`infer_los`) are added in the platform.

    Ports are checked against the model in :attr:`ZiInstrument.options`. A channel id
    reached from two different ports, e.g. an element listed on two ports, raises an
    error.
    """
    if couplers is None:
        couplers = {}
    common = qubits.keys() & couplers.keys()
    if len(common) > 0:
        raise ValueError(
            f"Elements {sorted(map(str, common))} are both qubits and couplers."
        )
    elements = qubits | couplers

    channels: dict[ChannelId, Channel] = {}
    for name, (instrument, instrument_ports) in cluster.items():
        _check_kind(name, instrument)
        for port, connected in instrument_ports.items():
            _check_port(name, instrument, port)
            for element in connected:
                if element not in elements:
                    raise ValueError(
                        f"Unknown element '{element}' on port {port} of '{name}'."
                    )
                new = _element_channels(
                    element, elements[element], name, instrument.kind, port
                )
                for id, channel in new.items():
                    if id in channels and channels[id] != channel:
                        raise ValueError(
                            f"Channel '{id}' is connected to both "
                            f"port {channels[id].path} of '{channels[id].device}' "
                            f"and port {port} of '{name}'."
                        )
                    channels[id] = channel
    return channels


def _lo_name(name: str, instrument: ZiInstrument, port: int) -> str:
    """LO name of an output port.

    On an SHFSG8 the outputs share a synthesizer in pairs (0/1, 2/3, ...), so the
    pair gets one name.
    """
    if instrument.model == "SHFSG8":
        return f"{name}/synth{port // 2}/lo"
    return f"{name}/o{port}/lo"


def infer_los(cluster: Cluster) -> dict[tuple[QubitId, bool], str]:
    """Infer the LO names of the output channels.

    ``cluster`` is the same input of :func:`map_ports`. The result maps
    ``(element, is_probe)`` to the LO name, one per synthesizer:

    - SHFQA and SHFSG4/SHFSG2 port: ``"{instrument}/o{port}/lo"``;
    - SHFSG8 port: ``"{instrument}/synth{port // 2}/lo"``, since its outputs share a
      synthesizer in pairs, which must have one frequency.

    The SHFSG model is read from :attr:`ZiInstrument.options`. Without it, the names
    are per port; LabOne Q then reports two different frequencies on a pair.
    HDAWG ports have no LO.
    """
    los = {}
    for name, (instrument, instrument_ports) in cluster.items():
        _check_kind(name, instrument)
        if instrument.kind == "HDAWG":
            continue
        is_probe = instrument.kind == "SHFQA"
        for port, connected in instrument_ports.items():
            _check_port(name, instrument, port)
            lo = _lo_name(name, instrument, port)
            for element in connected:
                los[element, is_probe] = lo
    return los
