"""Provider-neutral telephony transport over a bidirectional media WebSocket."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class InboundFrame:
    kind: str                         # start | audio | dtmf | mark | stop
    audio: bytes = b""
    call_id: str = ""
    stream_id: str = ""
    from_number: str = ""
    to_number: str = ""
    digit: str = ""
    mark: str = ""
    custom: dict = field(default_factory=dict)


class Transport(Protocol):
    sample_rate: int

    async def send_audio(self, pcm16: bytes) -> None: ...
    async def mark(self, name: str) -> None: ...
    async def clear(self) -> None: ...            # flush audio queued at the provider: required for barge-in
    async def hangup(self) -> None: ...
    async def transfer(self, reason: str) -> None: ...


class ProviderAdapter(Protocol):
    def parse(self, message: str) -> InboundFrame | None: ...
