"""Streaming speech interfaces. Real-time = audio in and out as small chunks, never whole utterances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class STTEvent:
    kind: str                 # speech_start | partial | final | speech_end | error
    text: str = ""
    lang: str | None = None
    confidence: float | None = None


class StreamingSTT(Protocol):
    async def start(self) -> None: ...
    async def send_audio(self, pcm16: bytes) -> None: ...
    def events(self) -> AsyncIterator[STTEvent]: ...
    async def close(self) -> None: ...


class StreamingTTS(Protocol):
    sample_rate: int

    async def start(self) -> None: ...
    def synthesize(self, text: str, lang: str) -> AsyncIterator[bytes]: ...   # pcm16 mono at sample_rate
    async def close(self) -> None: ...
