"""Offline speech stand-ins, used by tests and the call simulator.

MockSTT: the simulated caller sends frames that start with b"TXT:" + utf-8 text (one utterance) or silence.
         It emits speech_start → partials → final, like a real streaming STT.
MockTTS: returns silent PCM whose length matches natural speech rate (~2.6 words/s), so timing, barge-in
         and latency logic behave as on a real call.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from bank_voice_agent.speech.base import STTEvent


class MockSTT:
    def __init__(self, partial_delay_s: float = 0.02):
        self._q: asyncio.Queue[STTEvent] = asyncio.Queue()
        self.delay = partial_delay_s

    async def start(self) -> None: ...

    async def send_audio(self, pcm16: bytes) -> None:
        if not pcm16.startswith(b"TXT:"):
            return
        text = pcm16[4:].decode()
        await self._q.put(STTEvent("speech_start"))
        words = text.split()
        for i in range(1, len(words)):
            await asyncio.sleep(self.delay)
            await self._q.put(STTEvent("partial", " ".join(words[:i])))
        await asyncio.sleep(self.delay)
        await self._q.put(STTEvent("final", text))
        await self._q.put(STTEvent("speech_end"))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev.kind == "closed":
                return
            yield ev

    async def close(self) -> None:
        await self._q.put(STTEvent("closed"))


class MockTTS:
    def __init__(self, sample_rate: int = 8000, words_per_s: float = 2.6, first_chunk_delay_s: float = 0.0):
        self.sample_rate = sample_rate
        self.wps = words_per_s
        self.delay = first_chunk_delay_s
        self.spoken: list[tuple[str, str]] = []

    async def start(self) -> None: ...

    async def synthesize(self, text: str, lang: str) -> AsyncIterator[bytes]:
        self.spoken.append((lang, text))
        if self.delay:
            await asyncio.sleep(self.delay)
        seconds = max(0.3, len(text.split()) / self.wps)
        total = int(seconds * self.sample_rate) * 2
        step = 3200
        for i in range(0, total, step):
            yield b"\x00" * min(step, total - i)
            await asyncio.sleep(0)

    async def close(self) -> None: ...
