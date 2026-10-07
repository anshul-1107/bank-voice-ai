"""Sarvam AI streaming STT (Saaras) and TTS (Bulbul) over WebSockets.

Endpoints and message shapes per docs.sarvam.ai (checked Oct 2026):
  STT  wss://api.sarvam.ai/speech-to-text-realtime/ws  ?model=&language_code=&encoding=&sample_rate=&mode=
       send {"event": "audio_input", "audio": <b64>}
       recv types: session.begin, vad.speech_start, vad.speech_end, transcript.partial, transcript.final, error
  TTS  wss://api.sarvam.ai/text-to-speech/ws
       send {"type": "config", "data": {...}} once, then {"type": "text", "data": {"text": ...}} + {"type": "flush"}
       recv {"type": "audio", "data": {"audio": <b64>, ...}}
Field names inside payloads are parsed tolerantly; run tests/test_sarvam_live.py against your key before go-live.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets
from loguru import logger

from bank_voice_agent.config import SarvamSettings
from bank_voice_agent.speech.audio import resample_pcm16
from bank_voice_agent.speech.base import STTEvent


def _first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return default


class SarvamStreamingSTT:
    def __init__(self, cfg: SarvamSettings, sample_rate: int = 8000, language: str = "unknown"):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.language = language if language != "unknown" else "auto"
        self._ws = None
        self._q: asyncio.Queue[STTEvent] = asyncio.Queue()
        self._reader: asyncio.Task | None = None

    async def start(self) -> None:
        params = urlencode({"model": self.cfg.stt_model, "language_code": self.language, "encoding": "linear16",
                            "sample_rate": self.sample_rate, "mode": self.cfg.stt_mode, "stream_type": "fast",
                            "endpointing": "vad"})
        self._ws = await websockets.connect(f"{self.cfg.stt_url}?{params}",
                                            additional_headers={"API-SUBSCRIPTION-KEY": self.cfg.api_key},
                                            max_size=2**22, ping_interval=10)
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                kind = _first(msg, "type", "event", default="")
                data = msg.get("data", msg)
                if kind.endswith("speech_start"):
                    await self._q.put(STTEvent("speech_start"))
                elif kind.endswith("speech_end"):
                    await self._q.put(STTEvent("speech_end"))
                elif kind.endswith("partial"):
                    await self._q.put(STTEvent("partial", _first(data, "transcript", "text", default="")))
                elif kind.endswith("final"):
                    await self._q.put(STTEvent("final", _first(data, "transcript", "text", default=""),
                                               lang=_first(data, "language_code", "language")))
                elif kind == "error":
                    logger.error(f"sarvam stt error: {msg}")
                    await self._q.put(STTEvent("error", str(data)))
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._q.put(STTEvent("closed"))

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws:
            await self._ws.send(json.dumps({"event": "audio_input", "audio": base64.b64encode(pcm16).decode()}))

    async def events(self) -> AsyncIterator[STTEvent]:
        while True:
            ev = await self._q.get()
            if ev.kind == "closed":
                return
            yield ev

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        if self._reader:
            self._reader.cancel()


class SarvamStreamingTTS:
    """One persistent socket per call (saves ~150 ms handshake per sentence). Sentences are serialised with a lock;
    a cancelled sentence is drained to its completion event before the socket is reused."""

    def __init__(self, cfg: SarvamSettings, out_rate: int = 8000, speaker: str | None = None):
        self.cfg = cfg
        self.sample_rate = out_rate
        self.speaker = speaker or cfg.tts_speaker
        self.engine_rate = 22050 if cfg.tts_model.endswith("v2") else 24000
        self._ws = None
        self._lang = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._ws = await websockets.connect(f"{self.cfg.tts_url}?send_completion_event=true",
                                            additional_headers={"Api-Subscription-Key": self.cfg.api_key},
                                            max_size=2**23, ping_interval=10)

    async def _configure(self, lang: str) -> None:
        if lang == self._lang:
            return
        await self._ws.send(json.dumps({"type": "config", "data": {
            "language_code": "hi-IN" if lang == "hi" else "en-IN", "speaker": self.speaker,
            "model": self.cfg.tts_model, "pace": self.cfg.tts_pace, "speech_sample_rate": str(self.engine_rate),
            "output_audio_codec": "linear16", "enable_preprocessing": True,
            "min_buffer_size": 30, "max_chunk_length": 150}}))
        self._lang = lang

    async def synthesize(self, text: str, lang: str) -> AsyncIterator[bytes]:
        async with self._lock:
            if self._ws is None:
                await self.start()
            await self._configure(lang)
            await self._ws.send(json.dumps({"type": "text", "data": {"text": text}}))
            await self._ws.send(json.dumps({"type": "flush"}))
            done = False
            try:
                while not done:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "audio":
                        pcm = base64.b64decode(msg["data"]["audio"])
                        if pcm[:4] == b"RIFF":       # some codecs arrive wrapped in a WAV header
                            pcm = pcm[44:]
                        yield resample_pcm16(pcm, self.engine_rate, self.sample_rate)
                    elif t in ("event", "completion") or msg.get("data", {}).get("event_type") == "final":
                        done = True
                    elif t == "error":
                        logger.error(f"sarvam tts error: {msg}")
                        done = True
            except (GeneratorExit, asyncio.CancelledError):
                # barge-in: drain this sentence so the socket stays in sync for the next one
                asyncio.create_task(self._drain())
                raise

    async def _drain(self) -> None:
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=3.0))
                if msg.get("type") in ("event", "completion", "error"):
                    return
        except Exception:
            self._ws, self._lang = None, None   # reconnect on next sentence

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
