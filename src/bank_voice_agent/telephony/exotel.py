"""Exotel Voicebot applet (bidirectional AgentStream).

Per developer.exotel.com/docs/agentstream: Exotel sends connected / start / media / dtmf / mark / stop events;
the bot sends media (base64 PCM16 LE mono, default 8 kHz, chunks of 3,200–100,000 bytes in multiples of 320),
mark, and clear (flush buffered audio — used for barge-in).

Flow setup in the Exotel dashboard:  Voicebot applet (wss://<host>/telephony/exotel/ws)  →  Connect applet (human
queue). When the bot closes the stream, Exotel continues to the next applet, which is how transfer works.
For a plain hangup the bot calls the Exotel hangup API before closing.
"""

from __future__ import annotations

import base64
import json

import httpx
from fastapi import WebSocket
from loguru import logger

from bank_voice_agent.config import TelephonySettings
from bank_voice_agent.speech.audio import chunk
from bank_voice_agent.telephony.base import InboundFrame

EXOTEL_MIN_CHUNK = 3200


class ExotelAdapter:
    def parse(self, message: str) -> InboundFrame | None:
        msg = json.loads(message)
        ev = msg.get("event")
        if ev == "start":
            s = msg.get("start", {})
            return InboundFrame("start", call_id=s.get("call_sid", ""), stream_id=s.get("stream_sid", ""),
                                from_number=s.get("from", ""), to_number=s.get("to", ""),
                                custom=s.get("custom_parameters", {}) or {})
        if ev == "media":
            payload = msg.get("media", {}).get("payload", "")
            if payload:
                return InboundFrame("audio", audio=base64.b64decode(payload))
            return None
        if ev == "dtmf":
            return InboundFrame("dtmf", digit=str(msg.get("dtmf", {}).get("digit", "")))
        if ev == "mark":
            return InboundFrame("mark", mark=msg.get("mark", {}).get("name", ""))
        if ev == "stop":
            return InboundFrame("stop")
        return None


class ExotelTransport:
    def __init__(self, ws: WebSocket, cfg: TelephonySettings):
        self.ws = ws
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.stream_sid = ""
        self.call_sid = ""
        self._pending = b""
        self._seq = 0
        self.transfer_requested: str | None = None

    async def _send(self, payload: dict) -> None:
        self._seq += 1
        msg = {**payload, "stream_sid": self.stream_sid, "sequence_number": str(self._seq)}
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception as e:
            logger.warning(f"Exotel _send failed ({payload.get('event')}): {e}")

    async def send_audio(self, pcm16: bytes) -> None:
        self._pending += pcm16
        # Exotel requirement: chunks must be 3,200–100,000 bytes in multiples of 320.
        # 3,200 bytes = 200ms at 8kHz 16-bit mono.
        while len(self._pending) >= EXOTEL_MIN_CHUNK:
            c = self._pending[:EXOTEL_MIN_CHUNK]
            self._pending = self._pending[EXOTEL_MIN_CHUNK:]
            await self._send({"event": "media", "media": {"payload": base64.b64encode(c).decode()}})

    async def flush(self) -> None:
        if self._pending:
            data, self._pending = self._pending, b""
            # Pad to at least EXOTEL_MIN_CHUNK and multiple of 320
            if len(data) < EXOTEL_MIN_CHUNK:
                data = data + b"\x00" * (EXOTEL_MIN_CHUNK - len(data))
            elif len(data) % 320 != 0:
                data = data + b"\x00" * (320 - (len(data) % 320))
            for i in range(0, len(data), EXOTEL_MIN_CHUNK):
                c = data[i: i + EXOTEL_MIN_CHUNK]
                if len(c) < EXOTEL_MIN_CHUNK:
                    c = c + b"\x00" * (EXOTEL_MIN_CHUNK - len(c))
                await self._send({"event": "media", "media": {"payload": base64.b64encode(c).decode()}})

    async def mark(self, name: str) -> None:
        await self.flush()
        await self._send({"event": "mark", "mark": {"name": name}})

    async def clear(self) -> None:
        self._pending = b""
        await self._send({"event": "clear"})

    async def hangup(self) -> None:
        await self.flush()
        if self.cfg.exotel_api_key and self.call_sid:
            url = (f"https://{self.cfg.exotel_api_key}:{self.cfg.exotel_api_token}@{self.cfg.exotel_subdomain}"
                   f"/v1/Accounts/{self.cfg.exotel_sid}/Calls/{self.call_sid}")
            try:
                async with httpx.AsyncClient(timeout=5) as c:
                    await c.post(url, data={"Status": "completed"})
            except httpx.HTTPError as e:
                logger.warning(f"exotel hangup API failed: {e}")
        await self.ws.close()

    async def transfer(self, reason: str) -> None:
        # Closing the stream hands the call to the next applet (human queue) in the Exotel flow.
        self.transfer_requested = reason
        await self.flush()
        await self.ws.close()
