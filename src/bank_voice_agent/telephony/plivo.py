"""Plivo Audio Streams (bidirectional).

Answer XML (served by /telephony/plivo/answer):
  <Response><Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=8000">
     wss://<host>/telephony/plivo/ws</Stream></Response>
Plivo → bot: start / media / dtmf / playedStream / clearedAudio / stop
Bot → Plivo: playAudio, checkpoint, clearAudio
Transfer: Plivo REST "transfer call" to an XML URL that <Dial>s the human queue.
"""

from __future__ import annotations

import base64
import json

import httpx
from fastapi import WebSocket
from loguru import logger

from bank_voice_agent.config import TelephonySettings
from bank_voice_agent.telephony.base import InboundFrame


class PlivoAdapter:
    def parse(self, message: str) -> InboundFrame | None:
        msg = json.loads(message)
        ev = msg.get("event")
        if ev == "start":
            s = msg.get("start", {})
            return InboundFrame("start", call_id=s.get("callId", ""), stream_id=s.get("streamId", ""),
                                from_number=s.get("from", ""), to_number=s.get("to", ""),
                                custom=msg.get("extra_headers", {}) or {})
        if ev == "media":
            return InboundFrame("audio", audio=base64.b64decode(msg["media"]["payload"]))
        if ev == "dtmf":
            return InboundFrame("dtmf", digit=str(msg.get("dtmf", {}).get("digit", "")))
        if ev == "playedStream":
            return InboundFrame("mark", mark=msg.get("name", ""))
        if ev == "stop":
            return InboundFrame("stop")
        return None


class PlivoTransport:
    def __init__(self, ws: WebSocket, cfg: TelephonySettings):
        self.ws = ws
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.stream_sid = ""
        self.call_sid = ""
        self.transfer_requested: str | None = None

    async def send_audio(self, pcm16: bytes) -> None:
        await self.ws.send_text(json.dumps({"event": "playAudio", "media": {
            "contentType": "audio/x-l16", "sampleRate": self.sample_rate,
            "payload": base64.b64encode(pcm16).decode()}}))

    async def mark(self, name: str) -> None:
        await self.ws.send_text(json.dumps({"event": "checkpoint", "streamId": self.stream_sid, "name": name}))

    async def clear(self) -> None:
        await self.ws.send_text(json.dumps({"event": "clearAudio", "streamId": self.stream_sid}))

    async def _rest(self, method: str, path: str, data: dict | None = None) -> None:
        if not self.cfg.plivo_auth_id:
            return
        url = f"https://api.plivo.com/v1/Account/{self.cfg.plivo_auth_id}/Call/{self.call_sid}/{path}"
        try:
            async with httpx.AsyncClient(timeout=5, auth=(self.cfg.plivo_auth_id, self.cfg.plivo_auth_token)) as c:
                await c.request(method, url, json=data)
        except httpx.HTTPError as e:
            logger.warning(f"plivo {path} failed: {e}")

    async def hangup(self) -> None:
        await self._rest("DELETE", "")
        await self.ws.close()

    async def transfer(self, reason: str) -> None:
        self.transfer_requested = reason
        await self._rest("POST", "", {"legs": "aleg",
                                      "aleg_url": f"{self.cfg.public_base_url}/telephony/plivo/transfer?reason={reason}"})
        await self.ws.close()
