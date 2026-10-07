"""CallSession — one per phone call. Owns turn-taking, barge-in, fillers, silence handling and the audio player.

Why it feels human (vs. the course repo's ReplyOnPause loop):
  1. Streaming everywhere: STT partials → LLM tokens → TTS per sentence. First audio ~0.6–0.9 s after the caller stops.
  2. Smart end-of-turn: short grace after a final transcript, longer if the caller trails off ("aur...", "matlab...").
  3. Barge-in: caller can interrupt; we stop within one audio chunk and tell the model what was not heard.
     Backchannels ("haan", "ji", "ok") never interrupt.
  4. Fillers that vary ("Ek second, main check kar rahi hoon") while tools run; typing sound only for slow lookups.
  5. Language mirroring per turn with hysteresis (no flip-flopping on "ok").
  6. Silence handling: gentle re-prompt, then a polite goodbye.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass

from loguru import logger

from bank_voice_agent.agent.brain import ActionEvent, Brain, ErrorEvent, SentenceEvent, ToolEndEvent, ToolStartEvent
from bank_voice_agent.agent.persona import Persona
from bank_voice_agent.config import TurnTakingSettings
from bank_voice_agent.qa.record import CallRecorder
from bank_voice_agent.speech.audio import duration_s, keyboard_clicks
from bank_voice_agent.speech.base import StreamingSTT, StreamingTTS
from bank_voice_agent.speech.normalize import detect_lang, verbalize
from bank_voice_agent.telephony.base import InboundFrame, Transport

BACKCHANNELS = {"haan", "ha", "han", "ji", "hmm", "hm", "ok", "okay", "achha", "acha", "accha", "right", "yes", "yeah",
                "theek", "thik", "hai", "sure", "uh-huh", "mm", "haanji", "haan ji", "ji haan"}
_MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
                "dec"}
TRAILING = re.compile(r"\b(and|but|so|because|aur|lekin|par|matlab|toh|to|ki|jo|uh|um|like)\W*$", re.I)

REPROMPT = {"en": ["Hello, are you still there?", "Sorry, I can't hear you. Are you there?"],
            "hi": ["Hello ji, kya aap line par hain?", "Sorry, aapki awaaz nahi aa rahi. Aap sun pa rahe hain?"]}
GOODBYE_SILENT = {"en": "I'm not able to hear you, so I'll end the call now. Please call us back anytime. Thank you!",
                  "hi": "Aapki awaaz nahi aa rahi, main call end kar rahi hoon. Aap kabhi bhi call kar sakte hain. "
                        "Dhanyavaad!"}
MAX_DURATION = {"en": "I'm sorry, I need to wrap up now. I'm connecting you to a colleague who can continue.",
                "hi": "Maaf kijiye, mujhe ab call wrap up karni hogi. Main aapko colleague se connect kar rahi hoon."}


@dataclass
class _Say:
    text: str
    lang: str
    turn_id: int
    kind: str = "speech"


@dataclass
class _Sound:
    pcm: bytes
    turn_id: int


class CallSession:
    def __init__(self, transport: Transport, stt: StreamingSTT, tts: StreamingTTS, brain: Brain, persona: Persona,
                 recorder: CallRecorder, turn_cfg: TurnTakingSettings, lang: str = "en", seed: int | None = None):
        self.transport, self.stt, self.tts, self.brain = transport, stt, tts, brain
        self.persona, self.rec, self.cfg = persona, recorder, turn_cfg
        self.lang = "hi" if lang.startswith("hi") else "en"
        self.rng = random.Random(seed)

        self._out: asyncio.Queue[_Say | _Sound | None] = asyncio.Queue()
        self._turn_id = 0
        self._turn_t0: dict[int, float] = {}
        self._first_audio_logged: set[int] = set()
        self._speaking_until = 0.0
        self._user_speaking = False
        self._pending_user: list[str] = []
        self._eot_task: asyncio.Task | None = None
        self._respond_task: asyncio.Task | None = None
        self._player_task: asyncio.Task | None = None
        self._synth_task: asyncio.Task | None = None
        self._last_activity = time.monotonic()
        self._reprompts = 0
        self._last_filler: str | None = None
        self._terminal: dict | None = None
        self.ended = asyncio.Event()
        self.end_reason = ""

    # ------------------------------------------------------------------ lifecycle
    async def start(self, opening: str | None = None) -> None:
        """opening: a fixed, compliance-approved first line (recording disclosure, identity check for outbound).
        Scripted openings are instant (can be pre-synthesised) and can't drift. None = let the LLM open."""
        await self.stt.start()
        await self.tts.start()
        self._player_task = asyncio.create_task(self._player())
        self._bg = [asyncio.create_task(self._stt_loop()), asyncio.create_task(self._watchdog())]
        if opening:
            self._turn_id += 1
            self.brain.history.append({"role": "assistant", "content": opening})
            await self._out.put(_Say(opening, self.lang, self._turn_id))
        else:
            self._start_response(None)

    async def on_frame(self, frame: InboundFrame) -> None:
        if frame.kind == "audio":
            await self.stt.send_audio(frame.audio)
        elif frame.kind == "dtmf" and frame.digit == "0":
            await self._finish_with({"type": "transfer", "reason": "caller_request"}, "dtmf_0")
        elif frame.kind == "stop":
            await self.close("caller_hangup")

    async def close(self, reason: str) -> None:
        if self.ended.is_set():
            return
        self.end_reason = self.end_reason or reason
        me = asyncio.current_task()
        for t in [self._eot_task, self._respond_task, self._synth_task, self._player_task, *getattr(self, "_bg", [])]:
            if t and t is not me and not t.done():
                t.cancel()
        try:
            await self.stt.close()
            await self.tts.close()
        except Exception:
            pass
        self.ended.set()

    # ------------------------------------------------------------------ helpers
    @property
    def agent_speaking(self) -> bool:
        return time.monotonic() < self._speaking_until or not self._out.empty() or (
            self._synth_task is not None and not self._synth_task.done())

    @staticmethod
    def _is_backchannel(text: str) -> bool:
        words = re.findall(r"[a-zA-Zऀ-ॿ'-]+", text.lower())
        return bool(words) and all(w in BACKCHANNELS for w in words) and len(words) <= 3

    def _update_lang(self, text: str) -> None:
        """Mirror the caller, with hysteresis: dates, numbers and one-word answers never flip the language."""
        words = [w for w in re.findall(r"[a-zA-Z\u0900-\u097F']+", text)
                 if w.lower() not in _MONTH_WORDS and w.lower() not in BACKCHANNELS]
        if len(words) >= 3:
            self.lang = detect_lang(text)

    # ------------------------------------------------------------------ input side
    async def _stt_loop(self) -> None:
        async for ev in self.stt.events():
            self._last_activity = time.monotonic()
            if ev.kind == "speech_start":
                self._user_speaking = True
                if self._eot_task and not self._eot_task.done():
                    self._eot_task.cancel()          # caller is continuing their sentence
            elif ev.kind == "partial":
                await self._maybe_barge_in(ev.text)
            elif ev.kind == "final":
                self._user_speaking = False
                if not ev.text.strip():
                    continue
                await self._maybe_barge_in(ev.text)
                self._pending_user.append(ev.text.strip())
                grace = self.cfg.incomplete_grace_ms if TRAILING.search(ev.text) else self.cfg.endpoint_grace_ms
                if self._eot_task and not self._eot_task.done():
                    self._eot_task.cancel()
                self._eot_task = asyncio.create_task(self._end_of_turn(grace / 1000))
            elif ev.kind == "speech_end":
                self._user_speaking = False
            elif ev.kind == "error":
                self.rec.r.errors.append(f"stt:{ev.text[:80]}")

    async def _maybe_barge_in(self, text: str) -> None:
        if not self.agent_speaking or self._is_backchannel(text):
            return
        if len([w for w in text.split() if w.lower() not in BACKCHANNELS]) < self.cfg.barge_in_min_words:
            return
        await self._barge_in(text)

    async def _barge_in(self, heard: str) -> None:
        logger.info(f"barge-in: {heard!r}")
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        if self._synth_task and not self._synth_task.done():
            self._synth_task.cancel()
        while not self._out.empty():
            self._out.get_nowait()
        self._speaking_until = time.monotonic()
        await self.transport.clear()
        self.rec.barge_in()
        self.brain.history.append({"role": "system", "content":
                                   "[The caller interrupted you mid-sentence. They may not have heard the rest of "
                                   "your last message. Respond to what they say now; repeat key facts only if needed.]"})

    async def _end_of_turn(self, grace_s: float) -> None:
        await asyncio.sleep(grace_s)
        if self._user_speaking:
            return
        text = " ".join(self._pending_user).strip()
        self._pending_user.clear()
        if not text:
            return
        if self._is_backchannel(text) and self.agent_speaking:
            return                                  # "haan" while we talk = listening, not a new turn
        self._reprompts = 0
        self._update_lang(text)
        self.rec.caller(text, self.lang)
        self._start_response(text)

    # ------------------------------------------------------------------ output side
    def _start_response(self, text: str | None) -> None:
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        self._turn_id += 1
        self._turn_t0[self._turn_id] = time.monotonic()
        self._respond_task = asyncio.create_task(self._respond(text, self._turn_id))

    async def _respond(self, text: str | None, turn_id: int) -> None:
        said_anything = False
        filler_timer = asyncio.create_task(self._filler_after(turn_id, self.cfg.filler_after_ms / 1000)) \
            if text is not None else None
        sound_timer: asyncio.Task | None = None
        try:
            async for ev in self.brain.respond(text, self.lang):
                if isinstance(ev, SentenceEvent):
                    if filler_timer:
                        filler_timer.cancel()
                    self.rec.guard(ev.guard)
                    said_anything = True
                    await self._out.put(_Say(ev.text, self.lang, turn_id))
                elif isinstance(ev, ToolStartEvent):
                    if filler_timer:
                        filler_timer.cancel()
                    if not said_anything and ev.name not in ("end_call", "transfer_to_human", "mark_wrong_person"):
                        await self._say_filler(turn_id)
                        said_anything = True
                    sound_timer = asyncio.create_task(self._sound_after(turn_id, self.cfg.tool_sound_after_ms / 1000))
                elif isinstance(ev, ToolEndEvent):
                    if sound_timer:
                        sound_timer.cancel()
                elif isinstance(ev, ActionEvent):
                    self.rec.r.actions.append(ev.action)
                    if ev.action["type"] in ("end_call", "transfer"):
                        self._terminal = ev.action
                elif isinstance(ev, ErrorEvent):
                    self.rec.r.errors.append(ev.message)
            if self._terminal:
                await self._drain_then_terminate()
        except asyncio.CancelledError:
            raise
        finally:
            for t in (filler_timer, sound_timer):
                if t:
                    t.cancel()

    async def _filler_after(self, turn_id: int, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._say_filler(turn_id)

    async def _say_filler(self, turn_id: int) -> None:
        f = self.persona.filler(self.lang, self.rng, avoid=self._last_filler)
        self._last_filler = f
        await self._out.put(_Say(f, self.lang, turn_id, kind="filler"))

    async def _sound_after(self, turn_id: int, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._out.put(_Sound(keyboard_clicks(self.transport.sample_rate, 1.2), turn_id))

    async def _player(self) -> None:
        """Single consumer: synthesise and send, in order. Cancellable per item for barge-in."""
        while True:
            item = await self._out.get()
            if item is None:
                return
            self._synth_task = asyncio.create_task(self._play(item))
            try:
                await self._synth_task
            except asyncio.CancelledError:
                if self.ended.is_set():
                    return
            except Exception as e:  # TTS outage: log and keep the call alive
                logger.error(f"tts failure: {e}")
                self.rec.r.errors.append(f"tts:{type(e).__name__}")

    async def _play(self, item: _Say | _Sound) -> None:
        if isinstance(item, _Sound):
            await self._send(item.pcm, item.turn_id)
            await self.transport.mark(f"s{item.turn_id}")
            return
        self.rec.agent(item.text, item.lang, kind=item.kind)
        async for pcm in self.tts.synthesize(verbalize(item.text, item.lang), item.lang):
            await self._send(pcm, item.turn_id)
        await self.transport.mark(f"t{item.turn_id}")

    async def _send(self, pcm: bytes, turn_id: int) -> None:
        if turn_id not in self._first_audio_logged and turn_id in self._turn_t0 and turn_id > 1:
            self._first_audio_logged.add(turn_id)
            self.rec.latency((time.monotonic() - self._turn_t0[turn_id]) * 1000)
        now = time.monotonic()
        self._speaking_until = max(self._speaking_until, now) + duration_s(pcm, self.transport.sample_rate)
        await self.transport.send_audio(pcm)

    async def _wait_until_quiet(self, extra: float = 0.3) -> None:
        while not self._out.empty() or (self._synth_task and not self._synth_task.done()):
            await asyncio.sleep(0.05)
        await asyncio.sleep(max(0.0, self._speaking_until - time.monotonic()) + extra)

    async def _drain_then_terminate(self) -> None:
        await self._wait_until_quiet()
        action = self._terminal
        if action["type"] == "transfer":
            self.end_reason = f"transfer:{action.get('reason')}"
            await self.transport.transfer(action.get("reason", ""))
        else:
            self.end_reason = f"agent_hangup:{action.get('outcome')}"
            await self.transport.hangup()
        await self.close(self.end_reason)

    async def _finish_with(self, action: dict, reason: str) -> None:
        self.rec.r.actions.append(action)
        self._terminal = action
        await self._drain_then_terminate()

    # ------------------------------------------------------------------ watchdog
    async def _watchdog(self) -> None:
        t_start = time.monotonic()
        while not self.ended.is_set():
            await asyncio.sleep(0.25)
            now = time.monotonic()
            if self._terminal:
                continue
            if now - t_start > self.cfg.max_call_s:
                await self._out.put(_Say(MAX_DURATION[self.lang], self.lang, self._turn_id, kind="reprompt"))
                await self._finish_with({"type": "transfer", "reason": "max_duration"}, "max_duration")
                return
            busy = (self._user_speaking or self.agent_speaking or self._pending_user
                    or (self._respond_task and not self._respond_task.done()))
            if busy:
                self._last_activity = now
                continue
            if now - self._last_activity > self.cfg.silence_reprompt_s:
                self._last_activity = now
                if self._reprompts >= self.cfg.silence_hangup_reprompts:
                    await self._out.put(_Say(GOODBYE_SILENT[self.lang], self.lang, self._turn_id, kind="reprompt"))
                    await self._finish_with({"type": "end_call", "outcome": "no_response"}, "silence")
                    return
                line = REPROMPT[self.lang][min(self._reprompts, len(REPROMPT[self.lang]) - 1)]
                self._reprompts += 1
                await self._out.put(_Say(line, self.lang, self._turn_id, kind="reprompt"))
