"""The agent's turn loop: LLM stream → sentences → guardrail → events, with tool calls in between.

Used identically by the live phone session and by the offline evaluator, so what we test is what we ship.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from loguru import logger

from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.agent.llm import LLM, TextDelta, ToolCalls
from bank_voice_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor
from bank_voice_agent.qa.guardrails import GuardResult, check_sentence, correction_note


@dataclass
class SentenceEvent:
    text: str                 # safe, to be spoken
    guard: GuardResult


@dataclass
class ToolStartEvent:
    name: str
    args: dict


@dataclass
class ToolEndEvent:
    name: str
    result: dict
    latency_ms: float


@dataclass
class ActionEvent:
    action: dict              # {"type": "end_call" | "transfer" | ...}


@dataclass
class ErrorEvent:
    message: str


BrainEvent = SentenceEvent | ToolStartEvent | ToolEndEvent | ActionEvent | ErrorEvent

# Sentence boundary: . ? ! । followed by space/end, not after common abbreviations.
_BOUNDARY = re.compile(r"(?<!\bRs)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bNo)[.?!।](?=\s|$)")
_SOFT_BOUNDARY = re.compile(r"[,;:—](?=\s)")

LLM_ERROR_LINE = {
    "en": "Sorry, my system is a little slow right now. I can connect you to a colleague, or call you back. "
          "What would you prefer?",
    "hi": "Sorry ji, system thoda slow chal raha hai. Main aapko colleague se connect kar doon, ya call back karwa "
          "doon?",
}


@dataclass
class Brain:
    llm: LLM
    tools: ToolExecutor
    ledger: FactLedger
    system_prompt: str
    max_tool_rounds: int = 4
    llm_event_timeout_s: float = 12.0
    history: list[dict] = field(default_factory=list)
    max_history_messages: int = 40

    def _messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.history[-self.max_history_messages:]

    async def _timed(self, it: AsyncIterator) -> AsyncIterator:
        """Per-event timeout so a stuck provider never leaves the caller in silence."""
        while True:
            try:
                ev = await asyncio.wait_for(it.__anext__(), timeout=self.llm_event_timeout_s)
            except StopAsyncIteration:
                return
            yield ev

    def _guard(self, sentence: str, lang: str) -> SentenceEvent | None:
        sentence = sentence.strip()
        if not sentence:
            return None
        g = check_sentence(sentence, self.ledger, self.tools.gate.verified, lang)
        if g.blocked:
            logger.warning(f"guard blocked: {[v.code for v in g.violations]} :: {sentence!r}")
        return SentenceEvent(text=g.text, guard=g)

    async def respond(self, user_text: str | None, lang: str = "en") -> AsyncIterator[BrainEvent]:
        """One caller turn. user_text=None means 'agent speaks first' (call opening)."""
        if user_text is not None:
            self.ledger.add_text(user_text)               # caller's own numbers may be repeated back
            self.history.append({"role": "user", "content": user_text})

        for _round in range(self.max_tool_rounds + 1):
            buf, spoken, first = "", [], True
            tool_calls: ToolCalls | None = None
            notes: list[str] = []
            try:
                async for ev in self._timed(self.llm.stream(self._messages(), TOOL_SCHEMAS)):
                    if isinstance(ev, TextDelta):
                        buf += ev.text
                        while True:
                            m = _BOUNDARY.search(buf)
                            if not m and first and len(buf.split()) >= 8:
                                # ship the first clause early (lower latency), but never a stub like "Done,"
                                m = next((s for s in _SOFT_BOUNDARY.finditer(buf)
                                          if len(buf[: s.end()].split()) >= 4), None)
                            if not m:
                                break
                            sentence, buf = buf[: m.end()], buf[m.end():]
                            se = self._guard(sentence, lang)
                            if se:
                                first = False
                                spoken.append(se.text)
                                if se.guard.blocked:
                                    notes.append(correction_note(se.guard))
                                yield se
                    elif isinstance(ev, ToolCalls):
                        tool_calls = ev
            except (asyncio.TimeoutError, Exception) as e:  # provider down, timeout, bad response
                logger.error(f"LLM failure: {type(e).__name__}: {e}")
                yield ErrorEvent(f"llm:{type(e).__name__}")
                line = LLM_ERROR_LINE.get(lang, LLM_ERROR_LINE["en"])
                self.history.append({"role": "assistant", "content": line})
                yield SentenceEvent(text=line, guard=GuardResult(ok=True, text=line, original=line))
                return

            se = self._guard(buf, lang)
            if se:
                spoken.append(se.text)
                if se.guard.blocked:
                    notes.append(correction_note(se.guard))
                yield se

            msg: dict = {"role": "assistant", "content": " ".join(spoken) or None}
            if tool_calls and tool_calls.calls:
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                     for c in tool_calls.calls]
            if msg["content"] or msg.get("tool_calls"):
                self.history.append(msg)
            for n in notes:
                self.history.append({"role": "system", "content": n})

            if not (tool_calls and tool_calls.calls):
                return

            terminal = False
            for c in tool_calls.calls:
                yield ToolStartEvent(c.name, c.args)
                t0 = time.perf_counter()
                result = await asyncio.to_thread(self.tools.run, c.name, c.args)
                yield ToolEndEvent(c.name, result, (time.perf_counter() - t0) * 1000)
                self.history.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                     "content": json.dumps(result, ensure_ascii=False, default=str)})
                if c.name in ("transfer_to_human", "end_call"):
                    terminal = True
            while self.tools.actions:
                yield ActionEvent(self.tools.actions.pop(0))
            if terminal:
                return
        # Too many tool rounds: something is looping. Hand over rather than spin.
        yield ErrorEvent("tool_loop")
        yield ActionEvent({"type": "transfer", "reason": "agent_unable"})
