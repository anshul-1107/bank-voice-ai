"""Everything QA needs about one call, captured live. Saved at hangup; scored by the post-call worker.

Transcripts are stored with PII redacted (see redact()). Raw audio recording is the telephony provider's job
(Exotel / Plivo call recording) and is referenced by call_id.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from bank_voice_agent.qa.guardrails import GuardResult

_REDACTIONS = [
    (re.compile(r"\b\d{12}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b(\d{1,2})[/\-. ](\d{1,2})[/\-. ](19|20)\d{2}\b"), "[DOB]"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|"
                r"october|november|december)\s+(19|20)\d{2}\b", re.I), "[DOB]"),
    (re.compile(r"\b(otp|pin|cvv)\b\D{0,12}\d{3,8}", re.I), r"\1 [REDACTED]"),
]


def redact(text: str) -> str:
    for rx, rep in _REDACTIONS:
        text = rx.sub(rep, text)
    return text


@dataclass
class Turn:
    role: str                       # caller | agent | system
    text: str
    t: float                        # seconds since call start
    lang: str = "en"
    interrupted: bool = False
    kind: str = "speech"            # speech | filler | reprompt | sound


@dataclass
class CallRecord:
    call_id: str
    direction: str
    purpose: str
    persona: str
    prompt_version: str
    llm_model: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None
    customer_id: str | None = None
    turns: list[Turn] = field(default_factory=list)
    first_audio_latency_ms: list[float] = field(default_factory=list)
    barge_ins: int = 0
    reprompts: int = 0
    fillers: int = 0
    violations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    end_reason: str = ""
    outcome: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CallRecorder:
    def __init__(self, record: CallRecord):
        self.r = record
        self._t0 = time.monotonic()

    def now(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def caller(self, text: str, lang: str) -> None:
        self.r.turns.append(Turn("caller", redact(text), self.now(), lang))

    def agent(self, text: str, lang: str, kind: str = "speech") -> Turn:
        t = Turn("agent", redact(text), self.now(), lang, kind=kind)
        self.r.turns.append(t)
        if kind == "filler":
            self.r.fillers += 1
        if kind == "reprompt":
            self.r.reprompts += 1
        return t

    def guard(self, g: GuardResult) -> None:
        for v in g.violations:
            self.r.violations.append({"code": v.code, "severity": v.severity, "detail": redact(v.detail),
                                      "sentence": redact(g.original), "t": self.now(), "blocked": g.blocked})

    def latency(self, ms: float) -> None:
        self.r.first_audio_latency_ms.append(round(ms, 1))

    def barge_in(self) -> None:
        self.r.barge_ins += 1
        for t in reversed(self.r.turns):
            if t.role == "agent":
                t.interrupted = True
                break

    def finish(self, end_reason: str, tool_log: list, verification: dict, customer_id: str | None) -> CallRecord:
        self.r.ended_at = datetime.now(timezone.utc).isoformat()
        self.r.duration_s = self.now()
        self.r.end_reason = end_reason
        self.r.verification = verification
        self.r.customer_id = customer_id
        self.r.tool_calls = [{"name": e.name, "args": {k: redact(str(v)) for k, v in e.args.items()},
                              "ok": e.ok, "latency_ms": round(e.latency_ms, 1),
                              "blocked_unverified": e.blocked_unverified,
                              "result": redact(str(e.result))[:800]} for e in tool_log]
        ends = [a for a in self.r.actions if a.get("type") == "end_call"]
        if ends:
            self.r.outcome = ends[-1].get("outcome", "")
        elif any(a.get("type") == "transfer" for a in self.r.actions):
            self.r.outcome = "transferred"
        else:
            self.r.outcome = end_reason
        return self.r
