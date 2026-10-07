"""Layer 2 QA — in-call guardrails. Runs on EVERY sentence before it is spoken. Deterministic, < 1 ms.

A sentence that fails a CRITICAL or MAJOR check is never spoken; a safe line is said instead, the model is told what
went wrong so it self-corrects on the next turn, and the violation is stored for post-call QA.

Severity
  critical  → blocks the sentence AND auto-fails the call in post-call QA (e.g. a made-up amount, asking for OTP)
  major     → blocks the sentence; call goes to human review
  minor     → spoken (after cleanup); counted in quality metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bank_voice_agent.agent.facts import MONTHS, FactLedger


@dataclass
class Violation:
    code: str
    severity: str          # critical | major | minor
    detail: str


@dataclass
class GuardResult:
    ok: bool
    text: str                                   # what will actually be spoken
    original: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.ok


SAFE_LINES = {
    "en": "Just a moment, let me confirm that detail before I say it.",
    "hi": "Ek second, main yeh detail confirm kar leti hoon.",
}

_RUPEE = re.compile(r"(?:₹|Rs\.?\s?|INR\s?)(\d[\d,]*(?:\.\d{1,2})?)", re.I)
_BIGNUM = re.compile(r"(?<![\w₹])(\d{1,3}(?:,\d{2,3})+|\d{3,})(?![\d%])")
_DATE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\b", re.I)
_SUFFIX = re.compile(r"(?:ending(?:\s+in|\s+with)?|last\s+(?:4|four)\s+digits\s+(?:are\s+)?|XXXX)\s*(\d{4})\b", re.I)
_LONG_DIGITS = re.compile(r"\d{8,}|[A-Z]{2}\d{8}")

_SENSITIVE = r"(otp|o\.?t\.?p|one[- ]time password|pin|cvv|password|card number|net ?banking|upi pin|aadhaar otp)"
_ASKING = re.compile(
    rf"(share|tell|give|provide|send|read out|confirm|enter|bata(?:o|iye|yein|dijiye)?|bhej(?:iye|o)?|boliye)"
    rf"[^.?!]{{0,30}}\b{_SENSITIVE}\b|\b{_SENSITIVE}\b[^.?!]{{0,25}}(bata(?:o|iye|yein|dijiye)?|chahiye|share kar|bhej)",
    re.I)
_NEGATED = re.compile(r"\b(never|don'?t|do not|not|mat|nahi|nahin|kabhi nahi)\b", re.I)

_COERCION = re.compile(
    r"\b(legal action|police|court case|jail|arrest|fir\b|blacklist|defaulter list|your (family|employer|office|"
    r"relatives|neighbou?rs)|ghar (aa|pe aa)|office (aa|pe aa)|recovery agent (will|aayega)|seize|confiscate|"
    r"warrant|parivaar ko|office mein bata)", re.I)
_GUARANTEE = re.compile(r"\b(guarantee[ds]?|100 ?%|hundred percent|definitely approved?|pakka approve|"
                        r"you should invest|best investment|tax saving tip|returns? of \d)", re.I)
_WAIVER = re.compile(r"\b(waive[ds]?|waiver|maaf kar|no late fee|charges? (removed|reversed)|"
                     r"penalty (removed|cancelled))\b", re.I)
_PRETEND_HUMAN = re.compile(r"\b(i am|i'm|main) (a |ek )?(real )?(human|person|insaan|not a (bot|robot|machine|ai))\b"
                            r"|\bi am not (an? )?(ai|bot|robot)\b|\bmain (bot|robot|ai) nahi\b", re.I)
_MARKDOWN = re.compile(r"[*_#`>|]|^\s*[-•]\s", re.M)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def check_sentence(text: str, ledger: FactLedger, verified: bool, lang: str = "en") -> GuardResult:
    v: list[Violation] = []

    # 1. Grounding — every amount / date / account suffix must be a known fact of THIS call
    for m in _RUPEE.finditer(text):
        n = _num(m.group(1))
        if not ledger.has_number(n):
            v.append(Violation("ungrounded_amount", "critical", f"₹{m.group(1)} not from any tool or caller"))
        elif not verified and n in ledger.account_numbers and n not in ledger.public_numbers:
            v.append(Violation("unverified_disclosure", "critical", f"₹{m.group(1)} said before verification"))
    rupee_spans = [m.span() for m in _RUPEE.finditer(text)]
    for m in _BIGNUM.finditer(text):
        if any(a <= m.start() < b for a, b in rupee_spans):
            continue
        tok = m.group(1)
        if len(tok.replace(",", "")) == 4 and _SUFFIX.search(text[max(0, m.start() - 30): m.end()]):
            continue   # handled by suffix check
        if not ledger.has_number(_num(tok)):
            v.append(Violation("ungrounded_number", "major", f"{tok} not from any tool or caller"))
    for m in _DATE.finditer(text):
        day, month = int(m.group(1)), MONTHS.index(m.group(2).lower()) + 1
        if not ledger.has_date(day, month):
            v.append(Violation("ungrounded_date", "critical", f"{m.group(0)} not from any tool or caller"))
    for m in _SUFFIX.finditer(text):
        if not ledger.has_suffix(m.group(1)):
            v.append(Violation("ungrounded_account", "major", f"account ending {m.group(1)} unknown"))
        elif not verified:
            v.append(Violation("unverified_disclosure", "major", f"account ending {m.group(1)} before verification"))
    if _LONG_DIGITS.search(text):
        v.append(Violation("full_account_number", "critical", "8+ digit identifier spoken"))

    # 2. Safety and compliance
    for m in _ASKING.finditer(text):
        window = text[max(0, m.start() - 25): m.end()]
        if not _NEGATED.search(window):
            v.append(Violation("asked_sensitive_credential", "critical", m.group(0)))
    if _COERCION.search(text):
        v.append(Violation("coercive_language", "critical", _COERCION.search(text).group(0)))
    if _WAIVER.search(text):
        v.append(Violation("unauthorised_promise", "major", _WAIVER.search(text).group(0)))
    if _GUARANTEE.search(text):
        v.append(Violation("guarantee_or_advice", "major", _GUARANTEE.search(text).group(0)))
    if _PRETEND_HUMAN.search(text):
        v.append(Violation("claims_to_be_human", "critical", _PRETEND_HUMAN.search(text).group(0)))

    # 3. Style (spoken, but measured)
    clean = text
    if _MARKDOWN.search(text):
        v.append(Violation("markdown", "minor", "formatting characters"))
        clean = _MARKDOWN.sub("", text)
    if len(text.split()) > 45:
        v.append(Violation("too_long", "minor", f"{len(text.split())} words in one sentence"))

    blocking = [x for x in v if x.severity in ("critical", "major")]
    if blocking:
        return GuardResult(ok=False, text=SAFE_LINES.get(lang, SAFE_LINES["en"]), original=text, violations=v)
    return GuardResult(ok=True, text=clean.strip(), original=text, violations=v)


def correction_note(result: GuardResult) -> str:
    """System message fed back to the LLM so it fixes itself on the next turn."""
    reasons = "; ".join(f"{x.code}: {x.detail}" for x in result.violations if x.severity != "minor")
    return (f"[QA guard] Your sentence was NOT spoken because: {reasons}. The caller heard: \"{result.text}\". "
            "Only state amounts, dates and account digits that appear in tool results of this call, only after "
            "verification, and follow the compliance rules. Continue naturally.")
