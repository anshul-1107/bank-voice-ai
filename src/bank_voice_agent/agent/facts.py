"""Fact ledger: every number and date the agent is ALLOWED to say in this call.

Filled from (a) tool results, (b) the caller's own words, (c) approved FAQ text, (d) today's date.
The grounding guardrail rejects any spoken amount/date/account-suffix that is not in the ledger.
This is the single most important QA control for a banking voice agent: it makes hallucinated amounts impossible
to reach the caller's ear, instead of merely unlikely.
"""

from __future__ import annotations

import re
from datetime import date

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def _to_number(tok: str) -> float | None:
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


class FactLedger:
    def __init__(self, today: date | None = None):
        self.numbers: set[float] = set()
        self.dates: set[tuple[int, int]] = set()      # (day, month) — year is rarely spoken
        self.suffixes: set[str] = set()
        self.public_numbers: set[float] = set()        # from approved FAQ text: OK to say before verification
        self.account_numbers: set[float] = set()       # from account tools: only after verification
        today = today or date.today()
        self.add_date(today)

    # ---------- ingest ----------
    def add_date(self, d: date) -> None:
        self.dates.add((d.day, d.month))
        self.numbers.add(float(d.day))
        self.numbers.add(float(d.year))

    def add_text(self, text: str) -> None:
        for tok in _NUM.findall(text):
            n = _to_number(tok)
            if n is not None:
                self.numbers.add(n)
                digits = tok.replace(",", "").split(".")[0]
                if len(digits) >= 4:
                    self.suffixes.add(digits[-4:])
        low = text.lower()
        for m_idx, m in enumerate(MONTHS, start=1):
            for mt in re.finditer(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{m}", low):
                self.dates.add((int(mt.group(1)), m_idx))

    def add(self, obj, public: bool = False) -> None:
        """Walk any tool result and register its facts. public=True for approved FAQ content."""
        before = set(self.numbers)
        self._walk(obj)
        new = self.numbers - before
        (self.public_numbers if public else self.account_numbers).update(new)

    def _walk(self, obj) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                self._walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                self._walk(v)
        elif isinstance(obj, bool) or obj is None:
            return
        elif isinstance(obj, (int, float)):
            self.numbers.add(float(obj))
        elif isinstance(obj, date):
            self.add_date(obj)
        elif isinstance(obj, str):
            if _ISO.match(obj):
                self.add_date(date.fromisoformat(obj))
            else:
                self.add_text(obj)
                # "XXXX4821" style masks
                m = re.search(r"(\d{4})$", obj)
                if m:
                    self.suffixes.add(m.group(1))

    # ---------- check ----------
    def has_number(self, n: float) -> bool:
        return n in self.numbers

    def has_date(self, day: int, month: int) -> bool:
        return (day, month) in self.dates

    def has_suffix(self, s: str) -> bool:
        return s in self.suffixes
