"""Identity gate. Deterministic code, not the LLM, decides whether a caller is verified.

Rules (conservative defaults for a regulated lender):
- Never ask for OTP, PIN, CVV, password, full card or full account number. Ever.
- Factor 1: something we already matched (caller number on file, or the outbound dialled number) OR last 4 digits
  of a loan / policy number.
- Factor 2: date of birth.
- 3 failed attempts → LOCKED for the rest of the call. No account data, offer branch / human.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from bank_voice_agent.banking.core import CoreBanking
from bank_voice_agent.banking.models import Customer


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    LOCKED = "locked"


class IdentityGate:
    MAX_ATTEMPTS = 3

    def __init__(self, core: CoreBanking, candidate: Customer | None = None):
        self.core = core
        self.candidate = candidate          # matched by phone number; NOT yet verified
        self.customer: Customer | None = None
        self.state = VerificationState.UNVERIFIED
        self.attempts = 0
        self.wrong_person = False           # outbound: the person who answered said they are not the customer

    @property
    def verified(self) -> bool:
        return self.state == VerificationState.VERIFIED

    def verify(self, dob: date | None, account_last4: str | None = None) -> dict:
        if self.state == VerificationState.LOCKED:
            return {"verified": False, "locked": True,
                    "instruction": "Verification is locked for this call. Do not share any account details. "
                                   "Offer to connect to a human agent or visit the branch."}
        if self.state == VerificationState.VERIFIED:
            return {"verified": True, "customer_first_name": self.customer.first_name}

        if dob is None:
            return {"verified": False, "error": "need_dob",
                    "instruction": "Ask the caller for their date of birth."}

        pool: list[Customer] = []
        if account_last4 and len(account_last4) == 4 and account_last4.isdigit():
            pool = self.core.find_customer_by_account_suffix(account_last4)
            if self.candidate:
                pool = [c for c in pool if c.customer_id == self.candidate.customer_id] or pool
        elif self.candidate:
            pool = [self.candidate]
        else:
            return {"verified": False, "error": "need_account_last4",
                    "instruction": "This number is not registered. Ask for the last four digits of their loan "
                                   "or policy number, and their date of birth."}

        self.attempts += 1
        match = next((c for c in pool if c.dob == dob), None)
        if match:
            self.customer = match
            self.state = VerificationState.VERIFIED
            return {"verified": True, "customer_first_name": match.first_name}

        remaining = self.MAX_ATTEMPTS - self.attempts
        if remaining <= 0:
            self.state = VerificationState.LOCKED
            return {"verified": False, "locked": True,
                    "instruction": "Details did not match three times. Politely say you cannot share account "
                                   "details on this call, and offer a human agent or branch visit. "
                                   "Do not say which detail was wrong."}
        return {"verified": False, "attempts_remaining": remaining,
                "instruction": "Details did not match. Politely ask them to repeat their date of birth. "
                               "Do not say which detail was wrong."}

    def snapshot(self) -> dict:
        return {"state": self.state.value, "attempts": self.attempts,
                "customer_id": self.customer.customer_id if self.customer else None,
                "candidate_id": self.candidate.customer_id if self.candidate else None,
                "wrong_person": self.wrong_person}
