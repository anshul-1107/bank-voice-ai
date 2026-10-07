"""Banking tools exposed to the LLM, with the identity gate enforced in code.

Design rules
- The LLM never supplies money amounts to a write tool. Amounts are computed server-side from the account.
- Every account tool checks the gate first; an unverified call gets an instruction, never data.
- Every result is registered in the FactLedger so the grounding guardrail knows what may be spoken.
- Every call is logged (name, args, latency, ok) for post-call QA.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.agent.faq import FAQ, TOPICS
from bank_voice_agent.banking.core import CoreBanking
from bank_voice_agent.banking.identity import IdentityGate
from bank_voice_agent.banking.models import Loan, Policy
from bank_voice_agent.speech.normalize import inr

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "verify_identity",
        "description": "Verify the caller. Call once you have their date of birth (and last 4 digits of a loan or "
                       "policy number if the tool asked for it).",
        "parameters": {"type": "object", "properties": {
            "dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD"},
            "account_last4": {"type": "string", "description": "Last 4 digits of loan or policy number, if asked"}},
            "required": ["dob"]}}},
    {"type": "function", "function": {
        "name": "mark_wrong_person",
        "description": "The person on the line says they are not the customer. Share nothing further.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_loan_summary",
        "description": "Loans of the verified customer: EMI amount, next due date, overdue amount, outstanding, "
                       "EMIs remaining, autopay status.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_policy_summary",
        "description": "Insurance policies of the verified customer: product, premium, renewal date, status.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_foreclosure_quote",
        "description": "Amount needed to close a loan early, valid 7 days.",
        "parameters": {"type": "object", "properties": {
            "account_last4": {"type": "string"}}, "required": ["account_last4"]}}},
    {"type": "function", "function": {
        "name": "send_payment_link",
        "description": "SMS a secure payment link to the registered mobile. The amount is set by the system.",
        "parameters": {"type": "object", "properties": {
            "account_last4": {"type": "string", "description": "Last 4 of the loan or policy"},
            "purpose": {"type": "string", "enum": ["emi", "overdue", "foreclosure", "premium"]}},
            "required": ["account_last4", "purpose"]}}},
    {"type": "function", "function": {
        "name": "record_promise_to_pay",
        "description": "Record the date by which the customer says they will pay an overdue amount.",
        "parameters": {"type": "object", "properties": {
            "account_last4": {"type": "string"},
            "promised_date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["account_last4", "promised_date"]}}},
    {"type": "function", "function": {
        "name": "faq",
        "description": "Approved answer for a general (non-account) question. Topics: " + ", ".join(TOPICS),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": TOPICS}}, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Raise a service request or complaint for things you cannot do on the call.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": ["emi_date_change", "restructuring", "charge_dispute",
                                                    "complaint", "documents", "other"]},
            "summary": {"type": "string"},
            "priority": {"type": "string", "enum": ["normal", "high"]}},
            "required": ["category", "summary"]}}},
    {"type": "function", "function": {
        "name": "opt_out_of_reminder_calls",
        "description": "Customer asked not to receive reminder calls.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": "Warm-transfer to a human agent. Say one short line before calling this.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "enum": ["caller_request", "distress", "complaint", "complex_request",
                                                  "verification_locked", "agent_unable"]}},
            "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "end_call",
        "description": "End the call after saying goodbye.",
        "parameters": {"type": "object", "properties": {
            "outcome": {"type": "string", "enum": ["resolved", "payment_link_sent", "promise_to_pay", "ticket_raised",
                                                   "wrong_person", "callback_requested", "not_resolved"]}},
            "required": ["outcome"]}}},
]

ACCOUNT_TOOLS = {"get_loan_summary", "get_policy_summary", "get_foreclosure_quote", "send_payment_link",
                 "record_promise_to_pay", "create_ticket", "opt_out_of_reminder_calls"}


@dataclass
class ToolLogEntry:
    name: str
    args: dict
    result: dict
    ok: bool
    latency_ms: float
    blocked_unverified: bool = False


@dataclass
class ToolExecutor:
    core: CoreBanking
    gate: IdentityGate
    ledger: FactLedger
    today: date = field(default_factory=date.today)
    log: list[ToolLogEntry] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)       # transfer / end_call signals for the session

    def run(self, name: str, args: dict[str, Any]) -> dict:
        t0 = time.perf_counter()
        blocked = False
        try:
            if name in ACCOUNT_TOOLS and not self.gate.verified:
                blocked = True
                result = {"error": "not_verified",
                          "instruction": "Verify the caller with verify_identity before sharing or doing anything "
                                         "on the account. Ask for their date of birth."}
            else:
                fn = getattr(self, f"_t_{name}", None)
                if fn is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    # LLMs sometimes invent arguments (e.g. an "amount" for a payment link). Drop them: the
                    # system, not the model, decides amounts. Dropped args are kept in the log for QA.
                    allowed = set(inspect.signature(fn).parameters)
                    dropped = {k: v for k, v in args.items() if k not in allowed}
                    result = fn(**{k: v for k, v in args.items() if k in allowed})
                    if dropped:
                        result = {**result, "_ignored_args": sorted(dropped)}
            ok = "error" not in result
        except Exception as e:  # tools must never crash the call
            result = {"error": "system_unavailable",
                      "instruction": "Apologise, say the system is slow right now, and offer a payment link later, "
                                     "a ticket, or a human colleague.", "detail": type(e).__name__}
            ok = False
        if ok:
            self.ledger.add(result, public=(name == "faq"))
        self.log.append(ToolLogEntry(name=name, args=args, result=result, ok=ok,
                                     latency_ms=(time.perf_counter() - t0) * 1000, blocked_unverified=blocked))
        return result

    # ---------- helpers ----------
    def _customer_id(self) -> str:
        return self.gate.customer.customer_id

    def _find_loan(self, last4: str) -> Loan | None:
        return next((l for l in self.core.get_loans(self._customer_id()) if l.account_no.endswith(last4)), None)

    def _find_policy(self, last4: str) -> Policy | None:
        return next((p for p in self.core.get_policies(self._customer_id()) if p.policy_no.endswith(last4)), None)

    # ---------- tools ----------
    def _t_verify_identity(self, dob: str, account_last4: str | None = None) -> dict:
        try:
            d = date.fromisoformat(dob)
        except (ValueError, TypeError):
            return {"verified": False, "error": "bad_dob_format",
                    "instruction": "Ask them to repeat their date of birth, day month and year."}
        res = self.gate.verify(d, account_last4)
        if res.get("verified"):
            res = {**res, "next": "Thank them and continue with what they needed."}
        return res

    def _t_mark_wrong_person(self) -> dict:
        self.gate.wrong_person = True
        self.actions.append({"type": "end_call", "outcome": "wrong_person"})
        return {"ok": True, "instruction": "Apologise for the trouble, say nothing about any account, and end politely."}

    def _t_get_loan_summary(self) -> dict:
        loans = self.core.get_loans(self._customer_id())
        if not loans:
            return {"loans": [], "note": "No active loans."}
        out = []
        for l in loans:
            days = (l.next_due_date - self.today).days
            out.append({
                "loan": f"{l.product} ending {l.account_no[-4:]}",
                "emi_amount": inr(l.emi_amount),
                "next_due_date": l.next_due_date.isoformat(),
                "due_in_days": days,
                "overdue_amount": inr(l.overdue_amount),
                "late_fee": inr(l.late_fee),
                "status": l.status.value,
                "outstanding_principal": inr(l.outstanding_principal),
                "emis_remaining": l.emis_remaining,
                "autopay_active": l.autopay_mandate,
            })
        return {"loans": out}

    def _t_get_policy_summary(self) -> dict:
        pols = self.core.get_policies(self._customer_id())
        if not pols:
            return {"policies": [], "note": "No policies."}
        return {"policies": [{
            "policy": f"{p.product} with {p.insurer}, ending {p.policy_no[-4:]}",
            "premium": inr(p.premium),
            "sum_assured": inr(p.sum_assured),
            "renewal_date": p.renewal_date.isoformat(),
            "renews_in_days": (p.renewal_date - self.today).days,
            "status": p.status.value,
            "grace_period_ends": (p.renewal_date + timedelta(days=p.grace_days)).isoformat(),
        } for p in pols]}

    def _t_get_foreclosure_quote(self, account_last4: str) -> dict:
        loan = self._find_loan(account_last4)
        if not loan:
            return {"error": "loan_not_found", "instruction": "Ask which loan; read back the last 4 digits you have."}
        q = self.core.foreclosure_quote(loan.account_no, self.today)
        return {k: (inr(v) if isinstance(v, int) else v) for k, v in q.items()}

    def _t_send_payment_link(self, account_last4: str, purpose: str) -> dict:
        cid = self._customer_id()
        if purpose == "premium":
            pol = self._find_policy(account_last4)
            if not pol:
                return {"error": "policy_not_found"}
            link = self.core.send_payment_link(cid, pol.policy_no, pol.premium)
        else:
            loan = self._find_loan(account_last4)
            if not loan:
                return {"error": "loan_not_found"}
            if purpose == "foreclosure":
                amount = self.core.foreclosure_quote(loan.account_no, self.today)["total_payable"]
            elif purpose == "overdue":
                amount = loan.overdue_amount + loan.late_fee
                if amount == 0:
                    return {"error": "nothing_overdue", "instruction": "Tell them nothing is overdue on this loan."}
            else:
                amount = loan.emi_amount
            link = self.core.send_payment_link(cid, loan.account_no, amount)
        self.actions.append({"type": "payment_link_sent", "link_id": link.link_id})
        return {"sent": True, "amount": inr(link.amount), "sent_to": link.sent_to, "channel": link.channel,
                "link_valid_hours": 48}

    def _t_record_promise_to_pay(self, account_last4: str, promised_date: str) -> dict:
        loan = self._find_loan(account_last4)
        if not loan:
            return {"error": "loan_not_found"}
        d = date.fromisoformat(promised_date)
        if d < self.today or d > self.today + timedelta(days=30):
            return {"error": "date_out_of_range",
                    "instruction": "Promise dates must be within the next 30 days. Ask for a date in that range "
                                   "or offer a human colleague to discuss options."}
        return self.core.record_promise_to_pay(self._customer_id(), loan.account_no, d)

    def _t_faq(self, topic: str) -> dict:
        if topic not in FAQ:
            return {"error": "unknown_topic", "instruction": "Say you will have a colleague call back on this."}
        return {"approved_answer": FAQ[topic]}

    def _t_create_ticket(self, category: str, summary: str, priority: str = "normal") -> dict:
        t = self.core.create_ticket(self._customer_id(), category, summary, priority)
        self.actions.append({"type": "ticket", "ticket_id": t.ticket_id})
        return {"ticket_id": t.ticket_id, "resolution_time": "within 3 working days",
                "instruction": "Read the ticket number slowly, digit by digit."}

    def _t_opt_out_of_reminder_calls(self) -> dict:
        self.core.record_opt_out(self._customer_id())
        return {"ok": True, "note": "Reminder calls stopped. SMS and email reminders continue."}

    def _t_transfer_to_human(self, reason: str) -> dict:
        self.actions.append({"type": "transfer", "reason": reason})
        return {"ok": True, "instruction": "Transfer starting. Do not say anything further."}

    def _t_end_call(self, outcome: str) -> dict:
        self.actions.append({"type": "end_call", "outcome": outcome})
        return {"ok": True}
