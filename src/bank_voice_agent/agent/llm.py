"""LLM clients. One streaming interface; OpenAI-compatible implementation (Groq, OpenAI, Sarvam-M, vLLM...)
plus a rule-based demo model so the whole pipeline, guardrails and QA can run offline without any API key."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import AsyncIterator, Protocol

from bank_voice_agent.config import LLMSettings


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallReq:
    id: str
    name: str
    args: dict


@dataclass
class ToolCalls:
    calls: list[ToolCallReq] = field(default_factory=list)


LLMEvent = TextDelta | ToolCalls


class LLM(Protocol):
    def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]: ...


class OpenAICompatibleLLM:
    def __init__(self, cfg: LLMSettings):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "missing", timeout=cfg.timeout_s,
                                  max_retries=1)

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        # internal "name" on tool messages is for our own bookkeeping; strict providers reject it
        wire = [{k: v for k, v in m.items() if not (m["role"] == "tool" and k == "name")} for m in messages]
        resp = await self.client.chat.completions.create(
            model=self.cfg.model, messages=wire, tools=tools, tool_choice="auto", stream=True,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
        )
        pending: dict[int, dict] = {}
        async for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield TextDelta(delta.content)
            for tc in delta.tool_calls or []:
                slot = pending.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if pending:
            calls = []
            for slot in pending.values():
                try:
                    args = json.loads(slot["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCallReq(id=slot["id"] or f"call_{len(calls)}", name=slot["name"], args=args))
            yield ToolCalls(calls)


# ----------------------------------------------------------------------------------------------------------------
# Offline demo model. NOT for production: it exists so `make simulate` and the test-suite exercise the real
# pipeline, guardrails and QA end to end with zero API keys. Behaviour is deliberately simple and predictable.
# ----------------------------------------------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                       "dec"], start=1)}


def _parse_dob(text: str) -> str | None:
    t = text.lower()
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?[\s/\-.]+(\d{1,2}|[a-z]{3,9})[\s/\-.,]+(\d{4})", t)
    if not m:
        return None
    d, mo, y = m.groups()
    month = int(mo) if mo.isdigit() else _MONTHS.get(mo[:3])
    try:
        return date(int(y), month, int(d)).isoformat() if month else None
    except ValueError:
        return None


class RuleBasedDemoLLM:
    def __init__(self, today: date | None = None, hallucinate: bool = False):
        self.today = today or date.today()
        self.hallucinate = hallucinate      # used by tests to prove the grounding guardrail blocks bad numbers

    @staticmethod
    def _last_tool(messages: list[dict]) -> tuple[str | None, dict]:
        for m in reversed(messages):
            if m["role"] == "tool":
                return m.get("name"), json.loads(m["content"])
            if m["role"] == "user":
                break
        return None, {}

    @staticmethod
    def _verified(messages: list[dict]) -> bool:
        return any(m["role"] == "tool" and m.get("name") == "verify_identity" and '"verified": true' in m["content"]
                   for m in messages)

    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[LLMEvent]:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        u = user.lower()
        from bank_voice_agent.speech.normalize import detect_lang
        hi = detect_lang(user) == "hi" or (len(user.split()) < 3 and any(
            detect_lang(m["content"] or "") == "hi" for m in messages if m["role"] == "user"))
        tool, res = self._last_tool(messages)
        verified = self._verified(messages)

        async def say(text: str):
            for word in re.findall(r"\S+\s*", text):
                yield TextDelta(word)

        def call(name: str, **args) -> ToolCalls:
            return ToolCalls([ToolCallReq(id=f"c_{name}", name=name, args=args)])

        # --- react to a tool result first ---
        if tool == "verify_identity":
            if res.get("verified"):
                yield call("get_loan_summary") if "policy" not in u and "insurance" not in u else call(
                    "get_policy_summary")
            elif res.get("locked"):
                async for e in say("I'm sorry, I can't share account details on this call. Shall I connect you "
                                   "to a colleague?"):
                    yield e
            else:
                async for e in say("Sorry, that didn't match. Could you tell me your date of birth once more?"):
                    yield e
            return
        if tool == "get_loan_summary" and res.get("loans"):
            l = res["loans"][0]
            amount = "₹99,999" if self.hallucinate else l["emi_amount"]
            due = date.fromisoformat(l["next_due_date"])
            when = f"{due.day} {due.strftime('%B')}"
            if l["status"] == "overdue":
                text = (f"Ji, aapke {l['loan']} par {l['overdue_amount']} overdue hai. Kya main payment link bhej doon?"
                        if hi else f"Your {l['loan']} has {l['overdue_amount']} overdue. Shall I send a payment link?")
            else:
                text = (f"Ji, aapka EMI {amount} hai, {when} ko due hai. Payment link bhej doon?"
                        if hi else f"Your EMI of {amount} is due on {when}. Would you like a payment link?")
            async for e in say(text):
                yield e
            return
        if tool == "get_policy_summary" and res.get("policies"):
            p = res["policies"][0]
            d = date.fromisoformat(p["renewal_date"])
            async for e in say(f"Your {p['policy']} renews on {d.day} {d.strftime('%B')}, premium {p['premium']}. "
                               f"Shall I send a payment link?"):
                yield e
            return
        if tool == "send_payment_link":
            async for e in say(f"Done, the link for {res.get('amount')} is sent to your number ending "
                               f"{res.get('sent_to', '')[-4:]}. Anything else?" if res.get("sent") else
                               "Sorry, I couldn't send the link right now."):
                yield e
            return
        if tool in ("mark_wrong_person",):
            async for e in say("Koi baat nahi, maafi chahti hoon. Dhanyavaad!" if hi else
                               "No problem, sorry to disturb you. Thank you!"):
                yield e
            yield call("end_call", outcome="wrong_person")
            return
        if tool == "opt_out_of_reminder_calls":
            async for e in say("Ji, reminder calls band kar di gayi hain. SMS reminders aate rahenge." if hi else
                               "Done, reminder calls are stopped. You'll still get SMS reminders."):
                yield e
            return
        if tool == "get_foreclosure_quote" and "total_payable" in res:
            async for e in say(f"To close the loan you'd pay {res['total_payable']}, valid till "
                               f"{date.fromisoformat(res['valid_till']).day} "
                               f"{date.fromisoformat(res['valid_till']).strftime('%B')}. Shall I send a link?"):
                yield e
            return
        if tool == "transfer_to_human":
            return
        if tool == "end_call":
            return
        if tool == "faq":
            async for e in say(res.get("approved_answer", "Let me have a colleague call you back on that.")):
                yield e
            return

        # --- react to the caller ---
        if re.search(r"\b(otp|pin|cvv)\b", u):
            async for e in say("Please don't share that with anyone, including us. We never ask for it."):
                yield e
            return
        if re.search(r"(robot|bot\b|real person or|machine|ai hai)", u):
            async for e in say("I'm Neural Finance's AI assistant. I can help with most things, or connect you to a "
                               "colleague anytime."):
                yield e
            return
        if re.search(r"(naukri chali|job loss|lost my job|hospital|bimaar|death|expired|passed away)", u):
            async for e in say("Main samajh sakti hoon, yeh mushkil waqt hai. Main aapko ek colleague se connect "
                               "karti hoon jo options samjha sakte hain." if hi else
                               "I'm really sorry, that sounds very hard. Let me connect you to a colleague who can "
                               "talk through options."):
                yield e
            yield call("transfer_to_human", reason="distress")
            return
        if re.search(r"(human|agent|insaan|manager|real person)", u):
            async for e in say("Sure, connecting you to a colleague now."):
                yield e
            yield call("transfer_to_human", reason="caller_request")
            return
        if re.search(r"(not .*(him|her|me)|wrong number|galat number|main .* nahi)", u):
            yield call("mark_wrong_person")
            return
        if re.search(r"\b(stop|band karo|don't call|mat karo)\b", u):
            yield call("opt_out_of_reminder_calls")
            return
        dob = _parse_dob(user)
        if dob and not verified:
            yield call("verify_identity", dob=dob)
            return
        if verified and re.search(r"(close .*early|foreclos|band karna|poora chuka)", u):
            yield call("get_foreclosure_quote", account_last4=self._first_suffix(messages))
            return
        if re.search(r"(bounce|late fee|charges)", u):
            yield call("faq", topic="emi_bounce_charges")
            return
        if re.search(r"(link|bhej|send)", u) and verified:
            yield call("send_payment_link", account_last4=self._first_suffix(messages), purpose="emi")
            return
        if re.search(r"(emi|loan|policy|insurance|due|premium|kitna|balance)", u):
            if verified:
                yield call("get_policy_summary" if re.search("policy|insurance|premium", u) else "get_loan_summary")
            else:
                async for e in say("Ji, zaroor. Verification ke liye apni date of birth bataiye?" if hi else
                                   "Sure. For security, may I have your date of birth?"):
                    yield e
            return
        if re.search(r"(thank|bye|nothing|bas|that's all|shukriya|dhanyavaad)", u):
            async for e in say("Thank you for calling. Have a good day!" if not hi else "Dhanyavaad ji, aapka din "
                                                                                         "shubh ho!"):
                yield e
            yield call("end_call", outcome="resolved")
            return
        if not u.strip():
            async for e in say("Namaste! I'm Priya from Neural Finance. How can I help you today?"):
                yield e
            return
        if not verified and re.search(r"\b(haan|yes|speaking|bol raha|bol rahi|ji)\b", u):
            async for e in say("Dhanyavaad ji. Verification ke liye apni date of birth bataiye?" if hi else
                               "Thank you. For security, may I have your date of birth?"):
                yield e
            return
        async for e in say("Sorry, could you say that once more?"):
            yield e

    @staticmethod
    def _first_suffix(messages: list[dict]) -> str:
        for m in messages:
            if m["role"] == "tool" and m.get("name") == "get_loan_summary":
                loans = json.loads(m["content"]).get("loans", [])
                if loans:
                    return loans[0]["loan"][-4:]
        return "0000"
