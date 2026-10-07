import json
from datetime import date

import pytest

from bank_voice_agent.agent.brain import Brain, SentenceEvent, ToolStartEvent
from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.agent.llm import RuleBasedDemoLLM, TextDelta, ToolCallReq, ToolCalls
from bank_voice_agent.agent.tools import ToolExecutor
from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.identity import IdentityGate, VerificationState

TODAY = date(2026, 10, 7)


def setup(phone="+919800000001"):
    core = MockCoreBanking(n_customers=10, today=TODAY)
    gate = IdentityGate(core, core.find_customer_by_phone(phone))
    led = FactLedger(today=TODAY)
    return core, gate, led, ToolExecutor(core, gate, led, today=TODAY)


def test_account_tools_refuse_until_verified():
    core, gate, led, ex = setup()
    r = ex.run("get_loan_summary", {})
    assert r["error"] == "not_verified" and ex.log[-1].blocked_unverified
    assert ex.run("verify_identity", {"dob": "1990-05-14"})["verified"]
    r = ex.run("get_loan_summary", {})
    assert r["loans"][0]["emi_amount"] == "₹12,450"
    assert led.has_number(12450.0)


def test_three_wrong_dobs_lock_the_call():
    core, gate, led, ex = setup()
    for d in ("1991-01-01", "1991-01-02", "1991-01-03"):
        ex.run("verify_identity", {"dob": d})
    assert gate.state == VerificationState.LOCKED
    assert not ex.run("verify_identity", {"dob": "1990-05-14"})["verified"]   # even the right one now


def test_unregistered_caller_needs_last4():
    core, gate, led, ex = setup(phone="+911111111111")
    assert ex.run("verify_identity", {"dob": "1990-05-14"})["error"] == "need_account_last4"
    assert ex.run("verify_identity", {"dob": "1990-05-14", "account_last4": "4821"})["verified"]


def test_payment_link_amount_is_set_by_system_not_llm():
    core, gate, led, ex = setup()
    ex.run("verify_identity", {"dob": "1990-05-14"})
    r = ex.run("send_payment_link", {"account_last4": "4821", "purpose": "emi", "amount": 1})
    assert r["amount"] == "₹12,450"


def test_promise_to_pay_window():
    core, gate, led, ex = setup("+919800000003")
    ex.run("verify_identity", {"dob": "1986-01-23"})
    assert ex.run("record_promise_to_pay", {"account_last4": "0077", "promised_date": "2026-12-31"})["error"] \
        == "date_out_of_range"
    assert "promised_date" in ex.run("record_promise_to_pay", {"account_last4": "0077",
                                                              "promised_date": "2026-10-12"})


def test_tool_exception_never_crashes_call():
    core, gate, led, ex = setup()
    ex.run("verify_identity", {"dob": "1990-05-14"})
    core.get_loans = lambda cid: (_ for _ in ()).throw(TimeoutError())
    r = ex.run("get_loan_summary", {})
    assert r["error"] == "system_unavailable"


class ScriptedLLM:
    """Returns pre-baked LLM events per round: lets us test the brain exactly."""
    def __init__(self, rounds):
        self.rounds = list(rounds)

    async def stream(self, messages, tools):
        for ev in self.rounds.pop(0):
            yield ev


async def collect(brain, text, lang="en"):
    return [e async for e in brain.respond(text, lang)]


async def test_brain_blocks_hallucinated_amount_and_tells_model():
    core, gate, led, ex = setup()
    gate.verify(date(1990, 5, 14))
    llm = ScriptedLLM([[TextDelta("Your EMI is ₹13,000 this month. "), TextDelta("Anything else?")]])
    brain = Brain(llm, ex, led, "sys")
    evs = await collect(brain, "what's my emi")
    spoken = [e.text for e in evs if isinstance(e, SentenceEvent)]
    assert "₹13,000" not in " ".join(spoken)
    assert any(m["role"] == "system" and "QA guard" in m["content"] for m in brain.history)


async def test_brain_runs_tools_then_answers():
    core, gate, led, ex = setup()
    llm = ScriptedLLM([
        [ToolCalls([ToolCallReq("1", "verify_identity", {"dob": "1990-05-14"})])],
        [ToolCalls([ToolCallReq("2", "get_loan_summary", {})])],
        [TextDelta("Your EMI of ₹12,450 is due on 10 October.")],
    ])
    brain = Brain(llm, ex, led, "sys")
    evs = await collect(brain, "14 May 1990")
    assert [e.name for e in evs if isinstance(e, ToolStartEvent)] == ["verify_identity", "get_loan_summary"]
    spoken = [e for e in evs if isinstance(e, SentenceEvent)]
    assert spoken[-1].guard.ok and "12,450" in spoken[-1].text
    tool_msgs = [m for m in brain.history if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["verified"] is True


async def test_llm_outage_gives_graceful_line():
    core, gate, led, ex = setup()

    class Broken:
        async def stream(self, m, t):
            raise ConnectionError("provider down")
            yield  # pragma: no cover

    evs = await collect(Brain(Broken(), ex, led, "sys"), "hello")
    assert any(isinstance(e, SentenceEvent) and "colleague" in e.text for e in evs)
