"""Layer 1 QA — run the eval suite and gate releases.

    bva-evals                      # real LLM from .env (LLM__API_KEY ...)
    bva-evals --llm mock           # offline demo model (shows the harness; most model-quality checks will fail)
    bva-evals --repeat 3           # run each case 3x: voice agents are stochastic, gate on the worst run

Exit code 1 if any critical case fails or the non-critical pass rate is below --min-pass (default 0.9).
Writes evals/report.json for CI artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from bank_voice_agent.agent.brain import ActionEvent, Brain, SentenceEvent, ToolEndEvent, ToolStartEvent
from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.agent.llm import RuleBasedDemoLLM
from bank_voice_agent.agent.persona import Persona, build_system_prompt, opening_line, prompt_version
from bank_voice_agent.agent.tools import ToolExecutor
from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.identity import IdentityGate
from bank_voice_agent.banking.models import CallContext, CallPurpose
from bank_voice_agent.config import Settings
from bank_voice_agent.pipeline.factory import make_llm
from bank_voice_agent.speech.normalize import detect_lang

SCENARIOS = Path(__file__).parent / "scenarios.yaml"


@dataclass
class CaseResult:
    id: str
    critical: bool
    passed: bool
    failures: list[str] = field(default_factory=list)
    heard: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    blocked: int = 0
    seconds: float = 0.0


async def run_case(case: dict, settings: Settings, llm=None) -> CaseResult:
    t0 = time.perf_counter()
    core = MockCoreBanking(n_customers=20)
    persona = Persona.load(settings.persona)
    purpose = CallPurpose(case.get("purpose", "inbound"))
    caller = case.get("caller", "unknown")
    cand = core.find_customer_by_phone(caller) if caller != "unknown" else None
    direction = "inbound" if purpose == CallPurpose.INBOUND else "outbound"
    ref = next((l.account_no for l in core.get_loans(cand.customer_id)), None) if cand else None
    ctx = CallContext(call_id=f"eval-{case['id']}", direction=direction, purpose=purpose, caller_number=caller,
                      customer_id=cand.customer_id if cand and direction == "outbound" else None, reference_id=ref)
    gate, ledger = IdentityGate(core, cand), FactLedger()
    if cand:
        ledger.add_text(cand.first_name)
    ex = ToolExecutor(core, gate, ledger)
    brain = Brain(llm or make_llm(settings), ex, ledger,
                  build_system_prompt(persona, ctx, settings.bank_name, settings.grievance_officer,
                                      cand.first_name if cand else None,
                                      ani_matched=bool(cand) and direction == "inbound"))
    lang = "hi" if any(detect_lang(t) == "hi" for t in case["turns"][:1]) else "en"
    opening = opening_line(persona, direction, lang, settings.bank_name,
                           cand.first_name if cand and direction == "outbound" else None)
    brain.history.append({"role": "assistant", "content": opening})

    res = CaseResult(id=case["id"], critical=bool(case.get("critical")), passed=True)
    verified_at_turn: int | None = None
    heard_by_turn: list[list[str]] = []
    actions: list[dict] = []
    for i, turn in enumerate(case["turns"]):
        if any(a["type"] in ("end_call", "transfer") for a in actions):
            break
        lang = detect_lang(turn) if len(turn.split()) >= 3 else lang
        heard: list[str] = []
        async for ev in brain.respond(turn, lang):
            if isinstance(ev, SentenceEvent):
                heard.append(ev.text)
                if ev.guard.blocked:
                    res.blocked += 1
            elif isinstance(ev, ToolStartEvent):
                res.tools.append(ev.name)
            elif isinstance(ev, ToolEndEvent) and ev.name == "verify_identity" and ev.result.get("verified"):
                verified_at_turn = i if verified_at_turn is None else verified_at_turn
            elif isinstance(ev, ActionEvent):
                actions.append(ev.action)
        heard_by_turn.append(heard)
    res.heard = [s for h in heard_by_turn for s in h]
    all_heard = " ".join(res.heard)

    def fail(msg: str) -> None:
        res.passed = False
        res.failures.append(msg)

    for t in case.get("expect_tools", []):
        if t not in res.tools:
            fail(f"expected tool {t}")
    if case.get("expect_any_tools") and not set(case["expect_any_tools"]) & set(res.tools):
        fail(f"expected one of {case['expect_any_tools']}")
    for t in case.get("forbid_tools", []):
        if t in res.tools and not any(x.blocked_unverified for x in ex.log if x.name == t):
            fail(f"forbidden tool {t} executed")
    for rx in case.get("must_say", []):
        if not re.search(rx, all_heard):
            fail(f"never said /{rx}/")
    for rx in case.get("must_not_say", []):
        m = re.search(rx, all_heard)
        if m:
            fail(f"said forbidden /{rx}/: …{all_heard[max(0, m.start() - 30): m.end() + 10]}…")
    if case.get("no_amount_before_verified"):
        upto = len(heard_by_turn) if verified_at_turn is None else verified_at_turn
        pre = " ".join(s for h in heard_by_turn[:upto] for s in h)
        if "₹" in pre:
            fail("amount spoken before verification")
    if case.get("must_say_lang"):
        langs = [detect_lang(s) for s in res.heard if len(s.split()) >= 3]
        if langs and langs.count(case["must_say_lang"]) < len(langs) / 2:
            fail(f"replied mostly in {max(set(langs), key=langs.count)}, expected {case['must_say_lang']}")
    if "max_blocked" in case and res.blocked > case["max_blocked"]:
        fail(f"guardrail blocked {res.blocked} sentence(s) (model produced unsafe/ungrounded text)")
    end = case.get("end")
    if end and end != "none" and not any(a["type"] == end for a in actions):
        fail(f"expected terminal action {end}")
    res.seconds = round(time.perf_counter() - t0, 2)
    return res


async def run_all(settings: Settings, ids: list[str] | None, repeat: int, use_mock: bool) -> list[CaseResult]:
    cases = yaml.safe_load(SCENARIOS.read_text())
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    out: list[CaseResult] = []
    for c in cases:
        runs = [await run_case(c, settings, RuleBasedDemoLLM() if use_mock else None) for _ in range(repeat)]
        worst = next((r for r in runs if not r.passed), runs[0])   # gate on the worst run
        out.append(worst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["real", "mock"], default="real")
    ap.add_argument("--case", action="append")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--min-pass", type=float, default=0.9)
    ap.add_argument("--out", default="evals_report.json")
    a = ap.parse_args()
    logger.remove()
    settings = Settings()
    results = asyncio.run(run_all(settings, a.case, a.repeat, a.llm == "mock"))

    crit = [r for r in results if r.critical]
    rest = [r for r in results if not r.critical]
    crit_fail = [r for r in crit if not r.passed]
    rest_rate = sum(r.passed for r in rest) / len(rest) if rest else 1.0
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {'[critical] ' if r.critical else ''}{r.id}  ({r.seconds}s, tools={r.tools}, "
              f"blocked={r.blocked})")
        for f in r.failures:
            print(f"        - {f}")
    ok = not crit_fail and rest_rate >= a.min_pass
    print(f"\ncritical: {len(crit) - len(crit_fail)}/{len(crit)} passed | other: {rest_rate:.0%} "
          f"(min {a.min_pass:.0%}) | prompt {prompt_version()} | model "
          f"{'demo-rule-based' if a.llm == 'mock' else settings.llm.model}")
    print("RELEASE GATE:", "PASS" if ok else "FAIL")
    Path(a.out).write_text(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
