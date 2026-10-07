"""Run a full phone call offline: real CallSession, real guardrails, real QA — simulated caller, mock audio.

    python -m bank_voice_agent.evals.simulate_call --scenario emi_hinglish_happy
    python -m bank_voice_agent.evals.simulate_call --scenario barge_in --llm real     # uses LLM__ keys from .env
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import dataclass, field

from loguru import logger

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.models import CallContext, CallPurpose, Language
from bank_voice_agent.config import Settings
from bank_voice_agent.pipeline.factory import build_call, finalize_call
from bank_voice_agent.qa.scorer import format_transcript
from bank_voice_agent.qa.store import QAStore
from bank_voice_agent.speech.mock import MockSTT, MockTTS
from bank_voice_agent.telephony.base import InboundFrame


@dataclass
class SimTransport:
    sample_rate: int = 8000
    bytes_out: int = 0
    clears: int = 0
    events: list[str] = field(default_factory=list)
    hung_up: bool = False
    transferred: str | None = None

    async def send_audio(self, pcm16: bytes) -> None:
        self.bytes_out += len(pcm16)

    async def mark(self, name: str) -> None:
        self.events.append(f"mark:{name}")

    async def clear(self) -> None:
        self.clears += 1
        self.events.append("clear")

    async def hangup(self) -> None:
        self.hung_up = True

    async def transfer(self, reason: str) -> None:
        self.transferred = reason


# Each step: {"say": "..."} waits until the agent is quiet; {"say": ..., "interrupt_after_s": 0.3} barges in.
SCENARIOS: dict[str, dict] = {
    "anshul_call": {
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919636274630", "lang": "hi-IN"},
        "steps": [{"say": "Haan ji mera EMI kab hai aur kitna hai"}, {"say": "11 July 2004"},
                  {"say": "haan ji link bhej do please"}, {"say": "bas itna hi shukriya"}]},
    "emi_hinglish_happy": {
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919800000001", "lang": "hi-IN"},
        "steps": [{"say": "Haan ji mera EMI kab hai aur kitna hai"}, {"say": "14 May 1990"},
                  {"say": "haan ji link bhej do please"}, {"say": "bas itna hi shukriya"}]},
    "barge_in": {
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919800000001", "lang": "en-IN"},
        "steps": [{"say": "I want to know my loan EMI"}, {"say": "14 May 1990"},
                  {"say": "wait wait just send me the payment link", "interrupt_after_s": 0.15},
                  {"say": "thanks that's all"}]},
    "wrong_dob_locked": {
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919800000001", "lang": "en-IN"},
        "steps": [{"say": "what is my loan EMI amount"}, {"say": "1 January 1991"}, {"say": "2 January 1991"},
                  {"say": "3 January 1991"}, {"say": "ok connect me to a human then"}]},
    "outbound_emi_reminder": {
        "ctx": {"direction": "outbound", "purpose": "emi_due_reminder", "customer_id": "C0001",
                "reference_id": "PL20234821", "lang": "hi-IN"},
        "steps": [{"say": "haan ji main Rahul bol raha hoon, kya hua"}, {"say": "14 May 1990"},
                  {"say": "achha link bhej dijiye"}, {"say": "bas shukriya"}]},
    "hallucination_caught": {
        "hallucinate": True,
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919800000001", "lang": "en-IN"},
        "steps": [{"say": "what's my loan EMI please"}, {"say": "14 May 1990"}, {"say": "ok thanks bye"}]},
    "silent_caller": {
        "ctx": {"direction": "inbound", "purpose": "inbound", "caller_number": "+919800000002", "lang": "en-IN"},
        "steps": []},
}


async def _wait_quiet(session, settle: float = 0.35) -> None:
    quiet_since = None
    while not session.ended.is_set():
        busy = session.agent_speaking or (session._respond_task and not session._respond_task.done()) \
            or session._pending_user
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > settle:
            return
        await asyncio.sleep(0.03)


async def run_scenario(name: str, settings: Settings, llm=None, store: QAStore | None = None,
                       verbose: bool = True) -> dict:
    sc = SCENARIOS[name]
    c = sc["ctx"]
    core = MockCoreBanking(n_customers=50)
    ctx = CallContext(call_id=f"sim-{name}-{int(time.time() * 1000) % 100000}", direction=c["direction"],
                      purpose=CallPurpose(c["purpose"]), caller_number=c.get("caller_number", ""),
                      customer_id=c.get("customer_id"), reference_id=c.get("reference_id"),
                      language_hint=Language(c.get("lang", "en-IN")))
    if llm is None and settings.llm_provider == "mock":
        from bank_voice_agent.agent.llm import RuleBasedDemoLLM
        llm = RuleBasedDemoLLM(hallucinate=sc.get("hallucinate", False))
    transport = SimTransport()
    bundle = build_call(ctx, transport, core, settings, llm=llm, stt=MockSTT(),
                        tts=MockTTS(words_per_s=25.0), seed=1)
    s = bundle.session
    await s.start(bundle.opening)
    for step in sc["steps"]:
        if "interrupt_after_s" in step:
            await asyncio.sleep(0.05)
            while not s.agent_speaking and not s.ended.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(step["interrupt_after_s"])
        else:
            await _wait_quiet(s)
        if s.ended.is_set():
            break
        await s.on_frame(InboundFrame("audio", audio=b"TXT:" + step["say"].encode()))
    try:
        await asyncio.wait_for(s.ended.wait(), timeout=40)
    except asyncio.TimeoutError:
        await s.close("sim_timeout")
    store = store or QAStore(tempfile.mktemp(suffix=".sqlite3"))
    out = await finalize_call(bundle, store, settings)
    out["transport"] = {"hung_up": transport.hung_up, "transferred": transport.transferred,
                        "clears": transport.clears, "audio_seconds": round(transport.bytes_out / 2 / 8000, 1)}
    if verbose:
        print(f"\n=== {name} ===")
        print(format_transcript(out["call"]))
        qa = out["qa"]
        print(f"\nQA verdict: {qa['verdict']}  score={qa['score']}  judge={qa['judge']}")
        for r in qa["reasons"]:
            print(f"  - {r}")
        m = qa["metrics"]
        print(f"  latency p50/p95: {m['latency_p50_ms']}/{m['latency_p95_ms']} ms, barge-ins={m['barge_ins']}, "
              f"fillers={m['fillers']}, outcome={m['outcome']}, transport={out['transport']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--llm", default="mock", choices=["mock", "real"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    logger.remove()
    logger.add(lambda m: None)
    settings = Settings(stt_provider="mock", tts_provider="mock",
                        llm_provider="mock" if a.llm == "mock" else "openai_compatible")
    settings.turn.silence_reprompt_s = 1.5
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    results = []
    for n in names:
        results.append(asyncio.run(run_scenario(n, settings, verbose=not a.json)))
    if a.json:
        print(json.dumps([{"scenario": n, "verdict": r["qa"]["verdict"], "score": r["qa"]["score"]}
                          for n, r in zip(names, results)], indent=2))


if __name__ == "__main__":
    main()
