"""Builds a fully isolated CallSession per call. Nothing conversational is shared between calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from bank_voice_agent.agent.brain import Brain
from bank_voice_agent.agent.facts import FactLedger
from bank_voice_agent.agent.llm import LLM, OpenAICompatibleLLM, RuleBasedDemoLLM
from bank_voice_agent.agent.persona import Persona, build_system_prompt, opening_line, prompt_version
from bank_voice_agent.agent.tools import ToolExecutor
from bank_voice_agent.banking.core import CoreBanking
from bank_voice_agent.banking.identity import IdentityGate
from bank_voice_agent.banking.models import CallContext, CallPurpose, Language
from bank_voice_agent.config import Settings
from bank_voice_agent.pipeline.session import CallSession
from bank_voice_agent.qa.record import CallRecord, CallRecorder
from bank_voice_agent.qa.scorer import HeuristicJudge, LLMJudge, Scorer
from bank_voice_agent.qa.store import QAStore
from bank_voice_agent.speech.base import StreamingSTT, StreamingTTS
from bank_voice_agent.telephony.base import Transport


@dataclass
class CallBundle:
    session: CallSession
    recorder: CallRecorder
    executor: ToolExecutor
    gate: IdentityGate
    opening: str = ""


def make_llm(s: Settings) -> LLM:
    return RuleBasedDemoLLM() if s.llm_provider == "mock" else OpenAICompatibleLLM(s.llm)


def make_stt(s: Settings) -> StreamingSTT:
    if s.stt_provider == "mock":
        from bank_voice_agent.speech.mock import MockSTT
        return MockSTT()
    from bank_voice_agent.speech.sarvam import SarvamStreamingSTT
    return SarvamStreamingSTT(s.sarvam, sample_rate=s.telephony.sample_rate)


def make_tts(s: Settings, speaker: str) -> StreamingTTS:
    if s.tts_provider == "mock":
        from bank_voice_agent.speech.mock import MockTTS
        return MockTTS(s.telephony.sample_rate)
    from bank_voice_agent.speech.sarvam import SarvamStreamingTTS
    return SarvamStreamingTTS(s.sarvam, out_rate=s.telephony.sample_rate, speaker=speaker)


def build_call(ctx: CallContext, transport: Transport, core: CoreBanking, s: Settings, *,
               llm: LLM | None = None, stt: StreamingSTT | None = None, tts: StreamingTTS | None = None,
               seed: int | None = None) -> CallBundle:
    persona = Persona.load(s.persona)

    # Who might this be? Outbound: the customer we dialled. Inbound: ANI lookup. Either way: NOT verified yet.
    candidate = core.get_customer(ctx.customer_id) if ctx.customer_id else core.find_customer_by_phone(
        ctx.caller_number)
    if candidate and ctx.direction == "inbound":
        ctx.language_hint = candidate.preferred_language
    gate = IdentityGate(core, candidate)
    ledger = FactLedger()
    if candidate:
        ledger.add_text(candidate.first_name)
    executor = ToolExecutor(core=core, gate=gate, ledger=ledger)

    system_prompt = build_system_prompt(persona, ctx, s.bank_name, s.grievance_officer,
                                        first_name=candidate.first_name if candidate else None,
                                        ani_matched=candidate is not None and ctx.direction == "inbound")
    brain = Brain(llm=llm or make_llm(s), tools=executor, ledger=ledger, system_prompt=system_prompt)
    recorder = CallRecorder(CallRecord(call_id=ctx.call_id, direction=ctx.direction, purpose=ctx.purpose.value,
                                       persona=persona.name, prompt_version=prompt_version(),
                                       llm_model=s.llm.model if s.llm_provider != "mock" else "demo-rule-based"))
    session = CallSession(transport, stt or make_stt(s), tts or make_tts(s, persona.tts_speaker), brain, persona,
                          recorder, s.turn, lang="hi" if ctx.language_hint == Language.HI else "en", seed=seed)
    opening = opening_line(persona, ctx.direction, session.lang, s.bank_name,
                           candidate.first_name if candidate and ctx.direction == "outbound" else None)
    return CallBundle(session, recorder, executor, gate, opening)


async def finalize_call(bundle: CallBundle, store: QAStore, s: Settings, score_now: bool = True) -> dict:
    """Persist the call and run post-call QA. In prod, score in a worker (queue) instead of inline."""
    rec = bundle.recorder.finish(bundle.session.end_reason or "unknown", bundle.executor.log,
                                 bundle.gate.snapshot(),
                                 bundle.gate.customer.customer_id if bundle.gate.customer else None).to_dict()
    store.save_call(rec)
    if not score_now:
        return {"call": rec}
    judge = LLMJudge(s.judge) if s.judge.api_key else HeuristicJudge()
    qa = await asyncio.to_thread(Scorer(s.qa, judge).score, rec)
    store.save_qa(rec["call_id"], qa)
    logger.info(f"QA {rec['call_id']}: {qa['verdict']} score={qa['score']} reasons={qa['reasons'][:3]}")
    return {"call": rec, "qa": qa}


def context_from_start(call_id: str, from_number: str, to_number: str, custom: dict) -> CallContext:
    """Outbound campaign calls carry their purpose in the stream's custom parameters."""
    purpose = CallPurpose(custom.get("purpose", "inbound")) if custom.get("purpose") else CallPurpose.INBOUND
    outbound = purpose != CallPurpose.INBOUND
    return CallContext(call_id=call_id, direction="outbound" if outbound else "inbound", purpose=purpose,
                       caller_number=to_number if outbound else from_number,
                       customer_id=custom.get("customer_id"), reference_id=custom.get("reference_id"),
                       language_hint=Language(custom.get("lang", "en-IN")))
