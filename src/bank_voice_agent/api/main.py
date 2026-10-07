"""FastAPI app: telephony WebSockets, Plivo XML, QA review endpoints, and Executive Operations Dashboard.

Run:  uvicorn bank_voice_agent.api.main:app --host 0.0.0.0 --port 8000
Each WebSocket = one call = one isolated CallSession.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import uuid
import wave
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.models import (
    CallContext,
    CallPurpose,
    Customer,
    Language,
    Loan,
    LoanStatus,
    Policy,
    PolicyStatus,
)
from bank_voice_agent.campaigns.dialer import Dialer, DryRunProvider, ExotelProvider
from bank_voice_agent.campaigns.planner import CallJob, plan_day
from bank_voice_agent.config import settings
from bank_voice_agent.evals.simulate_call import SCENARIOS, run_scenario
from bank_voice_agent.pipeline.factory import build_call, context_from_start, finalize_call
from bank_voice_agent.qa.report import build_report
from bank_voice_agent.qa.store import QAStore
from bank_voice_agent.speech.sarvam import SarvamStreamingTTS
from bank_voice_agent.telephony.exotel import ExotelAdapter, ExotelTransport
from bank_voice_agent.telephony.plivo import PlivoAdapter, PlivoTransport

app = FastAPI(title="Bank Voice Agent")
core = MockCoreBanking()          # in-memory core banking with Anshul, Ditya, Ramu, and 500 customers
store = QAStore(settings.qa.db_path)
ACTIVE: dict[str, object] = {}
WEB_CALLS: dict[str, dict] = {}
_background: set[asyncio.Task] = set()

# Mount Static Assets
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def pcm16_to_wav_b64(pcm_bytes: bytes, sample_rate: int = 8000, channels: int = 1) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def synthesize_text(text: str, lang: str = "hi") -> str | None:
    if settings.sarvam.api_key and settings.tts_provider != "mock":
        try:
            tts = SarvamStreamingTTS(settings.sarvam, out_rate=8000, speaker=settings.sarvam.tts_speaker)
            await tts.start()
            chunks = []
            async for chunk in tts.synthesize(text, lang):
                chunks.append(chunk)
            await tts.close()
            if chunks:
                return pcm16_to_wav_b64(b"".join(chunks), sample_rate=8000)
        except Exception as e:
            logger.warning(f"Sarvam TTS synthesis error: {e}")
    return None


@app.get("/")
async def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(404, "Dashboard HTML not found")
    return FileResponse(index_file)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "active_calls": len(ACTIVE), "active_web_calls": len(WEB_CALLS)}


# ----------------------------------------------------------------------------- Telephony WebSockets
async def _run_call(ws: WebSocket, adapter, transport) -> None:
    await ws.accept()
    logger.info("Telephony WebSocket connected and accepted")
    bundle = None
    try:
        while True:
            msg = await ws.receive_text()
            frame = adapter.parse(msg)
            if frame is None:
                continue
            if frame.kind != "audio":
                logger.info(f"Telephony event: kind={frame.kind} data={msg[:120]}")
            if frame.kind == "start" and bundle is None:
                transport.stream_sid, transport.call_sid = frame.stream_id, frame.call_id
                logger.info(f"Call start frame received: call_sid={frame.call_id}, stream_sid={frame.stream_id}")
                ctx = context_from_start(frame.call_id or frame.stream_id, frame.from_number, frame.to_number,
                                         {**dict(ws.query_params), **frame.custom})
                bundle = build_call(ctx, transport, core, settings)
                ACTIVE[ctx.call_id] = bundle
                await bundle.session.start(bundle.opening)
                logger.info(f"call {ctx.call_id} started ({ctx.direction}/{ctx.purpose.value}) opening={bundle.opening!r}")
            elif bundle is not None:
                await bundle.session.on_frame(frame)
                if frame.kind == "stop":
                    logger.info("Received stop event from telephony provider")
                    break
            if bundle is not None and bundle.session.ended.is_set():
                logger.info(f"Session ended flag set: reason={bundle.session.end_reason}")
                break
    except WebSocketDisconnect as e:
        logger.info(f"Telephony WebSocket disconnected: code={e.code}")
    except Exception as e:
        logger.exception(f"Unhandled exception in telephony call session: {e}")
    finally:
        if bundle is not None:
            await bundle.session.close("caller_hangup")
            ACTIVE.pop(bundle.recorder.r.call_id, None)
            t = asyncio.create_task(finalize_call(bundle, store, settings))
            _background.add(t)
            t.add_done_callback(_background.discard)


@app.websocket("/telephony/exotel/ws")
async def exotel_ws(ws: WebSocket) -> None:
    await _run_call(ws, ExotelAdapter(), ExotelTransport(ws, settings.telephony))


@app.websocket("/telephony/plivo/ws")
async def plivo_ws(ws: WebSocket) -> None:
    await _run_call(ws, PlivoAdapter(), PlivoTransport(ws, settings.telephony))


@app.api_route("/telephony/plivo/answer", methods=["GET", "POST"])
async def plivo_answer(request: Request) -> Response:
    q = dict(request.query_params)
    extra = ",".join(f"{k}={v}" for k, v in q.items() if k in ("purpose", "customer_id", "reference_id", "lang"))
    host = settings.telephony.public_base_url.replace("https://", "wss://")
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Stream bidirectional="true" keepCallAlive="true" '
           f'contentType="audio/x-l16;rate={settings.telephony.sample_rate}" extraHeaders="{extra}">'
           f'{host}/telephony/plivo/ws</Stream></Response>')
    return Response(xml, media_type="application/xml")


@app.api_route("/telephony/plivo/transfer", methods=["GET", "POST"])
async def plivo_transfer(reason: str = "caller_request") -> Response:
    xml = ('<?xml version="1.0" encoding="UTF-8"?><Response><Speak>Connecting you to our team.</Speak>'
           '<Dial><Number>HUMAN_QUEUE_NUMBER</Number></Dial></Response>')
    return Response(xml, media_type="application/xml")


@app.post("/campaigns/status")
async def campaign_status(request: Request) -> dict:
    try:
        if "form" in request.headers.get("content-type", ""):
            form = dict(await request.form())
        else:
            form = await request.json()
        logger.info(f"call status callback: {form}")
    except Exception as e:
        logger.warning(f"error parsing campaign status: {e}")
    return {"ok": True}


# ----------------------------------------------------------------------------- Dashboard Customer Management
class NewCustomerPayload(BaseModel):
    name: str
    phone: str
    dob: str
    preferred_language: str = "hi-IN"
    gender: str = "M"
    loan_product: str | None = None
    emi_amount: int | None = None


@app.get("/api/customers")
async def list_customers(query: str | None = None) -> list[dict]:
    custs = core.all_customers()
    results = []
    for c in custs:
        if query:
            q = query.lower()
            if q not in c.name.lower() and q not in c.phone and q not in c.customer_id.lower():
                continue
        loans = core.get_loans(c.customer_id)
        pols = core.get_policies(c.customer_id)
        results.append({
            "customer_id": c.customer_id,
            "name": c.name,
            "first_name": c.first_name,
            "phone": c.phone,
            "dob": str(c.dob),
            "preferred_language": c.preferred_language.value,
            "gender": c.gender,
            "do_not_call": c.do_not_call,
            "loans": [l.model_dump() for l in loans],
            "policies": [p.model_dump() for p in pols],
        })
    return results[:80]


@app.post("/api/customers")
async def add_customer(payload: NewCustomerPayload) -> dict:
    cid = f"C{len(core.customers) + 10:04d}"
    fn = payload.name.split()[0]
    cust = Customer(
        customer_id=cid,
        name=payload.name,
        first_name=fn,
        phone=payload.phone,
        dob=date.fromisoformat(payload.dob),
        preferred_language=Language(payload.preferred_language),
        gender=payload.gender
    )
    core.customers[cid] = cust

    if payload.loan_product and payload.emi_amount:
        acct = f"LN{cid}99"
        core.loans[acct] = Loan(
            account_no=acct,
            customer_id=cid,
            product=payload.loan_product,
            sanctioned_amount=payload.emi_amount * 36,
            outstanding_principal=payload.emi_amount * 24,
            emi_amount=payload.emi_amount,
            interest_rate_pa=12.5,
            next_due_date=date.today() + timedelta(days=3),
            emis_remaining=24
        )
    return {"ok": True, "customer_id": cid}


# ----------------------------------------------------------------------------- Interactive Web Call Session
class WebCallStart(BaseModel):
    phone: str
    direction: str = "inbound"


class WebCallTurn(BaseModel):
    call_id: str
    text: str


class WebCallEnd(BaseModel):
    call_id: str


@app.post("/api/calls/web/start")
async def web_call_start(payload: WebCallStart) -> dict:
    call_id = f"web-{uuid.uuid4().hex[:8]}"
    ctx = CallContext(call_id=call_id, caller_number=payload.phone, direction=payload.direction)
    from bank_voice_agent.pipeline.factory import build_call
    from bank_voice_agent.telephony.base import Transport

    class WebTransport(Transport):
        async def send_audio(self, pcm16: bytes) -> None: pass
        async def mark(self, name: str) -> None: pass
        async def clear(self) -> None: pass
        async def hangup(self) -> None: pass
        async def transfer(self, reason: str) -> None: pass

    bundle = build_call(ctx, WebTransport(), core, settings)
    WEB_CALLS[call_id] = {
        "bundle": bundle,
        "started_at": time.time(),
        "turns": []
    }

    opening = bundle.opening
    WEB_CALLS[call_id]["turns"].append({"role": "agent", "text": opening, "time_s": 0.0})
    audio_wav = await synthesize_text(opening, "hi")

    return {
        "call_id": call_id,
        "opening": opening,
        "audio_b64": audio_wav
    }


@app.post("/api/calls/web/turn")
async def web_call_turn(payload: WebCallTurn) -> dict:
    item = WEB_CALLS.get(payload.call_id)
    if not item:
        raise HTTPException(404, "Web call session not found")

    bundle = item["bundle"]
    brain = bundle.session.brain
    t_now = round(time.time() - item["started_at"], 1)
    item["turns"].append({"role": "caller", "text": payload.text, "time_s": t_now})

    # Execute Brain respond
    from bank_voice_agent.agent.brain import ActionEvent, ErrorEvent, SentenceEvent, ToolEndEvent, ToolStartEvent

    spoken_sentences = []
    tool_events = []
    violations = []
    ended = False

    async for ev in brain.respond(payload.text, lang=bundle.session.lang):
        if isinstance(ev, SentenceEvent):
            spoken_sentences.append(ev.text)
            bundle.recorder.guard(ev.guard)
            if not ev.guard.ok:
                violations.extend([v.code for v in ev.guard.violations])
        elif isinstance(ev, ToolStartEvent):
            tool_events.append({"name": ev.name, "args": ev.args})
        elif isinstance(ev, ToolEndEvent):
            if tool_events:
                tool_events[-1]["result"] = ev.result
        elif isinstance(ev, ActionEvent):
            if ev.action.get("type") in ("end_call", "transfer"):
                ended = True

    response_text = " ".join(spoken_sentences)
    item["turns"].append({"role": "agent", "text": response_text, "time_s": round(time.time() - item["started_at"], 1)})

    # Audio synthesis
    audio_b64 = await synthesize_text(response_text, bundle.session.lang)

    return {
        "response": response_text,
        "audio_b64": audio_b64,
        "verification_state": bundle.gate.state.value,
        "strikes": bundle.gate.attempts,
        "violations": violations,
        "tool_calls": tool_events,
        "ended": ended
    }


@app.post("/api/calls/web/end")
async def web_call_end(payload: WebCallEnd) -> dict:
    item = WEB_CALLS.pop(payload.call_id, None)
    if not item:
        raise HTTPException(404, "Web call not found")

    bundle = item["bundle"]
    out = await finalize_call(bundle, store, settings)
    return out


# ----------------------------------------------------------------------------- Scripted Scenario Runner
class ScenarioPayload(BaseModel):
    scenario: str


@app.post("/api/calls/simulate-scenario")
async def api_simulate_scenario(payload: ScenarioPayload) -> dict:
    if payload.scenario not in SCENARIOS:
        raise HTTPException(400, f"Invalid scenario. Available: {list(SCENARIOS)}")
    out = await run_scenario(payload.scenario, settings, store=store, verbose=False)
    return out


# ----------------------------------------------------------------------------- Outbound Campaigns
@app.get("/api/campaign/preview")
async def campaign_preview() -> dict:
    day = date.today()
    jobs, summary = plan_day(core, day, settings.campaign)
    return summary


@app.post("/api/campaign/rehearse")
async def campaign_rehearse() -> dict:
    day = date.today()
    jobs, summary = plan_day(core, day, settings.campaign)
    noon = datetime.combine(day, datetime.min.time(), ZoneInfo("Asia/Kolkata")).replace(hour=11)
    camp_cfg = settings.campaign.model_copy()
    camp_cfg.calls_per_second = 200
    d = Dialer(DryRunProvider(speedup=2000), camp_cfg, now_fn=lambda: noon)
    stats = await d.run(jobs)
    return {"plan": summary, "dial": stats}


# ----------------------------------------------------------------------------- QA Endpoints
class HumanReview(BaseModel):
    verdict: str          # PASS | REVIEW | FAIL
    score: float | None = None
    notes: str = ""


@app.get("/api/qa/recent")
async def get_recent_calls(limit: int = 50) -> list[dict]:
    return store.recent_calls(limit)


@app.get("/qa/review-queue")
async def review_queue(limit: int = 50) -> list[dict]:
    return store.review_queue(limit)


@app.get("/qa/calls/{call_id}")
async def get_call(call_id: str) -> dict:
    rec = store.get_call(call_id)
    if not rec:
        raise HTTPException(404)
    return rec


@app.post("/qa/calls/{call_id}/review")
async def post_review(call_id: str, review: HumanReview) -> dict:
    if review.verdict not in ("PASS", "REVIEW", "FAIL"):
        raise HTTPException(400, "verdict must be PASS, REVIEW or FAIL")
    store.record_human_review(call_id, review.verdict, review.score, review.notes)
    return {"ok": True}


@app.get("/qa/report")
@app.get("/api/qa/report")
async def qa_report(day: str | None = None) -> dict:
    return build_report(store, date.fromisoformat(day) if day else date.today(), settings.qa)


# ----------------------------------------------------------------------------- System Status
@app.get("/api/system/status")
async def system_status() -> dict:
    return {
        "bank_name": settings.bank_name,
        "llm_model": settings.llm.model,
        "llm_provider": settings.llm_provider,
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "tts_speaker": settings.sarvam.tts_speaker,
        "telephony_provider": settings.telephony.provider,
        "public_base_url": settings.telephony.public_base_url,
        "active_calls": len(ACTIVE)
    }


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "provider": settings.telephony.provider}


class DialPhonePayload(BaseModel):
    phone: str
    customer_id: str | None = None
    purpose: str = "emi_due_reminder"
    reference_id: str = "LN10099"


@app.post("/api/calls/dial-phone")
async def api_dial_phone(payload: DialPhonePayload) -> dict:
    """Trigger an actual live outbound phone call via Exotel to the specified mobile number."""
    if settings.telephony.provider != "exotel":
        raise HTTPException(400, f"Outbound calling requires telephony provider 'exotel', currently set to: '{settings.telephony.provider}'")

    if not settings.telephony.exotel_voicebot_app_id:
        raise HTTPException(
            400,
            "Exotel Voicebot App ID is missing! Please configure TELEPHONY__EXOTEL_VOICEBOT_APP_ID in .env with your Exotel App ID (created in Exotel App Bazaar)."
        )

    cust = core.get_customer(payload.customer_id) if payload.customer_id else None
    phone_to_call = payload.phone or (cust.phone if cust else "")
    if not phone_to_call:
        raise HTTPException(400, "Phone number is required.")

    # Format phone for Exotel if needed (e.g. 9636274630 or 09636274630)
    clean_phone = phone_to_call.replace("+91", "").strip()
    if len(clean_phone) == 10:
        clean_phone = "0" + clean_phone  # standard Indian STD prefix used by Exotel

    purp = CallPurpose(payload.purpose) if payload.purpose in [p.value for p in CallPurpose] else CallPurpose.EMI_DUE_REMINDER
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    job = CallJob(
        customer_id=cust.customer_id if cust else "C0005",
        phone=clean_phone,
        purpose=purp,
        reference_id=payload.reference_id,
        lang=cust.preferred_language.value if cust else "hi-IN",
        priority=1,
        not_before=now,
    )
    provider = ExotelProvider(settings.telephony)
    try:
        call_sid = await provider.place_call(job)
        logger.info(f"Placed Exotel live outbound call to {clean_phone}, CallSid={call_sid}")
        return {
            "ok": True,
            "call_sid": call_sid,
            "phone": clean_phone,
            "customer_id": job.customer_id,
            "message": f"Calling {clean_phone} via Exotel! Pick up your phone."
        }
    except Exception as e:
        logger.exception("Failed to place Exotel call")
        raise HTTPException(500, f"Exotel error placing call: {e}")

