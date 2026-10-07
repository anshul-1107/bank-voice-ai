import base64
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from loguru import logger

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.campaigns.planner import capacity, in_window, plan_day
from bank_voice_agent.config import CampaignSettings, Settings
from bank_voice_agent.evals.simulate_call import run_scenario

logger.remove()


@pytest.fixture
def settings(tmp_path):
    s = Settings(stt_provider="mock", tts_provider="mock", llm_provider="mock")
    s.turn.silence_reprompt_s = 1.0
    s.qa.db_path = str(tmp_path / "qa.sqlite3")
    s.judge.api_key = ""
    return s


async def test_happy_call_passes_qa(settings):
    out = await run_scenario("emi_hinglish_happy", settings, verbose=False)
    assert out["qa"]["verdict"] in ("PASS", "REVIEW")
    assert out["transport"]["hung_up"]
    assert any(a["type"] == "payment_link_sent" for a in out["call"]["actions"])
    assert "Ji, abhi dekh leti hoon." in [t["text"] for t in out["call"]["turns"]] or out["call"]["fillers"] >= 1


async def test_barge_in_clears_audio(settings):
    out = await run_scenario("barge_in", settings, verbose=False)
    assert out["call"]["barge_ins"] >= 1
    assert out["transport"]["clears"] >= 1


async def test_hallucination_is_never_heard_and_flags_review(settings):
    out = await run_scenario("hallucination_caught", settings, verbose=False)
    heard = " ".join(t["text"] for t in out["call"]["turns"] if t["role"] == "agent")
    assert "99,999" not in heard
    assert out["qa"]["verdict"] in ("REVIEW", "FAIL")
    assert "ungrounded_amount" in out["qa"]["metrics"]["violation_codes"]


async def test_locked_verification_transfers(settings):
    out = await run_scenario("wrong_dob_locked", settings, verbose=False)
    assert out["call"]["verification"]["state"] == "locked"
    assert out["transport"]["transferred"]
    assert not any(t["name"] == "get_loan_summary" and t["ok"] for t in out["call"]["tool_calls"])


async def test_silent_caller_gets_reprompts_then_goodbye(settings):
    out = await run_scenario("silent_caller", settings, verbose=False)
    assert out["call"]["reprompts"] >= 2 and out["transport"]["hung_up"]


def test_campaign_plan_respects_dnc_and_window():
    today = date(2026, 10, 7)
    core = MockCoreBanking(n_customers=200, today=today)
    jobs, summary = plan_day(core, today, CampaignSettings())
    ids = {j.customer_id for j in jobs}
    assert "C0004" not in ids            # golden customer with do_not_call
    assert "C0003" in ids                # overdue
    assert len(ids) == len(jobs)         # max one call per customer per day
    assert all(j.not_before.hour >= 8 for j in jobs)
    tz = ZoneInfo("Asia/Kolkata")
    assert not in_window(datetime(2026, 10, 7, 19, 5, tzinfo=tz), CampaignSettings())
    assert not in_window(datetime(2026, 10, 7, 7, 59, tzinfo=tz), CampaignSettings())
    assert in_window(datetime(2026, 10, 7, 8, 0, tzinfo=tz), CampaignSettings())


def test_capacity_math():
    c = capacity(10_000, 150, 11, 2.0)
    assert c["avg_concurrent"] == pytest.approx(37.9, 0.1) and c["peak_concurrent"] == 76


def test_exotel_websocket_end_to_end(monkeypatch, tmp_path):
    """Drive the real FastAPI WebSocket with Exotel-format frames (mock STT/TTS/LLM)."""
    monkeypatch.setenv("STT_PROVIDER", "mock")
    monkeypatch.setenv("TTS_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("QA__DB_PATH", str(tmp_path / "qa.sqlite3"))
    import importlib

    import bank_voice_agent.config as cfg
    importlib.reload(cfg)
    import bank_voice_agent.api.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    def media(text):
        return json.dumps({"event": "media", "stream_sid": "S1",
                           "media": {"payload": base64.b64encode(b"TXT:" + text.encode()).decode()}})

    with TestClient(main.app) as client:
        with client.websocket_connect("/telephony/exotel/ws") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(json.dumps({"event": "start", "start": {"stream_sid": "S1", "call_sid": "CA1",
                                                                 "from": "+919800000001", "to": "+911600000000"}}))
            first = json.loads(ws.receive_text())
            assert first["event"] in ("media", "mark") and first["stream_sid"] == "S1"
            ws.send_text(media("mera EMI kab hai"))
            got = [json.loads(ws.receive_text()) for _ in range(6)]
            assert any(m["event"] == "media" for m in got)
            ws.send_text(json.dumps({"event": "stop", "stop": {"reason": "callended"}}))
        assert main.store is not None
