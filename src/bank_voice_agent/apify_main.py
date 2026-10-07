"""Apify Actor Runner for Neural Finance AI Voice Agent Outreach Pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from loguru import logger

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.models import CallPurpose, Customer, Language, Loan, LoanStatus
from bank_voice_agent.campaigns.dialer import Dialer, DryRunProvider, ExotelProvider
from bank_voice_agent.campaigns.planner import CallJob, plan_day
from bank_voice_agent.config import settings


def load_apify_input() -> dict:
    # 1. Apify default key-value store location
    apify_store = os.environ.get("APIFY_DEFAULT_KEY_VALUE_STORE_PATH")
    if apify_store:
        input_file = Path(apify_store) / "INPUT.json"
        if input_file.exists():
            return json.loads(input_file.read_text())

    # 2. Local apify_input.json
    local_input = Path("apify_input.json")
    if local_input.exists():
        try:
            return json.loads(local_input.read_text())
        except Exception:
            pass

    return {}


async def run_pipeline() -> None:
    raw_input = load_apify_input()
    mode = raw_input.get("mode", "dry_run_rehearse")
    logger.info(f"Starting Neural Finance Voice Outreach Pipeline in mode: {mode}")

    # Override settings from Apify Input if provided
    if raw_input.get("groqApiKey"):
        settings.llm.api_key = raw_input["groqApiKey"]
    if raw_input.get("sarvamApiKey"):
        settings.sarvam.api_key = raw_input["sarvamApiKey"]
    if raw_input.get("exotelSid"):
        settings.telephony.exotel_sid = raw_input["exotelSid"]
    if raw_input.get("exotelApiKey"):
        settings.telephony.exotel_api_key = raw_input["exotelApiKey"]
    if raw_input.get("exotelApiToken"):
        settings.telephony.exotel_api_token = raw_input["exotelApiToken"]
    if raw_input.get("exotelCallerId"):
        settings.telephony.exotel_caller_id = raw_input["exotelCallerId"]
    if raw_input.get("exotelVoicebotAppId"):
        settings.telephony.exotel_voicebot_app_id = raw_input["exotelVoicebotAppId"]

    contacts = raw_input.get("contacts", [])
    if not contacts and Path("contacts.csv").exists():
        import csv
        with open("contacts.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            contacts = list(reader)

    logger.info(f"Loaded {len(contacts)} leads for campaign outreach.")

    if mode == "serve_dashboard":
        import uvicorn
        logger.info("Serving Executive Operations Dashboard & Telephony WebSocket on port 8000...")
        uvicorn.run("bank_voice_agent.api.main:app", host="0.0.0.0", port=8000)
        return

    # Campaign Execution (Rehearsal or Live Calling)
    core = MockCoreBanking()
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

    # Register leads into banking core if not already present
    jobs: list[CallJob] = []
    for idx, c in enumerate(contacts):
        cid = f"LEAD{idx+1:04d}"
        phone = c.get("Phone") or c.get("phone", "")
        clean_phone = phone.replace("+91", "").strip()
        if len(clean_phone) == 10:
            clean_phone = "0" + clean_phone
        full_name = c.get("Name") or c.get("name", "Valued Customer")
        cust = Customer(
            customer_id=cid,
            name=full_name,
            first_name=full_name.split()[0],
            phone=phone,
            dob=date(1995, 1, 1),
            preferred_language=Language.HI,
            consent_service_calls=True,
            do_not_call=False
        )
        core.customers[cid] = cust
        job = CallJob(
            customer_id=cid,
            phone=clean_phone,
            purpose=CallPurpose.EMI_DUE,
            reference_id=f"LN{idx+100}",
            lang="hi-IN",
            priority=1,
            not_before=now_ist
        )
        jobs.append(job)

    if mode == "dry_run_rehearse":
        logger.info("⚡ Executing fast compliant rehearsal simulation (2,000x speedup)...")
        camp_cfg = settings.campaign.model_copy()
        camp_cfg.calls_per_second = 100
        noon = datetime.combine(date.today(), datetime.min.time(), ZoneInfo("Asia/Kolkata")).replace(hour=11)
        dialer = Dialer(DryRunProvider(speedup=2000), camp_cfg, now_fn=lambda: noon)
        stats = await dialer.run(jobs)
        logger.info(f"Rehearsal complete! Summary: {stats}")
        output = {"status": "success", "mode": "rehearsal", "leads_total": len(jobs), "dialer_stats": stats}
    else:
        logger.info("📞 Initiating live calls via Exotel...")
        camp_cfg = settings.campaign.model_copy()
        dialer = Dialer(ExotelProvider(settings.telephony), camp_cfg)
        stats = await dialer.run(jobs)
        logger.info(f"Live outreach complete! Summary: {stats}")
        output = {"status": "success", "mode": "live_dial", "leads_total": len(jobs), "dialer_stats": stats}

    # Save to Apify Dataset if present, or write to output.json
    apify_dataset = os.environ.get("APIFY_DEFAULT_DATASET_PATH")
    if apify_dataset:
        out_file = Path(apify_dataset) / "results.json"
        out_file.write_text(json.dumps(output, indent=2))
        logger.info(f"Saved results to Apify dataset at {out_file}")
    else:
        Path("campaign_output.json").write_text(json.dumps(output, indent=2))
        logger.info("Saved results to campaign_output.json")


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
