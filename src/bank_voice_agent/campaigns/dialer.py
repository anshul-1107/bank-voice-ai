"""Rate-limited outbound dialer.

- Global concurrency cap (live calls in progress) and calls-per-second cap (telco + provider limits).
- Hard stop outside the calling window, checked before EVERY dial, not just at start.
- Provider-specific "place call" behind one interface; DryRunProvider for testing the whole campaign offline.
- Every dial attempt is logged for audit (who, when, why, result).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

import httpx
from loguru import logger

from bank_voice_agent.campaigns.planner import CallJob, in_window, next_retry
from bank_voice_agent.config import CampaignSettings, TelephonySettings


class CallProvider(Protocol):
    async def place_call(self, job: CallJob) -> str: ...      # returns provider call id; raises on failure


class ExotelProvider:
    def __init__(self, cfg: TelephonySettings):
        self.cfg = cfg

    async def place_call(self, job: CallJob) -> str:
        url = (f"https://{self.cfg.exotel_api_key}:{self.cfg.exotel_api_token}@{self.cfg.exotel_subdomain}"
               f"/v1/Accounts/{self.cfg.exotel_sid}/Calls/connect.json")
        data = {"From": job.phone, "CallerId": self.cfg.exotel_caller_id,
                "Url": f"http://my.exotel.com/{self.cfg.exotel_sid}/exoml/start_voice/{self.cfg.exotel_voicebot_app_id}",
                "CustomField": json.dumps(job.custom_params()), "TimeLimit": 900, "TimeOut": 30,
                "StatusCallback": f"{self.cfg.public_base_url}/campaigns/status"}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, data=data)
            r.raise_for_status()
            return r.json()["Call"]["Sid"]


class PlivoProvider:
    def __init__(self, cfg: TelephonySettings):
        self.cfg = cfg

    async def place_call(self, job: CallJob) -> str:
        q = "&".join(f"{k}={v}" for k, v in job.custom_params().items())
        async with httpx.AsyncClient(timeout=10, auth=(self.cfg.plivo_auth_id, self.cfg.plivo_auth_token)) as c:
            r = await c.post(f"https://api.plivo.com/v1/Account/{self.cfg.plivo_auth_id}/Call/", json={
                "from": self.cfg.plivo_caller_id, "to": job.phone,
                "answer_url": f"{self.cfg.public_base_url}/telephony/plivo/answer?{q}", "answer_method": "POST",
                "ring_timeout": 30, "time_limit": 900})
            r.raise_for_status()
            return r.json()["request_uuid"]


class DryRunProvider:
    """Simulates answer rates and call lengths. Use to rehearse a 10k-call day in minutes."""

    def __init__(self, answer_rate: float = 0.62, seed: int = 5, speedup: float = 600.0):
        self.rng = random.Random(seed)
        self.answer_rate = answer_rate
        self.speedup = speedup

    async def place_call(self, job: CallJob) -> str:
        if self.rng.random() > self.answer_rate:
            raise ConnectionRefusedError("no_answer")
        await asyncio.sleep(self.rng.uniform(60, 180) / self.speedup)   # call in progress
        return f"dry-{job.customer_id}-{job.attempts}"


@dataclass
class Dialer:
    provider: CallProvider
    cfg: CampaignSettings
    audit: list[dict] = field(default_factory=list)
    now_fn: callable = lambda: datetime.now(ZoneInfo("Asia/Kolkata"))

    async def run(self, jobs: list[CallJob], stop_after_s: float | None = None) -> dict:
        sem = asyncio.Semaphore(self.cfg.max_concurrent_calls)
        interval = 1.0 / self.cfg.calls_per_second
        queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        for i, j in enumerate(jobs):
            queue.put_nowait((j.priority, i, j))
        stats = {"dialled": 0, "answered": 0, "no_answer": 0, "retries_scheduled": 0, "gave_up": 0,
                 "blocked_outside_window": 0, "deferred_to_later_today": 0, "peak_concurrency": 0}
        live = 0
        t_start = time.monotonic()
        tasks = []

        async def dial(job: CallJob) -> None:
            nonlocal live
            async with sem:
                live += 1
                stats["peak_concurrency"] = max(stats["peak_concurrency"], live)
                job.attempts += 1
                rec = {"customer_id": job.customer_id, "purpose": job.purpose.value, "ref": job.reference_id[-4:],
                       "attempt": job.attempts, "at": self.now_fn().isoformat()}
                try:
                    rec["call_id"] = await self.provider.place_call(job)
                    rec["result"] = "answered"
                    stats["answered"] += 1
                except Exception as e:
                    rec["result"] = str(e) or type(e).__name__
                    stats["no_answer"] += 1
                    retry_at = next_retry(job, self.now_fn(), self.cfg)
                    if retry_at:
                        stats["retries_scheduled"] += 1
                        job.not_before = retry_at
                        queue.put_nowait((job.priority + 10, id(job) + job.attempts, job))
                    else:
                        stats["gave_up"] += 1
                finally:
                    live -= 1
                    self.audit.append(rec)

        while True:
            if stop_after_s and time.monotonic() - t_start > stop_after_s:
                break
            if queue.empty():
                if not any(not t.done() for t in tasks):
                    break
                await asyncio.sleep(0.01)
                continue
            if not in_window(self.now_fn(), self.cfg):
                stats["blocked_outside_window"] = queue.qsize()
                logger.warning("outside calling window — stopping dialer")
                break
            prio, seq, job = await queue.get()
            if job.not_before > self.now_fn():
                # not due yet (retry gap / preferred hour). If nothing else can run now, stop this pass;
                # the scheduler re-invokes the dialer every few minutes in production.
                queue.put_nowait((prio, seq, job))
                live_tasks = any(not t.done() for t in tasks)
                if not live_tasks and all(j.not_before > self.now_fn() for _, _, j in queue._queue):
                    stats["deferred_to_later_today"] = queue.qsize()
                    break
                await asyncio.sleep(0.01)
                continue
            stats["dialled"] += 1
            tasks.append(asyncio.create_task(dial(job)))
            await asyncio.sleep(interval)
        await asyncio.gather(*tasks, return_exceptions=True)
        return stats
