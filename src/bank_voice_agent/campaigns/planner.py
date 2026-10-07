"""Outbound reminder planning: who to call today, why, and in what order.

Compliance rules baked in (defaults; confirm with the bank's compliance team):
- Calling window 08:00–19:00 IST only (RBI recovery-conduct rules; we apply it to all reminder calls).
- Caller ID must be a 1600-series number (TRAI mandate for BFSI service/transactional calls).
- Skip customers who opted out of reminder calls or have no service-call consent.
- At most `max_attempts_per_reminder` attempts per reminder, `retry_gap_minutes` apart, max 1 connected call per
  customer per day across all reminder types (one call covers EMI + policy if both are due).
- Overdue follow-ups pause while a promise-to-pay date is in the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.banking.models import CallPurpose, LoanStatus, PolicyStatus
from bank_voice_agent.config import CampaignSettings

PRIORITY = {  # lower = earlier in the day
    ("emi_overdue", None): 0,
    ("emi_due_reminder", 1): 1,
    ("policy_renewal_reminder", "grace"): 2,
    ("policy_renewal_reminder", 1): 3,
    ("policy_renewal_reminder", 7): 4,
    ("emi_due_reminder", 3): 5,
    ("policy_renewal_reminder", 15): 6,
}


@dataclass
class CallJob:
    customer_id: str
    phone: str
    purpose: CallPurpose
    reference_id: str
    lang: str
    priority: int
    not_before: datetime
    attempts: int = 0
    also_mention: list[str] = field(default_factory=list)   # other refs due, handled in the same call

    def custom_params(self) -> dict:
        return {"purpose": self.purpose.value, "customer_id": self.customer_id, "reference_id": self.reference_id,
                "lang": self.lang}


def plan_day(core: MockCoreBanking, day: date, cfg: CampaignSettings, contacted_today: set[str] | None = None,
             promises: dict[str, date] | None = None) -> tuple[list[CallJob], dict]:
    tz = ZoneInfo(cfg.timezone)
    contacted_today = contacted_today or set()
    promises = promises or {}
    skipped = {"do_not_call": 0, "no_consent": 0, "already_contacted": 0, "promise_pending": 0}
    by_customer: dict[str, list[tuple[int, CallPurpose, str]]] = {}

    for loan in core.loans.values():
        if loan.status == LoanStatus.CLOSED:
            continue
        days = (loan.next_due_date - day).days
        if loan.status == LoanStatus.OVERDUE:
            if promises.get(loan.account_no) and promises[loan.account_no] >= day:
                skipped["promise_pending"] += 1
                continue
            by_customer.setdefault(loan.customer_id, []).append(
                (PRIORITY[("emi_overdue", None)], CallPurpose.EMI_OVERDUE, loan.account_no))
        elif days in cfg.emi_remind_days_before and not (loan.autopay_mandate and days != 1):
            by_customer.setdefault(loan.customer_id, []).append(
                (PRIORITY[("emi_due_reminder", days)], CallPurpose.EMI_DUE, loan.account_no))

    for pol in core.policies.values():
        days = (pol.renewal_date - day).days
        key = None
        if pol.status == PolicyStatus.GRACE or (days < 0 and -days <= pol.grace_days):
            key = "grace"
        elif days in cfg.policy_remind_days_before:
            key = days
        if key is not None:
            by_customer.setdefault(pol.customer_id, []).append(
                (PRIORITY[("policy_renewal_reminder", key)], CallPurpose.POLICY_RENEWAL, pol.policy_no))

    jobs: list[CallJob] = []
    start = datetime.combine(day, time(cfg.window_start_hour, 0), tz)
    for cid, items in by_customer.items():
        c = core.get_customer(cid)
        if c.do_not_call:
            skipped["do_not_call"] += 1
            continue
        if not c.consent_service_calls:
            skipped["no_consent"] += 1
            continue
        if cid in contacted_today:
            skipped["already_contacted"] += 1
            continue
        items.sort()
        prio, purpose, ref = items[0]
        not_before = start
        if c.preferred_call_hour and cfg.window_start_hour <= c.preferred_call_hour < cfg.window_end_hour:
            not_before = datetime.combine(day, time(c.preferred_call_hour, 0), tz)
        jobs.append(CallJob(customer_id=cid, phone=c.phone, purpose=purpose, reference_id=ref,
                            lang=c.preferred_language.value, priority=prio, not_before=not_before,
                            also_mention=[r for _, _, r in items[1:]]))
    jobs.sort(key=lambda j: (j.priority, j.not_before))
    summary = {"jobs": len(jobs), "by_purpose": {}, "skipped": skipped}
    for j in jobs:
        summary["by_purpose"][j.purpose.value] = summary["by_purpose"].get(j.purpose.value, 0) + 1
    return jobs, summary


def in_window(now: datetime, cfg: CampaignSettings) -> bool:
    local = now.astimezone(ZoneInfo(cfg.timezone))
    return cfg.window_start_hour <= local.hour < cfg.window_end_hour


def capacity(calls_per_day: int, avg_handle_s: float, window_hours: float, peak_factor: float = 2.0) -> dict:
    """Concurrency needed to complete the day's calls inside the window, with headroom for the peak hour."""
    call_seconds = calls_per_day * avg_handle_s
    avg_concurrency = call_seconds / (window_hours * 3600)
    peak = avg_concurrency * peak_factor
    return {"call_minutes_per_day": round(call_seconds / 60), "avg_concurrent": round(avg_concurrency, 1),
            "peak_concurrent": round(peak), "calls_per_second_peak": round(calls_per_day / (window_hours * 3600)
                                                                           * peak_factor, 2)}


def next_retry(job: CallJob, now: datetime, cfg: CampaignSettings) -> datetime | None:
    """No-answer / busy → retry after the gap, only inside today's window, up to the attempt limit."""
    if job.attempts >= cfg.max_attempts_per_reminder:
        return None
    t = now + timedelta(minutes=cfg.retry_gap_minutes)
    end = now.astimezone(ZoneInfo(cfg.timezone)).replace(hour=cfg.window_end_hour, minute=0, second=0)
    return t if t < end else None
