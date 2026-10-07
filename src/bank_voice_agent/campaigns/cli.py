"""bva-campaign plan  [--date YYYY-MM-DD]          what would be called and why
   bva-campaign capacity --calls 10000 --aht 150    concurrency needed
   bva-campaign rehearse --customers 15000          dry-run a full day with simulated answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from loguru import logger

from bank_voice_agent.banking.core import MockCoreBanking
from bank_voice_agent.campaigns.dialer import Dialer, DryRunProvider
from bank_voice_agent.campaigns.planner import capacity, plan_day
from bank_voice_agent.config import Settings


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--customers", type=int, default=500)
    c = sub.add_parser("capacity")
    c.add_argument("--calls", type=int, default=10_000)
    c.add_argument("--aht", type=float, default=150, help="average handle time, seconds")
    c.add_argument("--hours", type=float, default=11)
    r = sub.add_parser("rehearse")
    r.add_argument("--customers", type=int, default=3000)
    a = ap.parse_args()
    s = Settings()

    if a.cmd == "capacity":
        print(json.dumps(capacity(a.calls, a.aht, a.hours), indent=2))
    elif a.cmd == "plan":
        day = date.fromisoformat(a.date)
        jobs, summary = plan_day(MockCoreBanking(n_customers=a.customers, today=day), day, s.campaign)
        print(json.dumps(summary, indent=2))
        for j in jobs[:10]:
            print(f"  p{j.priority} {j.purpose.value:26s} {j.customer_id} ref …{j.reference_id[-4:]} "
                  f"lang={j.lang} not_before={j.not_before:%H:%M} also={len(j.also_mention)}")
    elif a.cmd == "rehearse":
        logger.remove()
        day = date.today()
        jobs, summary = plan_day(MockCoreBanking(n_customers=a.customers, today=day), day, s.campaign)
        noon = datetime.combine(day, datetime.min.time(), ZoneInfo("Asia/Kolkata")).replace(hour=11)
        s.campaign.calls_per_second = 200          # compressed time
        d = Dialer(DryRunProvider(speedup=2000), s.campaign, now_fn=lambda: noon)
        stats = asyncio.run(d.run(jobs))
        print(json.dumps({"plan": summary, "dial": stats}, indent=2))


if __name__ == "__main__":
    main()
