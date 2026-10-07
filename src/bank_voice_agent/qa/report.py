"""Layer 4 QA — daily quality report + alerts. Run by cron after midnight IST (bva-qa-report --date 2026-10-07).

Answers four questions every morning:
  1. Are callers getting correct, compliant help?      pass / review / fail rates, critical issues
  2. Does it feel human?                               latency p50/p95, interruptions, language mismatches
  3. Is the AI judge still trustworthy?                judge vs human agreement on reviewed calls
  4. What should we fix first?                         top violation codes, top failure reasons
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from bank_voice_agent.config import QASettings, Settings
from bank_voice_agent.qa.store import QAStore

IST = ZoneInfo("Asia/Kolkata")


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def build_report(store: QAStore, day: date, cfg: QASettings) -> dict:
    start = datetime(day.year, day.month, day.day, tzinfo=IST).astimezone(ZoneInfo("UTC")).isoformat()
    end = (datetime(day.year, day.month, day.day, tzinfo=IST) + timedelta(days=1)).astimezone(
        ZoneInfo("UTC")).isoformat()
    rows = store.rows_between(start, end)
    n = len(rows)
    verdicts = Counter(r["verdict"] or "UNSCORED" for r in rows)
    lat, codes, reasons, outcomes, purposes = [], Counter(), Counter(), Counter(), Counter()
    barge, mismatch, scores, agree, reviewed = 0, 0, [], 0, 0
    for r in rows:
        rec = json.loads(r["record_json"])
        lat += rec.get("first_audio_latency_ms", [])
        outcomes[rec.get("outcome", "")] += 1
        purposes[rec.get("purpose", "")] += 1
        barge += rec.get("barge_ins", 0) > 0
        for v in rec.get("violations", []):
            codes[v["code"]] += 1
        if r["reasons_json"]:
            for x in json.loads(r["reasons_json"]):
                reasons[x.split(":")[0][:60]] += 1
        if r["metrics_json"]:
            mismatch += json.loads(r["metrics_json"]).get("language_mismatches", 0) > 0
        if r["score"] is not None:
            scores.append(r["score"])
        if r["human_verdict"]:
            reviewed += 1
            agree += r["human_verdict"] == r["verdict"]

    pass_rate = verdicts["PASS"] / n if n else None
    p95 = _pct(lat, 95)
    alerts = []
    if pass_rate is not None and pass_rate < cfg.alert_pass_rate_floor:
        alerts.append(f"pass rate {pass_rate:.1%} below floor {cfg.alert_pass_rate_floor:.0%}")
    if verdicts["FAIL"]:
        alerts.append(f"{verdicts['FAIL']} FAILED calls need review within 4 hours")
    if p95 and p95 > cfg.latency_p95_budget_ms:
        alerts.append(f"latency p95 {p95:.0f} ms over budget {cfg.latency_p95_budget_ms} ms")
    if reviewed >= 20 and agree / reviewed < 0.85:
        alerts.append(f"judge-human agreement {agree / reviewed:.0%} < 85%: recalibrate judge prompt")
    return {
        "date": day.isoformat(), "calls": n, "verdicts": dict(verdicts),
        "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
        "mean_score": round(statistics.mean(scores), 2) if scores else None,
        "latency_ms": {"p50": _pct(lat, 50), "p95": p95, "p99": _pct(lat, 99), "turns": len(lat)},
        "calls_with_barge_in": barge, "calls_with_language_mismatch": mismatch,
        "top_violations": codes.most_common(8), "top_reasons": reasons.most_common(8),
        "outcomes": dict(outcomes), "purposes": dict(purposes),
        "human_reviewed": reviewed, "judge_human_agreement": round(agree / reviewed, 3) if reviewed else None,
        "review_queue_open": len(store.review_queue(10_000)), "alerts": alerts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(datetime.now(IST).date() - timedelta(days=1)).isoformat())
    a = ap.parse_args()
    s = Settings()
    print(json.dumps(build_report(QAStore(s.qa.db_path), date.fromisoformat(a.date), s.qa), indent=2))


if __name__ == "__main__":
    main()
