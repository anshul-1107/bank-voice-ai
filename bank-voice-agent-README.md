# Bank Voice Agent

Real-time phone agent for an Indian bank / NBFC. Answers loan, EMI and insurance questions, calls customers about
upcoming EMIs and policy renewals, speaks Indian English and Hinglish, and checks the quality of every call.

Built on the ideas of the *realtime-phone-agents-course* (FastRTC / Twilio / Opik), rebuilt for:
per-call isolation, streaming turn-taking with barge-in, Indian telephony (Exotel / Plivo), Sarvam speech,
and a 4-layer QA pipeline.

## Quick start (no API keys needed)

```bash
pip install -e ".[dev]"
make test          # 30 tests: speech, guardrails, identity gate, tools, calls, QA, campaigns, Exotel WebSocket
make simulate      # 6 full phone calls offline, each scored by post-call QA
make capacity      # concurrency needed for 10,000 calls/day
make rehearse      # dry-run a reminder campaign day
```

Go live: copy `.env.example` to `.env`, add keys, `make evals` (must pass), `make serve`, point the Exotel
Voicebot applet at `wss://<host>/telephony/exotel/ws` (or Plivo answer URL at `/telephony/plivo/answer`).

## Layout

```
src/bank_voice_agent/
  pipeline/session.py     one CallSession per call: streaming turn-taking, barge-in, fillers, silence handling
  pipeline/factory.py     builds an isolated session per call; finalises + scores it after hangup
  agent/brain.py          LLM stream → sentences → guardrail → speech, with tool calls in between
  agent/tools.py          banking tools; identity gate enforced in code; amounts set by system, never the LLM
  agent/facts.py          fact ledger: the only numbers/dates the agent may say on this call
  agent/persona.py        Priya persona, system prompt (versioned), scripted openings (AI + recording disclosure)
  banking/identity.py     DOB + caller-ID/last-4 verification, 3 strikes lock
  banking/core.py         CoreBanking interface + realistic mock (500 customers, 4 golden test customers)
  speech/normalize.py     ₹3,12,640 → "three lakh twelve thousand…" / "teen lakh baarah hazaar…"
  speech/sarvam.py        Sarvam Saaras STT + Bulbul TTS (WebSocket streaming)
  telephony/exotel.py     Exotel Voicebot bidirectional stream (media / mark / clear)
  telephony/plivo.py      Plivo Audio Streams (playAudio / checkpoint / clearAudio)
  qa/guardrails.py        L2: per-sentence checks before speech (grounding, credentials, coercion, promises…)
  qa/scorer.py            L3: post-call scoring of 100% of calls (deterministic + LLM-judge rubric)
  qa/report.py            L4: daily report, alerts, judge-vs-human agreement
  evals/scenarios.yaml    L1: release-gate eval suite (18 cases, 10 critical)
  evals/run.py            eval runner + release gate (exit code for CI)
  evals/simulate_call.py  full offline call simulator
  campaigns/planner.py    who to call today (EMI due / overdue / policy renewal), 08:00–19:00 IST, DNC, retries
  campaigns/dialer.py     rate-limited dialer: concurrency + CPS caps, window check before every dial
  api/main.py             FastAPI: telephony WebSockets, QA review endpoints
```

## Quality assurance in one picture

| Layer | When | What | Blocks? |
|---|---|---|---|
| L1 Eval gate | before every deploy | 18 scripted calls incl. prompt injection, OTP offer, wrong person, distress; run 3x | Deploy fails if any critical case fails or <90% of others pass |
| L2 Guardrails | every sentence, <1 ms | made-up amounts/dates, account digits before verification, OTP/PIN requests, coercion, waivers, guarantees, "I am human" | Sentence never spoken; model told to self-correct |
| L3 Post-call scoring | 100% of calls | deterministic metrics + LLM judge on 6 dimensions | PASS / REVIEW / FAIL; FAIL reviewed in 4 h |
| L4 Human review + report | daily | all FAIL + REVIEW + 2% random PASS; judge-human agreement; alerts | Recalibrate judge if agreement < 85% |

Every FAIL becomes a new eval case, so the same mistake can't ship twice.

## Notes

- The rule-based demo model (`LLM_PROVIDER=mock`) exists only to run the pipeline offline. It fails the
  quality cases in the eval gate on purpose; the gate is meant to run against your real LLM.
- Sarvam payload field names are parsed tolerantly; run a live smoke test with your key before go-live.
- Compliance defaults (08:00–19:00 calling window, 1600-series caller ID, recording disclosure, grievance officer)
  must be confirmed by the bank's compliance team.
