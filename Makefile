.PHONY: install test simulate evals evals-offline serve campaign-plan capacity rehearse qa-report

install:
	pip install -e ".[dev]"

test:                 ## unit + integration tests, fully offline
	pytest -q

simulate:             ## full phone calls offline: real session, guardrails, QA; mock audio + demo model
	PYTHONPATH=src python -m bank_voice_agent.evals.simulate_call --scenario all

evals:                ## RELEASE GATE: eval suite against the real LLM in .env (run 3x, gate on worst)
	PYTHONPATH=src python -m bank_voice_agent.evals.run --repeat 3

evals-offline:        ## same harness with the demo model (shows how the gate reports)
	PYTHONPATH=src python -m bank_voice_agent.evals.run --llm mock

serve:
	PYTHONPATH=src uvicorn bank_voice_agent.api.main:app --host 0.0.0.0 --port 8000

campaign-plan:
	PYTHONPATH=src python -m bank_voice_agent.campaigns.cli plan

capacity:
	PYTHONPATH=src python -m bank_voice_agent.campaigns.cli capacity --calls 10000 --aht 150

rehearse:             ## dry-run a full reminder day with simulated answer rates
	PYTHONPATH=src python -m bank_voice_agent.campaigns.cli rehearse --customers 3000

qa-report:
	PYTHONPATH=src python -m bank_voice_agent.qa.report
