FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[prod]"
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# 1 worker per container: calls are long-lived WebSockets; scale with replicas, not workers.
CMD ["uvicorn", "bank_voice_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "10"]
