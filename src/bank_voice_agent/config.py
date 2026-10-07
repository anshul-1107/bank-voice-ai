"""All runtime settings. Override any value with env vars, e.g. LLM__MODEL=..., QA__PASS_THRESHOLD=4.0"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    # Any OpenAI-compatible endpoint. Defaults to Groq (same provider as the course repo).
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.3          # low: banking answers must be consistent
    max_tokens: int = 220             # phone replies are short; caps runaway answers
    timeout_s: float = 15.0


class JudgeSettings(BaseModel):
    """Separate, stronger model for post-call QA scoring. Never the same call path as the live agent."""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1"
    temperature: float = 0.0


class SarvamSettings(BaseModel):
    api_key: str = ""
    stt_url: str = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
    stt_model: str = "saaras:v3-realtime"
    stt_mode: str = "codemix"         # keeps Hinglish as spoken: "mera EMI kab hai"
    tts_url: str = "wss://api.sarvam.ai/text-to-speech/ws"
    tts_model: str = "bulbul:v2"
    tts_speaker: str = "priya"
    tts_pace: float = 1.05


class TelephonySettings(BaseModel):
    provider: Literal["exotel", "plivo"] = "exotel"
    sample_rate: int = 8000           # PSTN narrowband; both providers stream 8 kHz linear16 by default
    # Exotel
    exotel_sid: str = ""
    exotel_api_key: str = ""
    exotel_api_token: str = ""
    exotel_subdomain: str = "api.exotel.com"
    exotel_caller_id: str = ""        # must be a 1600-series number for BFSI service calls (TRAI)
    exotel_voicebot_app_id: str = ""
    # Plivo
    plivo_auth_id: str = ""
    plivo_auth_token: str = ""
    plivo_caller_id: str = ""
    public_base_url: str = "https://example.com"


class TurnTakingSettings(BaseModel):
    """The knobs that make the agent feel human. Tune with the latency report, not by feel."""
    endpoint_grace_ms: int = 250            # wait after STT final before answering
    incomplete_grace_ms: int = 900          # longer wait when the caller trails off ("aur...", "matlab...")
    barge_in_min_words: int = 2             # caller must say >=2 real words to interrupt (not "haan", "ok")
    barge_in_min_speech_ms: int = 400
    filler_after_ms: int = 700              # speak a filler if the first sentence isn't ready by then
    tool_sound_after_ms: int = 1500         # keyboard sound only for slow lookups
    silence_reprompt_s: float = 7.0
    silence_hangup_reprompts: int = 2
    max_call_s: int = 600


class QASettings(BaseModel):
    db_path: str = "data/qa.sqlite3"
    pass_threshold: float = 4.0             # mean rubric score (1-5) needed to PASS
    review_threshold: float = 3.0           # below this = FAIL
    random_review_rate: float = 0.02        # 2% of passing calls still go to humans (judge calibration)
    latency_p95_budget_ms: int = 1500       # first-audio latency budget per turn
    alert_pass_rate_floor: float = 0.92


class CampaignSettings(BaseModel):
    window_start_hour: int = 8              # RBI: no calls before 08:00
    window_end_hour: int = 19               # RBI: no calls after 19:00
    timezone: str = "Asia/Kolkata"
    max_concurrent_calls: int = 150
    calls_per_second: float = 3.0
    max_attempts_per_reminder: int = 3
    retry_gap_minutes: int = 120
    emi_remind_days_before: list[int] = Field(default_factory=lambda: [3, 1])
    policy_remind_days_before: list[int] = Field(default_factory=lambda: [15, 7, 1])


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    sarvam: SarvamSettings = Field(default_factory=SarvamSettings)
    telephony: TelephonySettings = Field(default_factory=TelephonySettings)
    turn: TurnTakingSettings = Field(default_factory=TurnTakingSettings)
    qa: QASettings = Field(default_factory=QASettings)
    campaign: CampaignSettings = Field(default_factory=CampaignSettings)

    stt_provider: Literal["sarvam", "mock"] = "sarvam"
    tts_provider: Literal["sarvam", "mock"] = "sarvam"
    llm_provider: Literal["openai_compatible", "mock"] = "openai_compatible"
    persona: str = "priya"
    bank_name: str = "Neural Finance"
    grievance_officer: str = "Ms. Anita Rao, grievance@neuralfinance.example, 1800-000-0000"
    redis_url: str = ""                    # empty = in-memory (dev)

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=[".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
    )


settings = Settings()
