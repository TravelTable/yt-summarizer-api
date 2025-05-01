
# config/settings.py

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, AnyHttpUrl, validator

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "YouTube Transcript Summarizer"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Extracts YouTube video transcripts, summarizes them using GPT-4, "
        "and estimates video categories."
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # OpenAI / GPT-4
    OPENAI_API_KEY: str
    OPENAI_API_BASE: Optional[AnyHttpUrl] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_TIMEOUT: int = 60  # seconds

    # YouTube
    YOUTUBE_CAPTIONS_LANGS: List[str] = ["en"]
    YOUTUBE_TRANSCRIPT_API_TIMEOUT: int = 30  # seconds

    # CORS
    ALLOWED_ORIGINS: List[AnyHttpUrl] = []

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = str(BASE_DIR / "logs")
    LOG_FILE: str = "app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"

    # Rate Limiting
    RATE_LIMIT: str = "30/minute"

    # Security
    SECRET_KEY: str = os.urandom(32).hex()

    # Misc
    MAX_TRANSCRIPT_CHARS: int = 20000  # To avoid excessive token usage
    SUMMARY_MAX_TOKENS: int = 512
    CATEGORY_MAX_TOKENS: int = 128

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_allowed_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @validator("YOUTUBE_CAPTIONS_LANGS", pre=True)
    def assemble_youtube_langs(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    @property
    def log_path(self) -> str:
        return str(Path(self.LOG_DIR) / self.LOG_FILE)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
