```python
# config/__init__.py

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field, AnyHttpUrl, validator


class Settings(BaseSettings):
    # FastAPI settings
    APP_NAME: str = Field("YouTube Summarizer", env="APP_NAME")
    DEBUG: bool = Field(False, env="DEBUG")
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")

    # OpenAI API
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_API_BASE: Optional[AnyHttpUrl] = Field(None, env="OPENAI_API_BASE")
    OPENAI_MODEL: str = Field("gpt-4", env="OPENAI_MODEL")
    OPENAI_TIMEOUT: int = Field(60, env="OPENAI_TIMEOUT")  # seconds

    # YouTube transcript settings
    YT_LANG: str = Field("en", env="YT_LANG")
    YT_RETRY: int = Field(3, env="YT_RETRY")

    # Summarization settings
    SUMMARY_MAX_TOKENS: int = Field(512, env="SUMMARY_MAX_TOKENS")
    SUMMARY_TEMPERATURE: float = Field(0.3, env="SUMMARY_TEMPERATURE")

    # Category estimation settings
    CATEGORY_TOP_K: int = Field(3, env="CATEGORY_TOP_K")

    # Security
    ALLOWED_ORIGINS: str = Field("*", env="ALLOWED_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("OPENAI_API_KEY")
    def openai_key_must_be_set(cls, v):
        if not v or v == "...":
            raise ValueError("OPENAI_API_KEY must be set")
        return v

    @property
    def allowed_origins_list(self):
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```