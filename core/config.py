import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    app_title: str = "Agentic RAG Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173"]

    # Gemini / LangChain
    # Set GEMINI_API_KEY in your .env file — never hard-code the key here.
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash-lite"

    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "jarvis-documents"

    # SerpAPI (Scout node — web and academic search)
    # Set SERP_API_KEY in your .env file.
    serp_api_key: Optional[str] = None

    # Retrieval
    embedding_model: str = "gemini-embedding-2-preview"
    max_search_results: int = 5

    # LangSmith tracing
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="Jarvis", alias="LANGSMITH_PROJECT")


settings = Settings()


def _configure_langsmith_env() -> None:
    """Mirror LangSmith settings into process env for LangChain callbacks."""
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        # Backward-compatible flag used by some integrations.
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key


_configure_langsmith_env()
