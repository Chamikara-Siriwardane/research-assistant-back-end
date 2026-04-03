from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


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
    gemini_model: str = "gemini-2.0-flash"

    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "jarvis-documents"

    # Retrieval
    embedding_model: str = "gemini-embedding-001"
    max_search_results: int = 5


settings = Settings()
