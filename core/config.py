from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    app_title: str = "Agentic RAG Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = False

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173"]

    # Gemini / LangChain
    # Set GEMINI_API_KEY in your .env file — never hard-code the key here.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_search_results: int = 5


settings = Settings()
