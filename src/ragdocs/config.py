"""Configuração isolada por projeto, sem exigir chave nos testes offline."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr

from ragdocs.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    project_root: Path = PROJECT_ROOT
    api_key: SecretStr = SecretStr("")
    chat_model: str = "openai/gpt-4.1-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    max_question_chars: int = 1000
    max_output_tokens: int = 700
    timeout_seconds: float = 30.0
    max_retries: int = 1
    max_history: int = 20
    base_url: str = "https://openrouter.ai/api/v1"
    max_pdf_files: int = 5
    max_pdf_bytes: int = 10 * 1024 * 1024
    max_collection_bytes: int = 20 * 1024 * 1024
    max_collection_pages: int = 100
    max_extracted_chars: int = 200_000
    max_page_stream_bytes: int = 10 * 1024 * 1024

    @property
    def evaluation_dir(self) -> Path:
        return self.project_root / "runtime" / "evaluation"

    def require_api_key(self) -> str:
        key = self.api_key.get_secret_value().strip()
        if not key:
            raise ConfigurationError(
                "Configure OPENROUTER_API_KEY no .env do projeto ou no ambiente."
            )
        return key


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or PROJECT_ROOT).resolve()
    load_dotenv(root / ".env", override=False)
    return Settings(
        project_root=root,
        api_key=SecretStr(os.getenv("OPENROUTER_API_KEY", "")),
        chat_model=os.getenv("RAGDOCS_CHAT_MODEL", "openai/gpt-4.1-mini"),
        embedding_model=os.getenv("RAGDOCS_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
    )
