import pytest
from pydantic import SecretStr

from ragdocs.config import Settings, load_settings
from ragdocs.errors import ConfigurationError


def test_key_is_required_and_not_repr():
    with pytest.raises(ConfigurationError):
        Settings().require_api_key()
    settings = Settings(api_key=SecretStr("test-secret"))
    assert settings.require_api_key() == "test-secret"
    assert "test-secret" not in repr(settings)


def test_loads_only_project_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RAGDOCS_CHAT_MODEL", raising=False)
    parent = tmp_path / "parent"
    project = parent / "project"
    project.mkdir(parents=True)
    (parent / ".env").write_text("OPENROUTER_API_KEY=wrong\n")
    assert not load_settings(project).api_key.get_secret_value()
    (project / ".env").write_text("OPENROUTER_API_KEY=correct\nRAGDOCS_CHAT_MODEL=custom/model\n")
    loaded = load_settings(project)
    assert loaded.require_api_key() == "correct"
    assert loaded.chat_model == "custom/model"
    assert loaded.evaluation_dir == project / "runtime" / "evaluation"
    assert loaded.max_pdf_files == 5
