"""Tests for task-specific LLM model routing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.task_models import TaskModelConfig, resolve_task_model
from obsidian_llm_wiki.providers import llm


@pytest.mark.parametrize("task", ["ingest", "maintenance", "query"])
def test_resolve_task_model_uses_unified_model_when_no_override_is_set(task: str):
    """Existing single-model configurations route every supported task to that model."""
    config = TaskModelConfig(model="gemma3:27b")

    assert resolve_task_model(config, task) == "gemma3:27b"


@pytest.mark.parametrize(
    ("task", "override_name", "override_model"),
    [
        ("ingest", "ingest_model", "qwen3:32b"),
        ("maintenance", "maintenance_model", "gemma3:12b"),
        ("query", "query_model", "qwen3:8b"),
    ],
)
def test_resolve_task_model_uses_the_matching_task_override(
    task: str, override_name: str, override_model: str
):
    """Each task can select its own model without changing the unified default."""
    config = TaskModelConfig(model="gemma3:27b", **{override_name: override_model})

    assert resolve_task_model(config, task) == override_model


@pytest.mark.parametrize("override_name", ["ingest_model", "maintenance_model", "query_model"])
def test_task_model_config_rejects_blank_task_override(override_name: str):
    """An explicitly blank override cannot silently replace the unified model."""
    with pytest.raises(ValueError, match="must not be blank"):
        TaskModelConfig(model="gemma3:27b", **{override_name: "   "})


@pytest.mark.parametrize("task", ["", "synthesis", "INGEST"])
def test_resolve_task_model_rejects_unsupported_task_names(task: str):
    """Routing is deliberately limited to the three declared pipeline tasks."""
    config = TaskModelConfig(model="gemma3:27b")

    with pytest.raises(ValueError, match="Unsupported task"):
        resolve_task_model(config, task)


def test_resolve_task_model_does_not_apply_another_tasks_override():
    """An override affects only its own task; the other tasks retain the unified model."""
    config = TaskModelConfig(model="gemma3:27b", ingest_model="qwen3:32b")

    assert resolve_task_model(config, "maintenance") == "gemma3:27b"
    assert resolve_task_model(config, "query") == "gemma3:27b"


def test_load_config_wires_task_model_environment_overrides(monkeypatch, tmp_path):
    """Task model env vars override their task while LLM_MODEL remains the fallback."""
    for key in ("LLM_MODEL", "INGEST_MODEL", "MAINTENANCE_MODEL", "QUERY_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=default-model\n"
        "INGEST_MODEL=ingest-model\n"
        "MAINTENANCE_MODEL=maintenance-model\n"
        "QUERY_MODEL=query-model\n",
        encoding="utf-8",
    )

    config = load_config(str(env_file))

    assert config.llm.model == "default-model"
    assert config.llm.ingest_model == "ingest-model"
    assert config.llm.maintenance_model == "maintenance-model"
    assert config.llm.query_model == "query-model"


@pytest.mark.parametrize(
    ("task", "expected_model"),
    [
        ("ingest", "ingest-model"),
        ("maintenance", "maintenance-model"),
        ("query", "query-model"),
    ],
)
def test_call_llm_routes_explicit_tasks_to_their_configured_model(
    monkeypatch, task: str, expected_model: str
):
    """Task selection passes the resolved model without mutating the shared config."""
    captured: dict[str, object] = {}

    class FakeClient:
        def chat(self, system, user, **kwargs):
            captured["system"] = system
            captured["user"] = user
            captured.update(kwargs)
            return "response"

    config = load_config(
        LLM_MODEL="default-model",
        INGEST_MODEL="ingest-model",
        MAINTENANCE_MODEL="maintenance-model",
        QUERY_MODEL="query-model",
        RETRY_COUNT="1",
    )
    monkeypatch.setattr(llm, "create_llm_client", lambda _: FakeClient())

    assert llm.call_llm("system", "user", config, task=task) == "response"
    assert captured["model"] == expected_model
    assert config.llm.model == "default-model"


def test_call_llm_keeps_default_callers_on_the_unified_model(monkeypatch):
    """Callers that do not opt into a task retain the original LLM_MODEL behavior."""
    captured: dict[str, object] = {}

    class FakeClient:
        def chat(self, system, user, **kwargs):
            captured.update(kwargs)
            return "response"

    config = load_config(LLM_MODEL="default-model", QUERY_MODEL="query-model", RETRY_COUNT="1")
    monkeypatch.setattr(llm, "create_llm_client", lambda _: FakeClient())

    assert llm.call_llm("system", "user", config) == "response"
    assert captured["model"] == "default-model"


def test_query_command_uses_the_configured_query_task(monkeypatch, tmp_path):
    """The actual query command opts into QUERY_MODEL rather than the default model."""
    import obsidian_llm_wiki.cli.query as query_module
    from obsidian_llm_wiki.cli import app

    wiki = tmp_path / "04-Wiki" / "concepts"
    wiki.mkdir(parents=True)
    (wiki / "seed.md").write_text("# Seed\nGrounding evidence.", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_call(_system, _messages, _config, **kwargs):
        captured.update(kwargs)
        return "Grounded answer [[concepts/seed.md]]."

    monkeypatch.setattr(query_module, "call_llm", fake_call)
    monkeypatch.setenv("QUERY_MODEL", "query-model")

    result = CliRunner().invoke(app, ["query", str(tmp_path), "--ask", "seed", "--json"])

    assert result.exit_code == 0, result.output
    assert captured["task"] == "query"
