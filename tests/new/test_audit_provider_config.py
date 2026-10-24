"""Config audit regressions: documented names, precedence and safe numeric bounds."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.config import load_config

MODEL_KEYS = [
    ("LLM_INGEST_MODEL", "INGEST_MODEL", "ingest_model"),
    ("LLM_MAINTENANCE_MODEL", "MAINTENANCE_MODEL", "maintenance_model"),
    ("LLM_QUERY_MODEL", "QUERY_MODEL", "query_model"),
    ("LLM_EXPAND_MODEL", "PASS2_MODEL", "expand_model"),
]


@pytest.fixture(autouse=True)
def isolated_config_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Do not inherit operator credentials or unrelated configuration in fixtures.
    monkeypatch.setattr("os.environ", {})


@pytest.mark.parametrize("canonical,legacy,attribute", MODEL_KEYS)
@pytest.mark.parametrize("mode", ["canonical", "legacy", "both", "canonical_blank"])
def test_documented_task_model_names_keep_legacy_aliases(canonical, legacy, attribute, mode):
    values = {"LLM_MODEL": "default"}
    if mode != "legacy":
        values[canonical] = " " if mode == "canonical_blank" else " new-model "
    if mode != "canonical":
        values[legacy] = " old-model "
    expected = (
        None if mode == "canonical_blank" else ("old-model" if mode == "legacy" else "new-model")
    )
    assert getattr(load_config(**values).llm, attribute) == expected


@pytest.mark.parametrize("canonical,legacy,attribute", MODEL_KEYS)
@pytest.mark.parametrize("explicit_file", [False, True])
def test_alias_precedence_follows_configuration_layers(
    tmp_path,
    monkeypatch,
    canonical,
    legacy,
    attribute,
    explicit_file,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    base = tmp_path / ".env"
    base.write_text(f"{canonical}=base\nVAULT_PATH={vault}\n", encoding="utf-8")
    (vault / ".env").write_text(f"{legacy}=vault\n", encoding="utf-8")
    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv(canonical, "process")
    env_file = str(base) if explicit_file else None
    config = load_config(env_file=env_file)
    assert getattr(config.llm, attribute) == ("vault" if explicit_file else "process")
    # Explicit kwargs win even when they use the old spelling and a lower
    # priority source uses the documented spelling.
    config = load_config(env_file=env_file, **{legacy: "explicit"})
    assert getattr(config.llm, attribute) == "explicit"


def test_blank_high_priority_alias_disables_lower_priority_model(tmp_path):
    env_file = tmp_path / "models.env"
    env_file.write_text("LLM_INGEST_MODEL=file-model\n", encoding="utf-8")
    assert load_config(str(env_file), INGEST_MODEL=" ").llm.ingest_model is None


# Resource limits are deliberately generous; zero is retained where it disables
# optional behavior, but never for chunk steps, worker counts or attempt counts.
INT_BOUNDS = [
    ("LLM_TIMEOUT_MS", "llm.timeout_ms", 1, 86_400_000),
    ("LLM_CONTEXT_WINDOW", "llm.context_window", 0, 10_000_000),
    ("CHUNK_SIZE", "chunk_size", 1, 10_000_000),
    ("COMPILE_CONCURRENCY", "compile_concurrency", 1, 256),
    ("REQUEST_MIN_INTERVAL_MS", "request_min_interval_ms", 0, 3_600_000),
    ("RETRY_COUNT", "retry_count", 1, 10),
    ("RETRY_BASE_MS", "retry_base_ms", 0, 3_600_000),
    ("RETRY_MULTIPLIER", "retry_multiplier", 1, 10),
]


def setting(config, attribute):
    if attribute.startswith("llm."):
        return getattr(config.llm, attribute.split(".")[1])
    return getattr(config, attribute)


@pytest.mark.parametrize("key,attribute,minimum,maximum", INT_BOUNDS)
@pytest.mark.parametrize("invalid", ["below", "above"])
def test_unsafe_execution_config_fails_before_workers_or_chunking(
    key,
    attribute,
    minimum,
    maximum,
    invalid,
):
    value = minimum - 1 if invalid == "below" else maximum + 1
    with pytest.raises(ValueError, match=key):
        load_config(**{key: str(value)})


@pytest.mark.parametrize("key,attribute,minimum,maximum", INT_BOUNDS)
def test_valid_numeric_config_and_malformed_string_fallback_are_preserved(
    key,
    attribute,
    minimum,
    maximum,
):
    for value in (minimum, maximum):
        assert setting(load_config(**{key: str(value)}), attribute) == value
    assert setting(load_config(**{key: "not-a-number"}), attribute) == setting(
        load_config(), attribute
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_size", 0),
        ("chunk_size", -1),
        ("compile_concurrency", 0),
        ("compile_concurrency", 257),
        ("retry_count", 0),
        ("retry_base_ms", -1),
        ("chunk_size", 1.5),
        ("compile_concurrency", True),
    ],
)
def test_programmatic_config_has_same_execution_safety_bounds(field, value):
    from obsidian_llm_wiki.config import Config

    with pytest.raises(ValueError, match=field.upper()):
        Config(**{field: value})


@pytest.mark.parametrize(
    "key",
    [
        "SIMILARITY_DEDUP_THRESHOLD",
        "CROSS_LINGUAL_THRESHOLD",
        "MOC_ASSIGNMENT_THRESHOLD",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_similarity_thresholds_must_be_finite_probabilities(key, value):
    with pytest.raises(ValueError, match=key):
        load_config(**{key: value})


@pytest.mark.parametrize(
    "key",
    [
        "MAX_SOURCE_CHARS",
        "MIN_SOURCE_CHARS",
        "CANDIDATE_LIMIT",
        "CANDIDATE_CHAR_BUDGET",
        "ANN_CANDIDATE_POOL",
        "EVIDENCE_WINDOW_CHARS",
        "EVIDENCE_CONTEXT_PARAGRAPHS",
        "INDEX_SHARD_SIZE",
        "REPORT_SHARD_SIZE",
        "GRAPH_SHARD_SIZE",
        "GRAPH_MONOLITH_THRESHOLD",
        "CONCEPT_MIN_BODY_CHARS",
        "ENTRY_MIN_BODY_CHARS",
        "CLIPPING_MIN_BODY_CHARS",
        "MAX_DOCUMENT_BYTES",
        "MAX_DOCUMENT_CANDIDATES",
        "PARSER_TIMEOUT_SECONDS",
        "MAX_PARSER_STDOUT_BYTES",
        "MAX_PARSER_STDERR_BYTES",
        "MAX_HTML_BYTES",
    ],
)
@pytest.mark.parametrize("value", ["-1", "100000000000000000000"])
def test_numeric_resource_budgets_are_nonnegative_and_bounded(key, value):
    with pytest.raises(ValueError, match=key):
        load_config(**{key: value})


def test_optional_zero_settings_remain_valid():
    keys = [
        "MIN_SOURCE_CHARS",
        "CONCEPT_MIN_BODY_CHARS",
        "ENTRY_MIN_BODY_CHARS",
        "CLIPPING_MIN_BODY_CHARS",
        "GRAPH_MONOLITH_THRESHOLD",
        "EVIDENCE_CONTEXT_PARAGRAPHS",
        "REQUEST_MIN_INTERVAL_MS",
        "RETRY_BASE_MS",
        "LLM_CONTEXT_WINDOW",
    ]
    config = load_config(**dict.fromkeys(keys, "0"))
    assert (
        config.min_source_chars == config.concept_min_body_chars == config.llm.context_window == 0
    )
