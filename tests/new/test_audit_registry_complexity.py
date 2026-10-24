from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry


def test_registry_point_writes_do_not_scan_unrelated_search_rows(tmp_path):
    counts = {"fts": [], "ann": []}
    for size in (100, 2000):
        with VaultRegistry(tmp_path / str(size)) as registry:
            connection = registry.connection
            with connection:
                connection.executemany(
                    "INSERT INTO concept_fts(concept_id,slug,title,aliases,tags,summary) "
                    "VALUES(?,?,?,'','','')",
                    ((f"noise-{i}", f"noise-{i}", f"Noise {i}") for i in range(size)),
                )
                connection.executemany(
                    "INSERT INTO ann_bucket VALUES(?,?,?,?,?,?)",
                    (
                        ("concept", "m", 4, table, i % 4096, f"noise-{i}")
                        for i in range(size)
                        for table in range(8)
                    ),
                )
            for operation in ("fts", "ann"):
                steps = []
                connection.set_progress_handler(lambda steps=steps: steps.append(1) or 0, 100)
                if operation == "fts":
                    registry.upsert_synthesis(
                        "new.md",
                        SourceDoc("New", "body"),
                        SourceSynthesis(
                            source_title="New",
                            source_summary="summary",
                            concepts=[ConceptNote("Target", "target", "summary")],
                        ),
                        ArtifactVersions("e", "p", "s", "m"),
                    )
                else:
                    registry.upsert_embedding(
                        "concept",
                        "target",
                        [1.0, 0.0, 0.0, 0.0],
                        model_id="m",
                        input_hash="h",
                        schema_version="s",
                    )
                connection.set_progress_handler(None, 0)
                counts[operation].append(len(steps))
    for operation, (small, large) in counts.items():
        assert large < small * 3, (operation, small, large)
