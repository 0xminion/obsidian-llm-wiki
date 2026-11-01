from obsidian_llm_wiki.core.models import ConceptLink, ConceptNote, MapOfContent, SourceSynthesis
from obsidian_llm_wiki.synth.dedupe import merge_bundle
from obsidian_llm_wiki.synth.publication import plan_publication


def test_occurrences_keep_unresolved_original_text_and_provenance():
    source = SourceSynthesis(
        "Router",
        "Router uses a scheduler.",
        source_file="router.md",
        concepts=[
            ConceptNote(
                "Router",
                "router",
                "Router source",
                canonical_id="router-id",
                related=[ConceptLink("scheduler", "depends_on", "scheduler")],
            )
        ],
        maps=[MapOfContent("Stack", "stack", "A stack", concept_slugs=["scheduler"])],
        review_receipt={
            "references": {
                "concepts[0].related[0]": {
                    "original_text": "scheduler",
                    "original_slug": "scheduler",
                    "source_file": "router.md",
                    "source_revision": "r1",
                    "target_id": "",
                    "status": "unresolved",
                },
                "maps[0].members[0]": {
                    "original_text": "scheduler",
                    "original_slug": "scheduler",
                    "source_file": "router.md",
                    "source_revision": "r1",
                    "target_id": "",
                    "status": "unresolved",
                },
            }
        },
    )
    target = SourceSynthesis(
        "Unrelated scheduler",
        "",
        source_file="target.md",
        concepts=[
            ConceptNote("scheduler", "scheduler", "A postal robot", canonical_id="foreign-id")
        ],
    )
    bundle = merge_bundle([source, target])
    plan = plan_publication(bundle)
    router = next(c for c in bundle.concepts if c.slug == "router")
    assert router.related == [], "slug/display equality is not reference authority"
    assert plan.unresolved["router"][0].display == "scheduler"
    assert plan.unresolved["router"][0].source_file == "router.md"
    assert not plan.derived.get("scheduler")
    assert plan.unresolved_members["stack"][0]["source_revision"] == "r1"
    assert bundle.maps[0].concept_slugs == []
