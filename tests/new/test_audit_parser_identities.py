import json

from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis


def test_model_slug_normalization_updates_all_local_references():
    payload = {
        "source_summary": "The source describes learning and optimization.",
        "concepts": [
            {"title": "Machine Learning", "slug": "Machine Learning", "summary": "Learning."},
            {"title": "Optimizer", "slug": "optimizer", "summary": "Optimization.",
             "related": [{"slug": "Machine Learning", "relation": "depends_on"}],
             "claims": [{"text": "Optimizers fit models.", "concept_slug": "Machine Learning"}]},
        ],
        "maps": [{"title": "Learning", "slug": "learning", "summary": "Learning methods.",
                  "concept_slugs": ["Machine Learning", "optimizer"]}],
    }
    synthesis = parse_single_source_synthesis(json.dumps(payload))
    assert synthesis is not None
    target = synthesis.concepts[0].slug
    assert target == "machine-learning"
    assert synthesis.concepts[1].related[0].slug == target
    assert synthesis.concepts[1].claims[0].concept_slug == target
    assert synthesis.maps[0].concept_slugs == [target, "optimizer"]


def test_chinese_title_generates_a_chinese_safe_identity():
    synthesis = parse_single_source_synthesis(json.dumps({
        "source_summary": "本文讨论机器学习的基本原理。",
        "concepts": [{"title": "机器 学习", "summary": "通过数据学习规律。"}],
    }))
    assert synthesis is not None
    assert synthesis.concepts[0].slug == "机器-学习"
