"""Source-language prose and generated headings do not request translation."""
from obsidian_llm_wiki.core.models import (
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
)
from obsidian_llm_wiki.render.obsidian import (
    render_concept_page,
    render_entry_page,
    render_moc_page,
)


def test_chinese_entry_headings_stay_chinese():
    synthesis = SourceSynthesis(
        "实验结果", "处理组增长到十五个单位，对照组保持十个单位。", language="zh",
        key_points=["实验记录了处理组的增长。"], open_questions=["长期效果尚未测量。"],
    )
    page = render_entry_page(synthesis, "实验来源", ["实验概念"])
    assert "## 主要发现" in page
    assert "## 来源" in page
    assert "## Key Findings" not in page


def test_concept_navigation_headings_follow_body_language():
    english = ConceptNote("Growth", "growth", "The intervention increased observed growth.",
                          related=[ConceptLink("control")], claims=[Claim("Growth increased.")])
    control = ConceptNote("Control", "control", "The control did not receive treatment.")
    page = render_concept_page(english, all_concepts={"growth": english, "control": control})
    assert "## Cross-References" in page
    assert "关联图谱" not in page
    chinese = ConceptNote("增长", "增长", "处理组的增长显著高于未接受处理的对照组。",
                          claims=[Claim("处理组有所增长。")])
    page = render_concept_page(chinese)
    assert "## 论断" in page
    assert "## Claims" not in page


def test_moc_does_not_embed_foreign_language_prose():
    en = ConceptNote("Growth", "growth", "Treatment increased observed growth.")
    zh = ConceptNote("增长", "增长", "处理组的增长显著高于对照组。")
    moc = MapOfContent("Growth studies", "growth-studies", "Evidence from growth experiments.",
                       concept_slugs=["growth", "增长"])
    page = render_moc_page(moc, all_concepts={"growth": en, "增长": zh})
    assert "[[增长]]" in page
    assert zh.summary not in page
    assert "## Concepts" in page
    assert "No cross-references available yet" not in page


def test_chinese_titles_are_not_translated_or_renamed_during_render(tmp_path):
    from obsidian_llm_wiki.core.models import SynthesisBundle
    from obsidian_llm_wiki.render.obsidian import render_vault

    concept = ConceptNote("机器学习", "机器学习", "从数据中学习可用于预测的规律。",
                          aliases=["Machine learning"])
    source = SourceSynthesis("实验研究", "研究讨论了从数据学习的方法。", language="zh",
                             source_file="实验研究.md", concepts=[concept])
    moc = MapOfContent("学习方法", "学习方法", "本图谱组织机器学习相关的概念。",
                       concept_slugs=[concept.slug])
    render_vault(tmp_path, SynthesisBundle([source], [concept], [moc]), {},
                 cross_lingual_links={})
    page = (tmp_path / "concepts" / "机器学习.md").read_text()
    assert "title: 机器学习\n" in page
    assert "# 机器学习\n" in page
    assert "Machine learning" in page  # retained as alias, not invented into title
    assert "title: 学习方法\n" in (tmp_path / "mocs" / "学习方法.md").read_text()
    assert "title: 实验研究\n" in (tmp_path / "entries" / "实验研究.md").read_text()
