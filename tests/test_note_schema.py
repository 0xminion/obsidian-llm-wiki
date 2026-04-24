"""Tests for centralized note schemas."""

from pipeline.models import Language, Plan, Template
from pipeline.create.templates import generate_entry_content
from pipeline.lint import check_entry_template_sections
from pipeline.note_schema import ENTRY_SCHEMAS, effective_entry_schema, entry_schema, markdown_headings


def test_all_entry_templates_generated_sections_match_schema(tmp_path):
    vault = tmp_path / "vault"
    entries = vault / "04-Wiki" / "entries"
    entries.mkdir(parents=True)

    for template_name, schema in ENTRY_SCHEMAS.items():
        language = Language.ZH if template_name == "chinese" else Language.EN
        plan = Plan(
            hash=f"{template_name}-hash",
            title=f"{template_name} note",
            language=language,
            template=Template(template_name),
        )
        content = generate_entry_content(
            plan,
            {
                "url": f"https://example.com/{template_name}",
                "type": "web",
                "author": "Author",
                "content": "A sufficiently detailed source paragraph for generated summary fallback.",
            },
            f"{template_name}-source",
        )
        for heading in markdown_headings(schema):
            assert heading in content
        (entries / f"{template_name}.md").write_text(content, encoding="utf-8")

    assert check_entry_template_sections(vault) == []


def test_chinese_language_standard_template_uses_chinese_schema(tmp_path):
    vault = tmp_path / "vault"
    entries = vault / "04-Wiki" / "entries"
    entries.mkdir(parents=True)
    plan = Plan(
        hash="zh-standard",
        title="中文笔记",
        language=Language.ZH,
        template=Template.STANDARD,
    )
    content = generate_entry_content(
        plan,
        {
            "url": "https://example.com/zh",
            "type": "web",
            "author": "Author",
            "content": "这是一个足够长的中文内容摘要，用于验证中文语言的标准模板仍然生成中文章节。",
        },
        "zh-source",
    )
    entries.joinpath("zh-standard.md").write_text(content, encoding="utf-8")

    assert effective_entry_schema("zh", "standard") is ENTRY_SCHEMAS["chinese"]
    assert "template: standard" in content
    assert "## 摘要" in content
    assert check_entry_template_sections(vault) == []


def test_unknown_entry_template_defaults_to_standard_schema():
    assert entry_schema("does-not-exist") is ENTRY_SCHEMAS["standard"]
