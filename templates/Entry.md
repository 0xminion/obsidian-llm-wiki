---
title: "Concise descriptive title"
source: "[[Source note name]]"
date_entry: YYYY-MM-DD
tags:
  - entry
  - topic-tag-1
  - topic-tag-2
  - topic-tag-3
  - topic-tag-4
  - topic-tag-5
status: review
reviewed: ""
review_notes: ""
template: standard
aliases: []
---

# Title

## Summary

3-5 sentence overview. Plain language, no fluff.

## Core insights

1. First core insight — clear explanation with evidence.
2. Second core insight — concrete example, no jargon.
3. Third core insight — extract everything significant.

## Other takeaways

4. Continues numbering from Core insights.
5. Additional important findings.

## Diagrams

n/a

## Open questions

1. First question or gap from the source.
2. Second open question.

## Linked concepts

- [[Concept note 1]]
- [[Concept note 2]]
- [[Related Entry or MoC]]

---

## Template Variants

Use the `template:` frontmatter field to select a variant. The lint script
checks sections based on template type. Available templates:

### template: standard (default)
Sections: Summary, Core insights, Other takeaways, Diagrams (optional — n/a if not needed), Open questions, Linked concepts

### template: chinese (for Chinese-language sources)
Frontmatter: add `language: zh`, use `template: chinese`. Tags stay English.
Sections (Chinese body text):
  摘要 (3-5句中文摘要)
  核心发现 (编号列表，关键发现和论点)
  其他要点 (继续编号)
  图表 (可选 — 仅在图表确实有助于理解时加入，否则写 'n/a')
  开放问题 (编号列表)
  关联概念 (破折号列表, wikilinks)

### template: technical
Sections: Summary, Key Findings, Data/Evidence, Methodology, Limitations, Linked concepts
Use for: research papers, data-heavy articles, technical documentation

### template: comparison
Sections: Summary, Side-by-Side Comparison, Pros and Cons, Verdict, Linked concepts
Use for: product comparisons, framework evaluations, "X vs Y" articles

### template: procedural
Sections: Summary, Prerequisites, Steps, Gotchas, Linked concepts
Use for: tutorials, how-tos, setup guides, workflows
