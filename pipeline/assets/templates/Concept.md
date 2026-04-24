---
title: "Concept name as concise phrase"
type: concept
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
sources:
  - "[[Source note 1]]"
  - "[[Source note 2]]"
tags:
  - concept
  - topic-tag-1
  - topic-tag-2
status: evergreen
aliases: []
---

# Concept Name

## Core concept

Single overview paragraph defining the concept. Plain language — like
explaining to a curious friend, NOT like writing a textbook. 2-3 sentences.

## Context

Flowing prose (2-4 paragraphs) covering how it works, why it matters,
real-world evidence, and any tensions or debates. Do NOT use sub-headings
within Context. Write naturally as connected paragraphs.

## Links

- [[Related Concept 1]]
- [[Related Concept 2]]
- [[Related Entry 1]]
- [[Related MoC]]

---

## Language variants

### English (default)
Sections: Core concept, Context, Links
Sources go in frontmatter `sources:` field (not body section).

### Chinese (language: zh in frontmatter)
Frontmatter: add `language: zh`, tags stay English, sources in frontmatter.
Sections (Chinese body text):
  核心概念 (一段概述，定义概念。通俗易懂，2-3句)
  背景 (连贯正文2-4段，涵盖运作机制、为什么重要、实际案例、争议与不确定性。
        不要在"背景"内使用子标题)
  关联 (破折号列表, wikilinks)
