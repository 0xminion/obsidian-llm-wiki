---
name: obsidian-ingest
description: |
  Canonical skill for agentic ingestion into obsidian-llm-wiki vaults.
  Trigger on URLs, PDFs, YouTube links, clippings, "save to obsidian", or vault maintenance requests.
  Use the packaged pipeline CLI; do not hand-write wiki files except inbox items.
version: "2.1.0"
category: obsidian
trigger: |
  - "obsidian "
  - "vault "
  - "clip "
  - "ingest "
  - "save to obsidian"
  - "wiki "
  - "knowledge base"
  - URL patterns (https://, youtube.com, youtu.be, x.com, twitter.com)
  - "pdf "
  - "article "
  - "read later"
---

# Obsidian LLM Wiki — Agentic Ingestion

## Purpose

Turn a URL, PDF, YouTube video, podcast, tweet/X URL, or markdown clipping into a structured Obsidian wiki entry using the `obsidian-llm-wiki` pipeline.

The agent’s job is operational: write inbox files, run the CLI, verify health, and report exact paths. The agent should not bypass the pipeline by directly creating `04-Wiki` notes.

## Canonical repository and install

```bash
git clone https://github.com/0xminion/obsidian-llm-wiki.git
cd obsidian-llm-wiki
python3 -m pip install -e .
pipeline init ~/MyVault
```

Runtime config lives at:

```text
~/MyVault/Meta/Scripts/.env
```

## One-shot ingestion

```bash
export VAULT_PATH=~/MyVault
printf '%s\n' 'https://example.com/article' > "$VAULT_PATH/01-Raw/article.url"
pipeline ingest "$VAULT_PATH"
```

For pre-extracted markdown:

```bash
cp article.md "$VAULT_PATH/02-Clippings/article.md"
pipeline ingest "$VAULT_PATH"
```

## Supported sources

| Source | Primary path | Fallback path |
|---|---|---|
| Web articles / X / PDFs | defuddle / liteparse | curl extraction → archive.org → camoufox where available |
| YouTube | transcript API | Supadata → faster-whisper |
| Podcasts | AssemblyAI | local whisper |
| Markdown clippings | direct ingest | no HTTP extraction |

## Vault structure

```text
01-Raw/                 URL inbox
02-Clippings/           markdown clipping inbox
03-Queries/             question inbox
04-Wiki/
├── sources/            original extracted content
├── entries/            summaries and insights
├── concepts/           evergreen atomic notes
└── mocs/               maps of content
05-Outputs/             query answers
06-Config/              wiki-index, tag-registry, edges, schema-version
07-WIP/                 user drafts; never touch
08-Archive-Raw/         processed URLs
09-Archive-Queries/     processed questions
10-Archive-Clippings/   processed clippings
Meta/
├── Scripts/            .env, logs, cache.db, telemetry
├── prompts/            runtime prompt overrides
└── Templates/          runtime note templates
```

## Commands

| Command | Purpose |
|---|---|
| `pipeline ingest ~/MyVault` | Full pipeline: extract → plan → create. |
| `pipeline ingest ~/MyVault --parallel 3` | Tune creation parallelism. |
| `pipeline ingest ~/MyVault --dry-run` | Preview without writes. |
| `pipeline ingest ~/MyVault --review` | Stage plans for human review. |
| `pipeline approve ~/MyVault --json` | Atomically approve staged plans. |
| `pipeline reject ~/MyVault --json` | Reject staged plans. |
| `pipeline review-status ~/MyVault --json` | Inspect pending review queue. |
| `pipeline compile ~/MyVault` | Cross-link, merge, rebuild MoCs/index/edges. |
| `pipeline lint ~/MyVault --fix` | Safe vault health fixes. |
| `pipeline validate ~/MyVault --fix` | Post-write quality repair. |
| `pipeline doctor ~/MyVault --json` | First-run/config diagnostics. |
| `pipeline graph-doctor ~/MyVault --json` | Graph integrity diagnostics. |
| `pipeline migrate ~/MyVault --yes --json` | Idempotent schema/assets migrations. |
| `pipeline fixture ~/MyVault --adversarial --overwrite --json` | Golden adversarial corpus. |
| `pipeline telemetry ~/MyVault --json` | Recent redacted pipeline events. |
| `pipeline query ~/MyVault --ask "question" --fast` | Direct LLM vault Q&A. |

## Safety rules

1. Never touch `07-WIP/`.
2. Do not write directly into `04-Wiki/`; use `pipeline ingest`, review commands, or migration commands.
3. Titles, LLM filename suggestions, aliases, and review paths are untrusted.
4. Empty LLM output is failure/degraded, not success.
5. Do not expose API keys in summaries, command lines, logs, telemetry, or markdown.
6. Source and entry stems must remain distinct (`foo-source` vs `foo`) to avoid ambiguous Obsidian wikilinks.

## Operational verification

After non-trivial changes or before reporting release readiness:

```bash
ruff check .
pyflakes pipeline tests
pytest -q
python3 -m pip wheel . -w /tmp/obsidian-llm-wiki-wheel --no-deps
python3 -m venv /tmp/obsidian-llm-wiki-venv
/tmp/obsidian-llm-wiki-venv/bin/pip install /tmp/obsidian-llm-wiki-wheel/*.whl
/tmp/obsidian-llm-wiki-venv/bin/pipeline init /tmp/obsidian-llm-wiki-vault
/tmp/obsidian-llm-wiki-venv/bin/pipeline fixture /tmp/obsidian-llm-wiki-vault --adversarial --overwrite --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline graph-doctor /tmp/obsidian-llm-wiki-vault --json
/tmp/obsidian-llm-wiki-venv/bin/pipeline migrate /tmp/obsidian-llm-wiki-vault --yes --json
```

Expected current baseline: 852 tests, 8 packaged prompts, 9 packaged templates.

## Response format for agents

Report:

- exact vault path;
- files added or commands run;
- whether `ingest`/`compile`/`lint`/`graph-doctor` passed;
- any DLQ or review items left for the user.

Do not say “saved” unless the command completed and the target file exists.
