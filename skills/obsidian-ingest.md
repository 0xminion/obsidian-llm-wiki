---
name: obsidian-ingest
description: "Process any URL, file, or link into the Obsidian vault. Drop URLs in chat, pipeline handles extraction + wiki creation."
version: 0.1.0
trigger: "obsidian"
---

# Obsidian Vault Processor

User says "obsidian" + URLs/files → write to inbox → run pipeline. The codebase handles everything else.

## Workflow

```bash
# 1. Write URLs to inbox (one .url file per URL)
echo "$URL" > "$VAULT_PATH/01-Raw/$SANITIZED.url"

# 2. Run pipeline (Python — canonical entry point)
cd ~/MyVault && ./run.sh
# or:
pipeline ingest ~/MyVault --parallel 3

# Background (3+ sources):
terminal(command="cd ~/MyVault && ./run.sh", background=true, notify_on_complete=true, timeout=1200)

# 3. Done — pipeline handles reindex, archive, sync internally
```

## Python CLI (canonical)

```bash
cd ~/MyVault && ./run.sh                      # full pipeline
pipeline ingest ~/MyVault --parallel 3        # explicit
pipeline ingest ~/MyVault --dry-run           # preview
pipeline ingest ~/MyVault --review            # save plans for review
pipeline ingest ~/MyVault --resume            # continue from reviewed plans
pipeline lint ~/MyVault                       # vault health checks
pipeline reindex ~/MyVault                    # rebuild wiki-index.md
pipeline stats ~/MyVault                      # show vault statistics
pipeline validate ~/MyVault                   # validate pipeline output
```

The repo root `run.sh` delegates to `python3 -m pipeline.cli`. The vault `run.sh` (created by `setup.sh`) also uses the Python pipeline.

## Pipeline

Three stages (all Python):
- **Stage 1 — Extract** (`pipeline/extract.py`): Parallel extraction via defuddle/TranscriptAPI/AssemblyAI. No agent. Output: `/tmp/obsidian-extracted-{hash}/{hash}.json`
- **Stage 2 — Plan** (`pipeline/plan.py`): Dedup + semantic concept matching via qmd + plan generation (1 agent). Output: `/tmp/extracted/plans.json`
- **Stage 3 — Create** (`pipeline/create.py`): N parallel agents write Source → Entry → Concept → MoC files. Output: vault files + reindex + archive

### Extraction Chain

| Source | Primary | Fallback |
|---|---|---|
| YouTube | TranscriptAPI (full URL) | Supadata → faster-whisper |
| Podcasts | iTunes lookup → search fallback → RSS → AssemblyAI | RSS description |
| X/Twitter | defuddle (FxTwitter API) | liteparse → browser |
| Web/URLs | defuddle | liteparse → defuddle --json |
| arxiv | defuddle (arxiv HTML) | alphaxiv.org |

### Timeouts

Stage 3 spawns hermes agents with 900s internal timeout. Terminal calls need ≥960s.

### MCP overhead

Hermes MCP servers (chrome-devtools, composio) add ~647s overhead per agent. Disable unused MCP servers in `~/.hermes/profiles/<profile>/config.yaml` under `mcp_servers`. After removal, stage 3 drops from ~930s to ~100s per source.

## Shell Scripts (supplementary)

The Python pipeline is canonical. Remaining shell scripts provide unique functionality:

| Script | Purpose |
|---|---|
| `setup-qmd.sh` | One-time qmd semantic search setup |
| `setup-git-hooks.sh` | Git initialization + hooks |
| `migrate-vault.sh` | Adopt existing vaults (scan/dry-run/execute) |

All pipeline operations (ingest, compile, lint, validate, reindex, stats, tags, query) are Python commands via `pipeline/cli.py`.

**Deleted** (replaced by Python pipeline): `compile-pass.sh`, `review-pass.sh`, `query-vault.sh`, `update-tag-registry.sh`, `lint-vault.sh`, `validate-output.sh`, `vault-stats.sh`, `process-inbox.sh`, `stage1-extract.sh`, `stage2-plan.sh`, `stage3-create.sh`, `reindex.sh`.

## Pitfalls

### Stage 3 timeout looks like failure but isn't

Terminal timeout < 900s kills parent shell but agent keeps running orphaned. Check for files:
```bash
find $VAULT_PATH/04-Wiki -newer /tmp/extracted/manifest.json -name "*.md"
```

### Podcast extraction

Apple Podcasts store IDs don't match iTunes lookup IDs. The code falls back to iTunes search by podcast name. If search returns wrong podcast, manually set the RSS feed URL.

### X/Twitter extraction

Defuddle uses FxTwitter API (`api.fxtwitter.com`). Some tweets return 404 (deleted, private, rate-limited). Defuddle accepts short content (≥100 chars) since some pages have minimal text.

## Note Structures

Check `template:` frontmatter field. Default `standard`.

| Template | Sections |
|---|---|
| standard (EN) | Summary → Core insights → Other takeaways → Diagrams → Open questions → Linked concepts |
| chinese (ZH) | 摘要 → 核心发现 → 其他要点 → 图表 → 开放问题 → 关联概念 |
| technical | Summary → Key Findings → Data/Evidence → Methodology → Limitations → Linked concepts |

**Concepts** (evergreen): Core concept → Context (flowing prose, no sub-headings) → Links
**MoCs**: Topic-specific bilingual sections (e.g., `Funding Rates / 资金费率`). NOT language-split. NO Open Questions.

## Naming

Source filenames = content title. See `title_to_filename()` in `pipeline/vault.py`.

- Chinese → Chinese. Papers → paper title. English → kebab-case.
- Tweets → topic, not tweet ID. YouTube → video title. Podcasts → episode title.
- ❌ NEVER: URL slugs, platform prefixes, author handles as filename

## Critical Rules

1. No stubs — every section needs real content
2. Tags: topic-specific English only (never `x.com`, `tweet`, `source`)
3. YAML: quote wikilinks (`source: "[[note]]"`), no nulls (`""` not `null`), quote titles with colons
4. Chinese body stays Chinese — English YAML/tags only
5. After pipeline: check for duplicates (`ls sources/ | sort | uniq -d`)
6. Stage 3 timeout ≠ failure — check vault for new files before re-running
7. Extraction: curl for APIs (Python urllib gets 403). Titles have markdown stripped automatically.
