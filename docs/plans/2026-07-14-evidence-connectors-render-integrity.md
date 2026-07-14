# Evidence, Connectors, and Render Integrity Plan

> **For Hermes:** Execute with focused TDD and an independent post-implementation review.

**Goal:** Add verified claim-level evidence and a stable source-connector boundary while correcting render rollback defects for reviewed pages and caller-owned bundles.

**Architecture:** Claims carry deterministic evidence spans; the LLM supplies a quote, and the pipeline verifies it against a source body before rendering clickable evidence links. Connectors expose one bounded, provenance-preserving extraction contract instead of adding another web fallback branch. Rendering operates on a defensive bundle copy and rolls back every pre-existing page changed by a failed render.

## Tasks

1. Add evidence dataclasses and backwards-compatible JSON conversion in `obsidian_llm_wiki/core/models.py`; test exact, ambiguous, and unmatched quotes.
2. Add deterministic quote resolution in a core evidence helper. It must emit UTF-8 character offsets, source hash, source filename, and a verification state; never invent offsets.
3. Thread source-aware evidence resolution through synthesis/cache/pipeline and retain evidence through semantic merges. Add integration tests for changed source hashes and merged concepts.
4. Render Claims with numbered markers and an `## Evidence` block containing the verified quote plus a clickable `[[source|title]]` link. Preserve legacy claims without evidence.
5. Define a `SourceConnector` contract in `ingest/connectors.py`: URL matching, validated/bounded extraction, normalized `SourceDoc`, and typed quality/failure result. Adapt generic web extraction as the first implementation, leaving specialist extractors behind the same dispatcher boundary. Add contract tests for redirect validation, byte caps, and provenance.
6. Correct `_RenderTransaction` so it snapshots/restores every existing page it can mutate, including reviewed generated metadata, while leaving user bodies intact. Render from a defensive deep copy of `SynthesisBundle` so failed renders never mutate caller state. Add fault-injection tests for reviewed frontmatter and cross-lingual MoC augmentation.
7. Run source/evidence/connector/render focused tests, full Python suite, Ruff, compilation, wheel build, plugin gates, independent security/integration review, then commit and push the follow-up.
