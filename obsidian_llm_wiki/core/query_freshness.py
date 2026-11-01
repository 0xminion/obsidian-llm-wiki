"""Read-only query admission over a pinned captured publication.

Input observation is delegated to publication_inputs, not inferred from prose,
filenames, or an LLM. These observations are NOT an atomic latest-at-commit
promise. History remains immutable even when it is ineligible for current Q&A.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType


class QueryFreshnessError(RuntimeError):
    def __init__(self, message, freshness):
        super().__init__(message)
        self.freshness = freshness


def observe_inputs(vault_root, manifest):
    """Fail closed until the shared safe input-mapping helper is installed."""
    try:
        from obsidian_llm_wiki.core.publication_inputs import input_freshness
    except ImportError:
        return {'status': 'unverifiable', 'changed_paths': [],
                'reasons': ['Publication input freshness inspector is unavailable']}
    result = input_freshness(vault_root, manifest)
    if (not isinstance(result, dict)
            or result.get('status') not in {'current', 'drifted', 'unverifiable'}
            or not isinstance(result.get('changed_paths'), list)
            or not isinstance(result.get('reasons'), list)):
        return {'status': 'unverifiable', 'changed_paths': [],
                'reasons': ['Invalid publication input freshness result']}
    return result


def require_current_inputs(config, committed, *, generation=None):
    """Check one validated publication and human inputs at an acceptance boundary."""
    if committed is None:
        report = {'status': 'unverifiable', 'generation': None, 'changed_paths': [],
                  'reasons': ['No immutable captured publication; build the wiki first']}
    elif generation is not None and committed[1]['generation'] != generation:
        report = {'status': 'generation_changed', 'generation': generation,
                  'changed_paths': [], 'reasons': ['Publication changed during query; retry']}
    else:
        report = dict(observe_inputs(config.vault, committed[1]))
        report['generation'] = committed[1]['generation']
    if report['status'] != 'current':
        details = '; '.join([*report['reasons'], *report['changed_paths']])
        raise QueryFreshnessError(
            f"Cannot answer as current: input/publication freshness {report['status']}. "
            f"{details}. Rebuild or resolve the publication before querying.", report,
        )
    captured_only = report.get('input_scope') == 'captured_only'
    return {**report, 'status': 'captured_only' if captured_only else 'observed_current',
            'scope': 'captured_generation',
            'guarantee': ('Captured evidence only; no live file for some sources' if captured_only
                          else 'Inputs matched at observations, not atomic latest-at-commit')}


def _check_schema(directory):
    """Never let a resealed unknown registry version become query authority."""
    from obsidian_llm_wiki.core.registry import _SCHEMA_VERSION

    connection = sqlite3.connect(':memory:')
    try:
        raw = bytearray((directory / '.llmwiki/registry.db').read_bytes())
        raw[18] = raw[19] = 1  # private copy of the closed WAL database only
        connection.deserialize(bytes(raw))
        connection.execute('PRAGMA query_only=ON')
        version = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if version != (str(_SCHEMA_VERSION),):
            raise RuntimeError('Unsupported generation registry schema version')
        contexts = {(name, revision): content for name, revision, content in connection.execute(
            'SELECT filename,revision,content FROM identity_source_context')}
        versions = dict(connection.execute('SELECT filename,extractor_version FROM source'))
        return contexts, versions
    finally:
        connection.close()


def classify_pages(graph, directory, manifest, config=None, *, policy_problem=""):
    """Attach deterministic source/owner status; never rewrite captured receipts.

    Entire mixed-owner pages are excluded if any contribution is not admitted.
    Unmapped navigation and other generated aggregates are not evidence. This
    conservative page-level boundary prevents a snippet choosing a stale unit.
    """
    from obsidian_llm_wiki.core.identity_plan import PLAN_VERSION
    from obsidian_llm_wiki.core.models import source_synthesis_from_dict
    from obsidian_llm_wiki.core.render_integrity import load_render_hashes, rendered_page_problem
    from obsidian_llm_wiki.core.source_files import validate_source_filename
    from obsidian_llm_wiki.ingest.sources import source_body_for_receipt
    from obsidian_llm_wiki.render.obsidian import _entry_slug
    from obsidian_llm_wiki.synth.review import (
        verify_review_receipt,
        verify_review_receipt_integrity,
    )

    try:
        contexts, versions = _check_schema(directory)
        plan = json.loads((directory / '.llmwiki/publications' /
                           (manifest['identity_plan_id'] + '.json')).read_bytes())
        if plan['version'] != PLAN_VERSION:
            raise RuntimeError('Unsupported identity publication plan version')
        publication = json.loads(plan['publication_json'])
        owners = {}
        statuses = {}
        for name, serialized in plan['sources']:
            name = validate_source_filename(name)
            synthesis = source_synthesis_from_dict(json.loads(serialized))
            status = publication['source_status'].get(name, '')
            reason = status
            if not status:
                captured = directory / '.llmwiki/source-inputs' / name
                revision = synthesis.review_receipt.get('source_sha256')
                recorded_version = manifest.get('compiler_versions', {}).get(
                    'artifacts', {}).get('extractor', '')
                # Missing/disagreeing producer stamps cannot authorize legacy replay.
                version = recorded_version if versions.get(name) == recorded_version else ''
                body, interpretation = source_body_for_receipt(
                    captured.read_bytes().decode('utf-8'), revision,
                    extractor_version=version, historical_content=contexts.get((name, revision)),
                )
                if body is None:
                    status, reason = 'unverifiable', interpretation
                else:
                    reason = verify_review_receipt_integrity(synthesis, body, name)
                    if reason:
                        status = 'unadmitted'
                    elif interpretation == 'historical-universal-newlines':
                        status = 'historical_interpretation'
                        reason = ('Verified historical extract-v2 universal-newline body; '
                                  'not current exact-source interpretation; rebuild required')
                    elif policy_problem or config is None:
                        status = 'unverifiable'
                        reason = policy_problem or (
                            'Current review policy unavailable; historical receipt integrity only'
                        )
                    else:
                        reason = verify_review_receipt(synthesis, body, name, config)
                        if reason:
                            status = 'unadmitted'
            statuses[name] = (status or 'eligible', reason)
            for relative in (f'sources/{name}', f'entries/{_entry_slug(synthesis)}.md',
                             *(f'concepts/{c.slug}.md' for c in synthesis.concepts),
                             *(f'mocs/{m.slug}.md' for m in synthesis.maps)):
                owners.setdefault(relative, set()).add(name)
        prefix = f"04-Wiki/OLW-Publications/generations/{manifest['generation']}/"
        render_hashes = load_render_hashes(directory)
        classified = {}
        for path, page in graph.pages.items():
            if page.orphaned:
                classified[path] = page
                continue
            relative = path.removeprefix(prefix)
            names = sorted(owners.get(relative, ()))
            extra = {}
            if relative.startswith('concepts/'):
                extra = publication['concept_status'].get(Path(relative).stem, {})
            elif relative.startswith('mocs/'):
                extra = publication['moc_status'].get(Path(relative).stem, {})
            by_source = tuple((name, extra.get(name) or statuses[name][0]) for name in names)
            reasons = tuple(f'{name}: {status} {statuses[name][1]}'.strip()
                            for name, status in by_source if status != 'eligible')
            if not names:
                state, reasons = 'unverifiable', ('No admitted source ownership for this page',)
            elif reasons:
                state = ('unverifiable' if all(status in {'eligible', 'unverifiable'}
                                              for _name, status in by_source)
                         else 'ineligible')
            else:
                state = 'eligible'
            render_problem = rendered_page_problem(relative, page.rendered_sha256, render_hashes)
            if render_problem:
                # Keep an existing source/policy exclusion as the primary status.
                # Render uncertainty remains visible and never grants eligibility.
                reasons = (*reasons, render_problem)
                if state == 'eligible':
                    state = 'render_unverified'
            classified[path] = replace(page, publication_status=state,
                                       status_reasons=reasons, source_statuses=by_source)
        return replace(graph, pages=MappingProxyType(classified))
    except (OSError, KeyError, TypeError, ValueError, IndexError, sqlite3.DatabaseError) as exc:
        raise RuntimeError(f'Unverifiable query publication metadata: {exc}') from exc


def eligible_graph(graph):
    """Filter before ranking/top-k so excluded pages cannot crowd out healthy ones."""
    pages = {path: page for path, page in graph.pages.items() if page.current_eligible}
    edges = tuple(e for e in graph.edges if e.source in pages and e.target in pages)
    return replace(graph, pages=MappingProxyType(pages), edges=edges,
                   outbound=MappingProxyType({p: tuple(t for t in graph.outbound[p] if t in pages)
                                              for p in pages}),
                   inbound=MappingProxyType({p: tuple(t for t in graph.inbound[p] if t in pages)
                                             for p in pages}))


def page_diagnostics(graph):
    return [{'path': p.path, 'status': p.publication_status, 'eligible': p.current_eligible,
             'orphaned': p.orphaned, 'orphaned_reason': p.orphaned_reason,
             'reasons': list(p.status_reasons), 'source_statuses': dict(p.source_statuses)}
            for p in graph.pages.values()]
