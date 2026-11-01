"""Resolve the current generation without replacing a human Obsidian homepage."""

import json
from urllib.parse import quote

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.immutable_publication import current_publication


@app.command("current")
def current(vault: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")):
    """Print the validated current entrypoint and Obsidian open URI; never write."""
    _, config = resolve_vault(vault)
    try:
        snapshot = current_publication(config.wiki_dir)
        if snapshot is None:
            raise RuntimeError("no committed immutable generation; run build first")
        _, manifest = snapshot
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(json.dumps({"error": str(exc)}) if as_json else str(exc))
        raise typer.Exit(code=1) from exc
    entrypoint = manifest["entrypoint"]
    result = {
        "generation": manifest["generation"],
        "entrypoint": entrypoint,
        "obsidian_uri": "obsidian://open?path=" + quote(str(config.vault / entrypoint), safe=""),
        "native_ui_verified": False,
        "warnings": manifest["errors"],
    }
    typer.echo(json.dumps(result, ensure_ascii=False) if as_json else result["obsidian_uri"])
