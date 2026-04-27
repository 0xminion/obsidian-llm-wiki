"""Watch mode for automatic recompilation on vault changes."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pipeline.config import Config
from pipeline.store import ContentStore

log = logging.getLogger(__name__)


def watch_compile(cfg: Config, incremental: bool = True, bidirectional: bool = True) -> None:
    """Auto-trigger compile on file changes.

    Uses watchdog if available, otherwise falls back to polling every 30s.
    """
    import importlib.util
    if importlib.util.find_spec("watchdog") is not None:
        _watch_with_watchdog(cfg, incremental=incremental, bidirectional=bidirectional)
    else:
        log.info("watchdog not installed; falling back to 30s polling")
        _watch_with_polling(cfg, incremental=incremental, bidirectional=bidirectional)


def _watch_with_watchdog(cfg: Config, incremental: bool = True, bidirectional: bool = True) -> None:
    from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]
    from watchdog.observers import Observer  # type: ignore[import-untyped]

    from pipeline.compile.core import IncrementalCompiler, _compiling, run_compile

    class _CompileHandler(FileSystemEventHandler):
        def __init__(self, inc: IncrementalCompiler) -> None:
            self.inc = inc
            self._last_compile = 0.0

        def on_modified(self, event) -> None:
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix != ".md":
                return
            self._maybe_compile()

        def on_created(self, event) -> None:
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix != ".md":
                return
            self._maybe_compile()

        def _maybe_compile(self) -> None:
            if _compiling.is_set():
                return
            now = time.time()
            if now - self._last_compile < 5.0:
                return
            self._last_compile = now
            log.info("Watchdog: file change detected, triggering compile...")
            try:
                run_compile(cfg, incremental=incremental, full=False, bidirectional=bidirectional, watch_mode=True)
            except Exception:
                log.exception("Watch compile failed")

    with ContentStore.open_vault_cache(cfg.vault_path) as store:
        handler = _CompileHandler(IncrementalCompiler(store))
        observer = Observer()
        for d in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
            if d.exists():
                observer.schedule(handler, str(d), recursive=False)
        observer.start()
        log.info("Watchdog watching vault: %s", cfg.vault_path)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()


def _watch_with_polling(cfg: Config, incremental: bool = True, bidirectional: bool = True, interval: float = 30.0) -> None:
    from pipeline.compile.core import IncrementalCompiler, _compiling, run_compile

    with ContentStore.open_vault_cache(cfg.vault_path) as store:
        inc = IncrementalCompiler(store)
        log.info("Polling watch started (every %.0fs): %s", interval, cfg.vault_path)
        try:
            while True:
                time.sleep(interval)
                changed, current = inc.get_files_for_compile(cfg, full=False)
                if changed:
                    log.info("Polling: %d changed file(s), triggering compile...", len(changed))
                    if _compiling.is_set():
                        log.debug("Polling: compile already in progress, skipping")
                    else:
                        try:
                            run_compile(cfg, incremental=incremental, full=False, bidirectional=bidirectional, watch_mode=True)
                        except Exception:
                            log.exception("Watch compile failed")
                else:
                    log.debug("Polling: no changes")
        except KeyboardInterrupt:
            pass
