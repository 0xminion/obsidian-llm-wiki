"""Stage 3 — Create: batch creation of vault files via parallel agents.

This package is split into focused sub-modules:
  - prompts:    Prompt loading and batch prompt construction
  - agent:      Agent orchestration, concept convergence, batch execution
  - templates:  Template-based file creation (deterministic + optional agent insights)
  - validate:   Output validation and auto-repair
  - orchestrator: Main entry point (create_all) and post-processing
"""

# Re-export everything for backward compatibility
from pipeline.utils import strip_qmd_noise as _strip_qmd_noise
from pipeline.create.prompts import _load_prompt, build_batch_prompt
from pipeline.create.agent import _run_agent, concept_convergence, create_batch
from pipeline.create.templates import (
    generate_source_content,
    generate_entry_content,
    generate_entry_insights,
    create_file_templates,
    _generate_concept_template,
)
from pipeline.create.validate import validate_output, _repair_violations
from pipeline.create.orchestrator import create_all

__all__ = [
    "_strip_qmd_noise",
    "_load_prompt",
    "build_batch_prompt",
    "_run_agent",
    "concept_convergence",
    "create_batch",
    "generate_source_content",
    "generate_entry_content",
    "generate_entry_insights",
    "create_file_templates",
    "_generate_concept_template",
    "validate_output",
    "_repair_violations",
    "create_all",
]
