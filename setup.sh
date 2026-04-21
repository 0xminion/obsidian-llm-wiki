#!/usr/bin/env bash
# ============================================================================
# setup.sh — One-command setup for obsidian-llm-wiki
# ============================================================================
# Usage: ./setup.sh [VAULT_PATH]
#   VAULT_PATH defaults to ~/MyVault
#
# What it does:
#   1. Creates vault directory structure
#   2. Copies scripts, lib, prompts, templates
#   3. Checks dependencies (required + optional)
#   4. Creates .env from .env.example if missing
#   5. Runs preflight validation
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_PATH="${1:-$HOME/MyVault}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; ERRORS=$((ERRORS + 1)); }

ERRORS=0

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  obsidian-llm-wiki setup                   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Vault path: $VAULT_PATH"
echo ""

# ─── Step 1: Create vault directories ───────────────────────────────────────

echo -e "${BOLD}[1/5] Creating vault structure...${NC}"

dirs=(
  "01-Raw"
  "02-Clippings"
  "03-Queries"
  "04-Wiki/sources"
  "04-Wiki/entries"
  "04-Wiki/concepts"
  "04-Wiki/mocs"
  "05-Outputs/answers"
  "05-Outputs/visualizations"
  "06-Config"
  "07-WIP"
  "08-Archive-Raw"
  "09-Archive-Queries"
  "Meta/Scripts"
  "Meta/Templates"
  "Meta/lib"
  "Meta/prompts"
)

created=0
for d in "${dirs[@]}"; do
  target="$VAULT_PATH/$d"
  if [ ! -d "$target" ]; then
    mkdir -p "$target"
    created=$((created + 1))
  fi
done

if [ "$created" -gt 0 ]; then
  ok "Created $created directories"
else
  ok "Vault directories already exist"
fi

# ─── Step 2: Copy files ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}[2/5] Copying pipeline files...${NC}"

# Scripts
cp_count=0
for f in "$SCRIPT_DIR"/scripts/*.sh; do
  [ -f "$f" ] || continue
  dest="$VAULT_PATH/Meta/Scripts/$(basename "$f")"
  if [ ! -f "$dest" ] || ! diff -q "$f" "$dest" &>/dev/null; then
    cp "$f" "$dest"
    cp_count=$((cp_count + 1))
  fi
done
chmod +x "$VAULT_PATH/Meta/Scripts/"*.sh 2>/dev/null || true

# Clean up any legacy scripts from vault (from pre-v2.2.0 installs)
for f in process-inbox.sh stage1-extract.sh stage2-plan.sh stage3-create.sh reindex.sh build_batch_prompt.py; do
  [ -f "$VAULT_PATH/Meta/Scripts/$f" ] && rm "$VAULT_PATH/Meta/Scripts/$f" && cp_count=$((cp_count + 1))
done

# Lib
for f in "$SCRIPT_DIR"/lib/*.sh "$SCRIPT_DIR"/lib/*.py; do
  [ -f "$f" ] || continue
  dest="$VAULT_PATH/Meta/lib/$(basename "$f")"
  if [ ! -f "$dest" ] || ! diff -q "$f" "$dest" &>/dev/null; then
    cp "$f" "$dest"
    cp_count=$((cp_count + 1))
  fi
done

# Prompts
for f in "$SCRIPT_DIR"/prompts/*.prompt; do
  [ -f "$f" ] || continue
  dest="$VAULT_PATH/Meta/prompts/$(basename "$f")"
  if [ ! -f "$dest" ] || ! diff -q "$f" "$dest" &>/dev/null; then
    cp "$f" "$dest"
    cp_count=$((cp_count + 1))
  fi
done

# Templates
for f in "$SCRIPT_DIR"/templates/*.md; do
  [ -f "$f" ] || continue
  dest="$VAULT_PATH/Meta/Templates/$(basename "$f")"
  if [ ! -f "$dest" ] || ! diff -q "$f" "$dest" &>/dev/null; then
    cp "$f" "$dest"
    cp_count=$((cp_count + 1))
  fi
done

if [ "$cp_count" -gt 0 ]; then
  ok "Copied/updated $cp_count files"
else
  ok "All files up to date"
fi

# ─── Step 3: Check dependencies ──────────────────────────────────────────────

echo ""
echo -e "${BOLD}[3/5] Checking dependencies...${NC}"

# Required
for cmd in bash jq curl python3; do
  if command -v "$cmd" &>/dev/null; then
    ok "$cmd: $(command -v "$cmd")"
  else
    fail "$cmd: NOT FOUND (required)"
  fi
done

# Agent (hermes)
if command -v hermes &>/dev/null; then
  ok "hermes: $(command -v hermes)"
else
  warn "hermes: NOT FOUND — needed for AI processing. Install from https://github.com/nicekate/hermes-agent"
fi

# Optional tools
for cmd in qmd yt-dlp ffmpeg defuddle ob; do
  if command -v "$cmd" &>/dev/null; then
    ok "$cmd (optional): $(command -v "$cmd")"
  else
    warn "$cmd (optional): not found"
  fi
done

# ─── Step 4: Set up .env ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}[4/5] Environment config...${NC}"

if [ ! -f "$VAULT_PATH/Meta/Scripts/.env" ]; then
  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$VAULT_PATH/Meta/Scripts/.env"
    ok "Created .env from .env.example — edit with your API keys"
  else
    # Create minimal .env
    cat > "$VAULT_PATH/Meta/Scripts/.env" << 'ENVEOF'
# API Keys (get from respective services)
TRANSCRIPT_API_KEY=
SUPADATA_API_KEY=
ASSEMBLYAI_API_KEY=

# Vault path
VAULT_PATH=$HOME/MyVault

# Agent (hermes is default)
AGENT_CMD=hermes

# Parallelism
PARALLEL=3
ENVEOF
    ok "Created .env template — edit with your API keys"
  fi
else
  ok ".env already exists"
fi

# ─── Step 5: Create convenience wrapper ──────────────────────────────────────

echo ""
echo -e "${BOLD}[5/5] Creating run wrapper...${NC}"

# Install Python package if setup.py/pyproject.toml exists
if [ -f "$SCRIPT_DIR/pyproject.toml" ] || [ -f "$SCRIPT_DIR/setup.py" ]; then
  if pip install -e "$SCRIPT_DIR" --quiet 2>/dev/null; then
    ok "Installed pipeline Python package (editable)"
  else
    warn "pip install failed — run.sh will use python3 -m pipeline.cli from repo"
  fi
fi

cat > "$VAULT_PATH/run.sh" << 'RUNEOF'
#!/usr/bin/env bash
# Wrapper: process inbox without remembering VAULT_PATH every time
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VAULT_PATH="$SCRIPT_DIR"

# Use Python pipeline (canonical). Fallback to shell if Python not available.
if command -v pipeline &>/dev/null; then
  exec pipeline ingest "$SCRIPT_DIR" "$@"
elif python3 -c "import pipeline.cli" 2>/dev/null; then
  exec python3 -m pipeline.cli ingest "$SCRIPT_DIR" "$@"
else
  echo "ERROR: Python pipeline not found. Install with: pip install -e /path/to/obsidian-llm-wiki" >&2
  exit 1
fi
RUNEOF
chmod +x "$VAULT_PATH/run.sh"
ok "Created $VAULT_PATH/run.sh"

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
if [ "$ERRORS" -gt 0 ]; then
  echo -e "${RED}Setup completed with $ERRORS error(s). Fix the missing dependencies above.${NC}"
  exit 1
fi

echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Setup complete!                             ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit your API keys:"
echo "     $VAULT_PATH/Meta/Scripts/.env"
echo ""
echo "  2. Add URLs to process:"
echo "     echo 'https://example.com/article' > $VAULT_PATH/01-Raw/my-source.url"
echo ""
echo "  3. Run the pipeline:"
echo "     cd $VAULT_PATH && ./run.sh"
echo ""
echo "  That's it. Drop URLs in 01-Raw/, run ./run.sh, check 04-Wiki/."
echo ""
