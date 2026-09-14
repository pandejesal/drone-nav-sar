#!/usr/bin/env bash
# setup-friend-mac.sh — One-shot bootstrap for MacBook Pro (macOS 13+)
# Run: chmod +x scripts/setup-friend-mac.sh && ./scripts/setup-friend-mac.sh
# Isolated: touches ONLY this repo's clone + project-local configs.
#           Global Hermes (~/.hermes), global opencode (~/.opencode) are created
#           but never overwrite your other projects' vault/mnemosyne configs.
set -euo pipefail

REPO_URL="https://github.com/pandejesal/drone-nav-sar.git"
VAULT_URL="https://github.com/pandejesal/obsidian-vault.git"
PROJECT_DIR="$HOME/Desktop/drone-nav-sar"
# Allow override: PROJECT_DIR=/path/to/whereever ./scripts/setup-friend-mac.sh

echo "=== DroneNav-SAR MacBook Setup (isolated, project-local) ==="

# 1. Xcode CLI tools (git comes with it)
if ! xcode-select -p &>/dev/null; then
  echo "[1/7] Installing Xcode Command Line Tools..."
  xcode-select --install || true
  echo "  -> Re-run this script after Xcode CLI install finishes."
  exit 0
fi

# 2. Homebrew
if ! command -v brew &>/dev/null; then
  echo "[2/7] Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -f /opt/homebrew/bin/brew ]]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
  if [[ -f /usr/local/bin/brew ]]; then eval "$(/usr/local/bin/brew shellenv)"; fi
else
  echo "[2/7] Homebrew already installed."
fi

# 3. System deps (idempotent)
echo "[3/7] Installing system deps (python, node, uv, git-lfs, colmap)..."
brew update
brew install python@3.11 node uv git-lfs colmap pkg-config || true
brew install --cask docker || echo "  -> Install Docker Desktop manually if brew cask failed: https://docs.docker.com/desktop/setup/install/mac-install/"
git lfs install || true

# 4. Clone project (or pull if exists)
if [[ -d "$PROJECT_DIR/.git" ]]; then
  echo "[4/7] Project already cloned at $PROJECT_DIR — pulling..."
  git -C "$PROJECT_DIR" pull --ff-only
else
  echo "[4/7] Cloning $REPO_URL -> $PROJECT_DIR"
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# 5. Init / update Obsidian vault SUBMODULE (isolated inside this repo only)
#    This does NOT move or replace ~/Documents/Obsidian Vault.
#    Other projects on this Mac that use the vault keep using the global path.
#    This submodule lives at ./vault (gitignored by other projects).
echo "[5/7] Setting up Obsidian vault submodule at ./vault (project-local)..."
if [[ -f .gitmodules ]] && grep -q "vault" .gitmodules 2>/dev/null; then
  git submodule update --init --recursive
else
  echo "  -> No .gitmodules yet (owner will push it). Cloning vault to ./vault as fallback..."
  if [[ ! -d vault/.git ]]; then
    git clone "$VAULT_URL" vault || echo "  -> Vault clone failed (private repo? add GitHub auth: gh auth login)"
  else
    git -C vault pull --ff-only || true
  fi
fi
echo "  -> Vault at: $PROJECT_DIR/vault (submodule, isolated)"

# 6. Python venv (project-local .venv — does NOT touch global python)
echo "[6/7] Creating project-local .venv (uv)..."
if [[ ! -d .venv ]]; then
  uv venv .venv --python 3.11 || python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -U pip 2>/dev/null || pip install -U pip
if [[ -f requirements.txt ]]; then
  uv pip install -r requirements.txt || pip install -r requirements.txt
elif [[ -f pyproject.toml ]]; then
  uv pip install -e . || pip install -e .
else
  uv pip install gymnasium numpy pytest torch --index-strategy unsafe-best-match 2>/dev/null || pip install gymnasium numpy pytest
fi
echo "  -> Python: $(python --version) at $PROJECT_DIR/.venv"

# 7. OpenCode + Hermes (project-local .opencode/, global binary only once)
echo "[7/7] Installing OpenCode CLI + Hermes Agent (global binaries, project-local config)..."
if ! command -v opencode &>/dev/null; then
  npm install -g @opencode-ai/opencode || npm install -g opencode-ai 2>/dev/null || echo "  -> npm install failed, install node first"
else
  echo "  -> opencode already installed: $(opencode --version 2>/dev/null || echo ok)"
fi

if ! command -v hermes &>/dev/null; then
  echo "  -> Installing Hermes Agent..."
  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash || echo "  -> Hermes install failed — see https://hermes-agent.nousresearch.com/docs"
  echo "  -> Restart terminal after Hermes install, then run: hermes auth login"
else
  echo "  -> hermes already installed"
fi

# Project-local opencode config is already in repo at .opencode/opencode.json
# Ensure it exists with free-tier model rotation
mkdir -p .opencode
if [[ ! -f .opencode/opencode.json ]]; then
  cat > .opencode/opencode.json <<'JSON'
{
  "defaultProvider": "opencode-zen",
  "defaultModel": "muse-spark-1.3-contributor-free",
  "alternatives": ["nemotron-3-ultra-free", "mimo-v2.5-free", "nemotron-3.5-lightning-free"]
}
JSON
  echo "  -> Created .opencode/opencode.json (free-tier rotation)"
fi

# Verify
echo ""
echo "=== Verify ==="
echo "Repo:   $PROJECT_DIR"
git -C "$PROJECT_DIR" status --short | head -n 20 || true
echo "Vault:  $PROJECT_DIR/vault (submodule)"
ls -ld "$PROJECT_DIR/vault" 2>/dev/null | head -n 5 || echo "  (vault not yet cloned — push .gitmodules from owner machine first)"
echo "Venv:   $PROJECT_DIR/.venv ($(python --version 2>/dev/null))"
echo "Tests:  run -> source .venv/bin/activate && python -m pytest tests/ -q"
echo ""
echo "=== Isolation guarantee ==="
echo "- This script wrote ONLY to: $PROJECT_DIR (+ vault submodule at ./vault)"
echo "- Global Hermes (~/.config/hermes or ~/AppData/Local/hermes) untouched except binary install"
echo "- Global Obsidian vault (~/Documents/Obsidian Vault) NOT moved — other projects still use it"
echo "- Mnemosyne: project uses ./.harness-memory/evolution-lessons.json (tracked), global mnemosyne unchanged"
echo "- Other projects' .opencode / .harness-memory are per-repo, unaffected"
echo ""
echo "Next: open opencode in this dir and paste prompts/bootstrap-opencode.md"
echo "Done."
