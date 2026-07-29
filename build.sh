#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

step() { echo; echo "▶ $*"; }
ok()   { echo "  ✓ $*"; }

require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' is required but not found in PATH." >&2
        exit 1
    fi
}

step "Checking prerequisites"
require_cmd python
require_cmd dotnet
ok "python $(python --version)"
ok "dotnet  $(dotnet --version)"

step "Python - creating virtual environment"
if [[ -d "$VENV_DIR" ]]; then
    echo "  WARNING: venv already exists at $VENV_DIR - skipping creation"
else
    python -m venv "$VENV_DIR"
    ok "venv created at $VENV_DIR"
fi

step "Python - installing dependencies"
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip

if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    # agent-framework-mlx pins agent-framework-core==1.3.0 in its own metadata, which
    # conflicts with the newer agent-framework pinned in requirements.txt. Install
    # everything else normally, then install agent-framework-mlx with --no-deps (its
    # real runtime deps, mlx/mlx-lm, are already listed in requirements.txt).
    grep -v -E '^agent-framework-mlx==' "$REPO_ROOT/python/requirements.txt" \
        | pip install --quiet -r /dev/stdin
    pip install --quiet --no-deps agent-framework-mlx==0.6.0
    ok "installed full requirements (including agent-framework-mlx)"
else
    echo "  (non-Apple-Silicon platform detected - skipping agent-framework-mlx)"
    grep -v -E '^(agent-framework-mlx==|mlx==|mlx-lm==)' "$REPO_ROOT/python/requirements.txt" \
        | pip install --quiet -r /dev/stdin
    ok "installed requirements (agent-framework-mlx excluded)"
fi

step "Python - syntax check"
find "$REPO_ROOT/python" -name "*.py" -not -path "$VENV_DIR/*" \
    | xargs python -m py_compile
ok "all .py files are syntactically valid"

deactivate

# ─── .NET ────────────────────────────────────────────────────────────────────

step ".NET - build"
dotnet build "$REPO_ROOT/dotnet/HybridAgentDemos.slnx" \
    --configuration Release \
    --nologo \
    -v minimal
ok "solution built successfully"

# ─────────────────────────────────────────────────────────────────────────────

echo
echo "✅  All checks passed."
