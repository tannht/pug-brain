#!/usr/bin/env bash
# PugBrain One-Line Installer
# Compatible with: Linux, macOS, Windows (Git Bash / MSYS2 / WSL / Cygwin)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/tannht/pug-brain/pug-master/install.sh | bash
#
# Windows native (PowerShell):
#   irm https://raw.githubusercontent.com/tannht/pug-brain/pug-master/install.ps1 | iex

set -euo pipefail

# ─── Colors (safe for non-color terminals) ────────────────────────────────────
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
    CYAN=$(tput setaf 6); BOLD=$(tput bold); RESET=$(tput sgr0)
else
    RED=""; GREEN=""; YELLOW=""; CYAN=""; BOLD=""; RESET=""
fi

info()  { echo "${CYAN}[PugBrain]${RESET} $*"; }
ok()    { echo "${GREEN}[PugBrain]${RESET} $*"; }
warn()  { echo "${YELLOW}[PugBrain]${RESET} $*"; }
fail()  { echo "${RED}[PugBrain]${RESET} $*" >&2; exit 1; }

# ─── Detect OS ────────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s 2>/dev/null || echo unknown)" in
        Linux*)     echo "linux"   ;;
        Darwin*)    echo "macos"   ;;
        CYGWIN*)    echo "windows" ;;
        MINGW*)     echo "windows" ;;
        MSYS*)      echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m 2>/dev/null || echo unknown)" in
        x86_64|amd64)   echo "x86_64"  ;;
        arm64|aarch64)  echo "arm64"   ;;
        *)              echo "$(uname -m 2>/dev/null || echo unknown)" ;;
    esac
}

OS="$(detect_os)"
ARCH="$(detect_arch)"

echo ""
echo "${BOLD}  PugBrain Installer${RESET}"
echo "  OS: ${OS} | Arch: ${ARCH}"
echo ""

# ─── 1. Check prerequisites ──────────────────────────────────────────────────

# --- Git ---
if ! command -v git >/dev/null 2>&1; then
    fail "Git is not installed. Please install Git first.
    Linux:   sudo apt install git  (or yum, pacman, etc.)
    macOS:   xcode-select --install
    Windows: https://git-scm.com/download/win"
fi

# --- Node.js ---
if ! command -v node >/dev/null 2>&1; then
    warn "Node.js is not installed. Dashboard will NOT be built."
    warn "Install Node.js v18+ for dashboard support: https://nodejs.org"
    HAS_NODE=false
else
    NODE_VERSION="$(node --version 2>/dev/null || echo "v0")"
    info "Node.js: ${NODE_VERSION}"
    HAS_NODE=true
fi

# --- Python ---
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        # Verify it's actually Python 3
        if "$cmd" -c "import sys; sys.exit(0 if sys.version_info.major == 3 else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "Python 3 is not installed. Please install Python 3.11+ first.
    Linux:   sudo apt install python3 python3-pip
    macOS:   brew install python@3.12
    Windows: https://www.python.org/downloads/"
fi

PYTHON_VERSION="$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
info "Python: ${PYTHON_CMD} (${PYTHON_VERSION})"

# Check Python version >= 3.11 (using Python itself to avoid sort -V portability issues)
if ! $PYTHON_CMD -c "import sys; sys.exit(0 if (sys.version_info.major, sys.version_info.minor) >= (3, 11) else 1)" 2>/dev/null; then
    fail "Python version must be 3.11+. Current version: ${PYTHON_VERSION}
    Please upgrade Python: https://www.python.org/downloads/"
fi

# --- pip ---
PIP_CMD=""
for cmd in pip3 pip; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PIP_CMD="$cmd"
        break
    fi
done

# Fallback: try python -m pip
if [ -z "$PIP_CMD" ]; then
    if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
        PIP_CMD="$PYTHON_CMD -m pip"
    fi
fi

if [ -z "$PIP_CMD" ]; then
    fail "pip is not installed. Please install pip first.
    Linux:   sudo apt install python3-pip
    macOS:   python3 -m ensurepip --upgrade
    Windows: python -m ensurepip --upgrade"
fi

info "pip: ${PIP_CMD}"

# ─── 2. Setup install directory ───────────────────────────────────────────────
setup_install_dir() {
    case "$OS" in
        windows)
            # Windows (Git Bash/MSYS2/Cygwin): prefer USERPROFILE, fallback HOME
            local win_home="${USERPROFILE:-${HOME:-}}"
            if [ -z "$win_home" ]; then
                fail "Cannot determine home directory. Set USERPROFILE or HOME."
            fi
            echo "${win_home}/.pugbrain/workspace/pug-brain"
            ;;
        *)
            # Linux / macOS / unknown
            echo "${HOME:?HOME not set}/.pugbrain/workspace/pug-brain"
            ;;
    esac
}

INSTALL_DIR="$(setup_install_dir)"
info "Install directory: ${INSTALL_DIR}"

# Create parent directories
mkdir -p "$(dirname "$INSTALL_DIR")"

# ─── 3. Clone or update repository ───────────────────────────────────────────
REPO_URL="https://github.com/tannht/pug-brain.git"
BRANCH="pug-master"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Repository already exists. Updating..."
    cd "$INSTALL_DIR" || fail "Cannot cd to ${INSTALL_DIR}"
    git fetch origin "$BRANCH" 2>/dev/null || warn "git fetch failed (offline?)"
    git checkout "$BRANCH" 2>/dev/null || true
    git pull origin "$BRANCH" 2>/dev/null || warn "git pull failed (offline?)"
elif [ -d "$INSTALL_DIR" ]; then
    # Directory exists but not a git repo — back up and re-clone
    warn "Directory exists but is not a git repo. Backing up..."
    mv "$INSTALL_DIR" "${INSTALL_DIR}.bak.$(date +%s)"
    info "Cloning PugBrain repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR" || fail "Cannot cd to ${INSTALL_DIR}"
else
    info "Cloning PugBrain repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR" || fail "Cannot cd to ${INSTALL_DIR}"
fi

# ─── 4. Install Python package ───────────────────────────────────────────────
info "Installing PugBrain Python core..."

# Build pip install flags
PIP_FLAGS="install --upgrade"

# macOS with externally-managed Python (Homebrew Python 3.13+)
if [ "$OS" = "macos" ]; then
    if $PIP_CMD install --help 2>&1 | grep -q -- "--break-system-packages"; then
        PIP_FLAGS="${PIP_FLAGS} --break-system-packages"
    fi
fi

# Linux with externally-managed Python (Debian 12+, Ubuntu 23.04+, Fedora 38+)
if [ "$OS" = "linux" ]; then
    if $PIP_CMD install --help 2>&1 | grep -q -- "--break-system-packages"; then
        PIP_FLAGS="${PIP_FLAGS} --break-system-packages"
    fi
fi

# Windows: ensure --user install to avoid permission issues
if [ "$OS" = "windows" ]; then
    PIP_FLAGS="${PIP_FLAGS} --user"
fi

# Run pip install
$PIP_CMD $PIP_FLAGS ".[server]" || {
    warn "Full install failed. Trying without [server] extras..."
    $PIP_CMD $PIP_FLAGS "." || fail "pip install failed. Check Python/pip installation."
}

# ─── 5. Install & Build Dashboard (optional, requires Node.js) ────────────────
if [ "$HAS_NODE" = true ] && [ -d "$INSTALL_DIR/dashboard" ]; then
    info "Installing Node.js dashboard dependencies..."
    (cd "$INSTALL_DIR/dashboard" && npm install 2>/dev/null) || warn "npm install failed. Dashboard may not work."

    if command -v npm >/dev/null 2>&1; then
        info "Building PugBrain Dashboard UI..."
        (cd "$INSTALL_DIR/dashboard" && npm run build 2>/dev/null) || warn "Dashboard build failed. You can build later with: cd dashboard && npm run build"
    fi
else
    if [ "$HAS_NODE" = false ]; then
        warn "Skipping dashboard build (Node.js not installed)."
    fi
fi

# ─── 6. Setup PATH hints ─────────────────────────────────────────────────────
add_path_hint() {
    local shell_name=""
    local rc_file=""

    case "$OS" in
        macos)
            if [ -f "$HOME/.zshrc" ]; then
                shell_name="zsh"; rc_file="$HOME/.zshrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                shell_name="bash"; rc_file="$HOME/.bash_profile"
            fi
            ;;
        linux)
            if [ -f "$HOME/.bashrc" ]; then
                shell_name="bash"; rc_file="$HOME/.bashrc"
            elif [ -f "$HOME/.zshrc" ]; then
                shell_name="zsh"; rc_file="$HOME/.zshrc"
            fi
            ;;
    esac

    if [ -n "$rc_file" ]; then
        # Check if pip --user bin is in PATH
        local user_bin=""
        user_bin="$($PYTHON_CMD -c 'import site; print(site.getusersitepackages().replace("lib/python3", "bin").split("/lib/")[0] + "/bin")' 2>/dev/null || echo "")"

        if [ -n "$user_bin" ] && [ -d "$user_bin" ]; then
            case "$PATH" in
                *"$user_bin"*) ;;  # already in PATH
                *)
                    warn "pip user bin directory not in PATH: ${user_bin}"
                    warn "Add this to your ${rc_file}:"
                    warn "  export PATH=\"${user_bin}:\$PATH\""
                    ;;
            esac
        fi
    fi
}

# ─── 7. Verify installation ──────────────────────────────────────────────────
info "Verifying installation..."

# Refresh PATH to pick up newly installed scripts
hash -r 2>/dev/null || true

# Try to find pug command
PUG_FOUND=false
if command -v pug >/dev/null 2>&1; then
    PUG_FOUND=true
elif command -v pugbrain >/dev/null 2>&1; then
    PUG_FOUND=true
elif $PYTHON_CMD -m neural_memory --version >/dev/null 2>&1; then
    PUG_FOUND=true
fi

echo ""
if [ "$PUG_FOUND" = true ]; then
    ok "Installation complete! Gau gau!"
    echo ""
    echo "  ${BOLD}Available commands:${RESET}"
    echo "    pug         - PugBrain CLI (main command)"
    echo "    pugbrain    - PugBrain CLI (alias)"
    echo "    pug-mcp     - PugBrain MCP server"
    echo ""
    echo "  ${BOLD}Quick start:${RESET}"
    echo "    pug status              - Check brain status"
    echo "    pug remember \"Hello\"    - Store a memory"
    echo "    pug recall \"Hello\"      - Recall memories"
    echo "    pug serve               - Start web dashboard"
    echo ""
    echo "  ${BOLD}Install directory:${RESET} ${INSTALL_DIR}"
else
    warn "pug command not found in PATH."
    add_path_hint
    echo ""
    warn "Installation may have succeeded but the CLI is not in your PATH."
    echo ""
    echo "  ${BOLD}Try these fixes:${RESET}"
    case "$OS" in
        macos)
            echo "    1. Add to PATH:  export PATH=\"\$HOME/Library/Python/${PYTHON_VERSION}/bin:\$PATH\""
            echo "    2. Or restart your terminal"
            echo "    3. Or run:  ${PYTHON_CMD} -m neural_memory --help"
            ;;
        linux)
            echo "    1. Add to PATH:  export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo "    2. Or restart your terminal"
            echo "    3. Or run:  ${PYTHON_CMD} -m neural_memory --help"
            ;;
        windows)
            echo "    1. Close and reopen Git Bash / terminal"
            echo "    2. Or add Python Scripts to PATH"
            echo "    3. Or run:  ${PYTHON_CMD} -m neural_memory --help"
            ;;
    esac
    echo ""
    echo "  ${BOLD}Install directory:${RESET} ${INSTALL_DIR}"
fi
