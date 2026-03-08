# PugBrain Installer for Windows (PowerShell)
# Usage:
#   irm https://raw.githubusercontent.com/tannht/pug-brain/pug-master/install.ps1 | iex
#   OR
#   .\install.ps1

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# ─── Helper functions ─────────────────────────────────────────────────────────
function Write-Info  { param([string]$Msg) Write-Host "[PugBrain] $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host "[PugBrain] $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "[PugBrain] $Msg" -ForegroundColor Yellow }
function Write-Fail  { param([string]$Msg) Write-Host "[PugBrain] $Msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  PugBrain Installer (Windows PowerShell)" -NoNewline -ForegroundColor White
Write-Host ""
Write-Host "  OS: Windows | Arch: $([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)" -ForegroundColor DarkGray
Write-Host ""

# ─── 1. Check prerequisites ──────────────────────────────────────────────────

# --- Git ---
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "Git is not installed. Download from: https://git-scm.com/download/win"
}

# --- Node.js (optional for dashboard) ---
$HAS_NODE = $false
if (Get-Command node -ErrorAction SilentlyContinue) {
    $nodeVersion = & node --version 2>$null
    Write-Info "Node.js: $nodeVersion"
    $HAS_NODE = $true
} else {
    Write-Warn "Node.js not installed. Dashboard will NOT be built."
    Write-Warn "Install Node.js v18+ for dashboard support: https://nodejs.org"
}

# --- Python ---
$PYTHON_CMD = $null
foreach ($cmd in @("python3", "python", "py")) {
    try {
        $result = & $cmd -c "import sys; print(sys.version_info.major)" 2>$null
        if ($result -eq "3") {
            $PYTHON_CMD = $cmd
            break
        }
    } catch {
        continue
    }
}

# Try py launcher with -3 flag (Windows Python Launcher)
if (-not $PYTHON_CMD) {
    try {
        $result = & py -3 -c "import sys; print(sys.version_info.major)" 2>$null
        if ($result -eq "3") {
            $PYTHON_CMD = "py -3"
        }
    } catch {}
}

if (-not $PYTHON_CMD) {
    Write-Fail "Python 3 is not installed. Download from: https://www.python.org/downloads/"
}

$pyVersion = & $PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
Write-Info "Python: $PYTHON_CMD ($pyVersion)"

# Check Python version >= 3.11
$versionCheck = & $PYTHON_CMD -c "import sys; print('ok' if (sys.version_info.major, sys.version_info.minor) >= (3, 11) else 'fail')" 2>$null
if ($versionCheck -ne "ok") {
    Write-Fail "Python version must be 3.11+. Current: $pyVersion. Download from: https://www.python.org/downloads/"
}

# --- pip ---
$PIP_CMD = $null
foreach ($cmd in @("pip3", "pip")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $PIP_CMD = $cmd
        break
    }
}

# Fallback: python -m pip
if (-not $PIP_CMD) {
    try {
        & $PYTHON_CMD -m pip --version 2>$null | Out-Null
        $PIP_CMD = "$PYTHON_CMD -m pip"
    } catch {}
}

if (-not $PIP_CMD) {
    Write-Fail "pip is not installed. Run: $PYTHON_CMD -m ensurepip --upgrade"
}

Write-Info "pip: $PIP_CMD"

# ─── 2. Setup install directory ───────────────────────────────────────────────
$INSTALL_DIR = Join-Path $env:USERPROFILE ".pugbrain\workspace\pug-brain"
Write-Info "Install directory: $INSTALL_DIR"

$parentDir = Split-Path $INSTALL_DIR -Parent
if (-not (Test-Path $parentDir)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

# ─── 3. Clone or update repository ───────────────────────────────────────────
$REPO_URL = "https://github.com/tannht/pug-brain.git"
$BRANCH = "pug-master"

if (Test-Path (Join-Path $INSTALL_DIR ".git")) {
    Write-Info "Repository already exists. Updating..."
    Push-Location $INSTALL_DIR
    try {
        & git fetch origin $BRANCH 2>$null
        & git checkout $BRANCH 2>$null
        & git pull origin $BRANCH 2>$null
    } catch {
        Write-Warn "git update failed (offline?)"
    }
    Pop-Location
} elseif (Test-Path $INSTALL_DIR) {
    Write-Warn "Directory exists but is not a git repo. Backing up..."
    $backupName = "${INSTALL_DIR}.bak.$(Get-Date -Format 'yyyyMMddHHmmss')"
    Rename-Item $INSTALL_DIR $backupName
    Write-Info "Cloning PugBrain repository..."
    & git clone -b $BRANCH $REPO_URL $INSTALL_DIR
} else {
    Write-Info "Cloning PugBrain repository..."
    & git clone -b $BRANCH $REPO_URL $INSTALL_DIR
}

Push-Location $INSTALL_DIR

# ─── 4. Install Python package ───────────────────────────────────────────────
Write-Info "Installing PugBrain Python core..."

try {
    if ($PIP_CMD -match " ") {
        # Handle "python -m pip" style commands
        $parts = $PIP_CMD -split " "
        & $parts[0] $parts[1..$parts.Length] install --upgrade --user ".[server]"
    } else {
        & $PIP_CMD install --upgrade --user ".[server]"
    }
} catch {
    Write-Warn "Full install failed. Trying without [server] extras..."
    try {
        if ($PIP_CMD -match " ") {
            $parts = $PIP_CMD -split " "
            & $parts[0] $parts[1..$parts.Length] install --upgrade --user "."
        } else {
            & $PIP_CMD install --upgrade --user "."
        }
    } catch {
        Write-Fail "pip install failed. Check Python/pip installation."
    }
}

# ─── 5. Install & Build Dashboard (optional) ─────────────────────────────────
if ($HAS_NODE -and (Test-Path (Join-Path $INSTALL_DIR "dashboard"))) {
    Write-Info "Installing Node.js dashboard dependencies..."
    Push-Location (Join-Path $INSTALL_DIR "dashboard")
    try {
        & npm install 2>$null
        Write-Info "Building PugBrain Dashboard UI..."
        & npm run build 2>$null
    } catch {
        Write-Warn "Dashboard build failed. Build later with: cd dashboard; npm run build"
    }
    Pop-Location
} else {
    if (-not $HAS_NODE) {
        Write-Warn "Skipping dashboard build (Node.js not installed)."
    }
}

# ─── 6. Verify installation ──────────────────────────────────────────────────
Write-Info "Verifying installation..."

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

$PUG_FOUND = $false
if (Get-Command pug -ErrorAction SilentlyContinue) {
    $PUG_FOUND = $true
} elseif (Get-Command pugbrain -ErrorAction SilentlyContinue) {
    $PUG_FOUND = $true
} else {
    try {
        & $PYTHON_CMD -m neural_memory --version 2>$null | Out-Null
        $PUG_FOUND = $true
    } catch {}
}

Write-Host ""
if ($PUG_FOUND) {
    Write-Ok "Installation complete! Gau gau!"
    Write-Host ""
    Write-Host "  Available commands:" -ForegroundColor White
    Write-Host "    pug         - PugBrain CLI (main command)"
    Write-Host "    pugbrain    - PugBrain CLI (alias)"
    Write-Host "    pug-mcp     - PugBrain MCP server"
    Write-Host ""
    Write-Host "  Quick start:" -ForegroundColor White
    Write-Host "    pug status              - Check brain status"
    Write-Host "    pug remember `"Hello`"    - Store a memory"
    Write-Host "    pug recall `"Hello`"      - Recall memories"
    Write-Host "    pug serve               - Start web dashboard"
    Write-Host ""
    Write-Host "  Install directory: $INSTALL_DIR" -ForegroundColor DarkGray
} else {
    Write-Warn "pug command not found in PATH."
    Write-Host ""
    Write-Warn "Installation may have succeeded but the CLI is not in your PATH."
    Write-Host ""
    Write-Host "  Try these fixes:" -ForegroundColor White

    # Find Python Scripts directory
    $scriptsDir = & $PYTHON_CMD -c "import site; print(site.getusersitepackages().replace('site-packages','Scripts'))" 2>$null
    Write-Host "    1. Add to PATH:  `$env:Path += `";$scriptsDir`""
    Write-Host "    2. Or restart your terminal / PowerShell"
    Write-Host "    3. Or run:  $PYTHON_CMD -m neural_memory --help"
    Write-Host ""
    Write-Host "  To permanently add to PATH:" -ForegroundColor White
    Write-Host "    [Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$scriptsDir', 'User')"
    Write-Host ""
    Write-Host "  Install directory: $INSTALL_DIR" -ForegroundColor DarkGray
}

Pop-Location
