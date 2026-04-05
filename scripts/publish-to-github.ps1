# Publishes this repository to your GitHub account as: kokoro-tts-studio
#
# Prerequisites:
#   1. Install GitHub CLI: winget install GitHub.cli
#   2. Log in once: gh auth login
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\publish-to-github.ps1
#
# If `origin` already exists, only `git push -u origin main` runs.

$ErrorActionPreference = 'Stop'
$RepoName = 'kokoro-tts-studio'
$Description = 'Local TTS Studio: Kokoro-82M + XTTS-v2 in Gradio, bilingual EN/TR UI'

$ghExe = Get-Command gh -ErrorAction SilentlyContinue
if (-not $ghExe) {
    $fallback = 'C:\Program Files\GitHub CLI\gh.exe'
    if (Test-Path $fallback) { $ghExe = @{ Source = $fallback } }
    else { throw 'GitHub CLI not found. Install: winget install GitHub.cli' }
} else {
    $ghExe = @{ Source = $ghExe.Source }
}

& $ghExe.Source auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Run: gh auth login' -ForegroundColor Yellow
    exit 1
}

$root = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $root

$hasOrigin = $false
git remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) { $hasOrigin = $true }

if ($hasOrigin) {
    Write-Host "Remote origin present. Pushing main..."
    git push -u origin main
} else {
    & $ghExe.Source repo create $RepoName --public --description $Description --source=. --remote=origin --push
}
