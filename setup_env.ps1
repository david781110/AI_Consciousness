# AI Consciousness Project Environment Setup/Activation Script

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   AI Consciousness Project Environment Setup   " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

$VENV_PATH = "$PSScriptRoot\.venv"

if (Test-Path "$VENV_PATH") {
    Write-Host "Found virtual environment at $VENV_PATH" -ForegroundColor Green
    Write-Host "Activating environment..." -ForegroundColor Yellow
    & "$VENV_PATH\Scripts\Activate.ps1"
    Write-Host "Environment activated! You can now run your scripts." -ForegroundColor Green
    Write-Host "Example: python phase2_consciousness_test.py" -ForegroundColor Gray
} else {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please make sure you have run the setup process." -ForegroundColor Red
}

Write-Host "===============================================" -ForegroundColor Cyan
