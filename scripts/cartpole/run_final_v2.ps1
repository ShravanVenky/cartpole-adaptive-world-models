param(
    [ValidateSet('smoke','benchmark','full','report')][string]$Mode = 'smoke',
    [string]$RunDir = ''
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$RunnerPath = Join-Path $PSScriptRoot 'run_final_v2.py'
$RunnerArgs = @($RunnerPath, '--mode', $Mode, '--device', 'cuda')
if ($RunDir) { $RunnerArgs += @('--run-dir', $RunDir) }
Push-Location -LiteralPath $ProjectRoot
try {
    & $PythonPath @RunnerArgs
    if ($LASTEXITCODE -ne 0) { throw "Final V2 runner failed with exit code $LASTEXITCODE. Inspect saved status.json before resuming." }
} finally { Pop-Location }
