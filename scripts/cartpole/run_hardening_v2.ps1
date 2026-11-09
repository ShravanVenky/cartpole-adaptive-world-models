param(
    [ValidateSet('smoke','benchmark','full','report')][string]$Mode = 'smoke',
    [string]$Stages = 'A,B,C,D',
    [string]$RunDir = '',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskArgs = @((Join-Path $PSScriptRoot 'run_hardening_v2.py'), '--mode', $Mode, '--stages', $Stages, '--device', $Device)
if ($RunDir) { $taskArgs += @('--run-dir', $RunDir) }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
