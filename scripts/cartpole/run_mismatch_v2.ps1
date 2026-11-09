param([ValidateSet('smoke','benchmark','full','report')][string]$Mode='smoke', [string]$RunDir='', [string]$Stages='all')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$arguments=@((Join-Path $PSScriptRoot 'run_mismatch_v2.py'),'--mode',$Mode,'--stages',$Stages,'--device','cuda')
if ($RunDir) { $arguments+=@('--run-dir',$RunDir) }
& (Join-Path $projectRoot '.venv/Scripts/python.exe') @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
