param(
    [string]$Config = "configs/stage3_t2e.yaml",
    [string]$Gpu = "configs/gpu/rtx3090_4gpu.yaml",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts/launch.py --config $Config --gpu $Gpu @Rest
