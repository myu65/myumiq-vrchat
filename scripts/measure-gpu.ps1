param(
    [ValidateRange(1,3600)][int]$Seconds = 60,
    [Parameter(Mandatory=$true)][string]$RunName,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../../myumiq-vrchat.local-artifacts')
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'Use a simple run name, e.g. baseline or vr-512-60.' }
$tool = (Get-Command nvidia-smi -ErrorAction Stop).Source
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = Join-Path $OutputDirectory "$RunName.csv"
if (Test-Path -LiteralPath $output) { throw "Refusing to overwrite $output" }
'timestamp,gpu_index,gpu_name,driver_version,gpu_util_percent,memory_used_mib,memory_total_mib' | Set-Content -LiteralPath $output -Encoding UTF8
for ($i = 0; $i -lt $Seconds; $i++) {
    $sample = & $tool '--query-gpu=timestamp,index,name,driver_version,utilization.gpu,memory.used,memory.total' '--format=csv,noheader,nounits'
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed; the CSV is incomplete.' }
    $sample | Add-Content -LiteralPath $output -Encoding UTF8
    if ($i -lt $Seconds - 1) { Start-Sleep -Seconds 1 }
}
Write-Output "Saved device-wide samples: $output"
