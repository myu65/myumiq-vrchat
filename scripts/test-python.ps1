param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskArtifacts = Join-Path (Split-Path -Parent $taskRepo) 'myumiq-vrchat.local-artifacts\python-tests'
$taskRun = Join-Path $taskArtifacts ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N'))
$taskResolvedRun = [IO.Path]::GetFullPath($taskRun)
if (-not $taskResolvedRun.StartsWith([IO.Path]::GetFullPath($taskArtifacts) + '\') -or
    (Test-Path -LiteralPath $taskRun)) {
    throw 'Test artifacts must use a new directory beneath the parent artifact directory.'
}
New-Item -ItemType Directory -Path $taskRun | Out-Null
Push-Location $taskRepo
try {
    & $Python -m pytest -q --tb=short ('--basetemp=' + (Join-Path $taskRun 'temp')) `
        -o ('cache_dir=' + (Join-Path $taskRun 'cache')) `
        ('--junitxml=' + (Join-Path $taskRun 'pytest.xml'))
    if ($LASTEXITCODE -ne 0) { throw ('Python tests failed. Results: ' + $taskRun) }
    Write-Output ('Test results: ' + $taskRun)
} finally {
    Pop-Location
}
