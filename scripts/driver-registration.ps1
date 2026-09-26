param(
    [ValidateSet('Status', 'Register', 'Unregister')][string]$Action = 'Status',
    [Parameter(Mandatory=$true)][string]$SteamVR,
    [string]$Driver = (Join-Path $PSScriptRoot '../build/myumiq')
)
$ErrorActionPreference = 'Stop'
$driverPath = (Resolve-Path -LiteralPath $Driver).Path.TrimEnd('\', '/')
$manifest = Get-Content -LiteralPath (Join-Path $driverPath 'driver.vrdrivermanifest') -Raw | ConvertFrom-Json
if ($manifest.name -cne 'myumiq') { throw 'Expected the MyuMIQ driver manifest.' }
$tool = Join-Path $SteamVR 'bin/win64/vrpathreg.exe'
if (-not (Test-Path -LiteralPath $tool)) { throw "SteamVR vrpathreg not found: $tool" }
# The installed tool remains the authority for its syntax and registration result.
$found = @(& $tool finddriver myumiq)
$findExit = $LASTEXITCODE
if ($findExit -notin 0,1) { throw "finddriver failed ($findExit): $found" }
if ($findExit -eq 0) {
    $foundPath = ($found -join "`n").Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $foundPath)) { throw "Unrecognized finddriver output: $foundPath" }
    if (Test-Path -LiteralPath $foundPath -PathType Leaf) {
        if ((Split-Path -Leaf $foundPath) -ine 'driver.vrdrivermanifest') { throw 'Unexpected finddriver file.' }
        $foundPath = Split-Path -Parent $foundPath
    }
    if ((Resolve-Path -LiteralPath $foundPath).Path.TrimEnd('\','/') -ine $driverPath) {
        throw "A different MyuMIQ driver is registered at $foundPath; no changes made."
    }
}
if ($Action -eq 'Status') { Write-Output "MyuMIQ registered: $($findExit -eq 0); target: $driverPath"; return }
if (Get-Process vrserver,vrcompositor,VRChat -ErrorAction SilentlyContinue) { throw 'Close SteamVR and VRChat before changing driver registration.' }
if ($Action -eq 'Register' -and $findExit -eq 1) {
    if (-not (Test-Path -LiteralPath (Join-Path $driverPath 'bin/win64/driver_myumiq.dll'))) { throw 'Build the driver first.' }
    & $tool adddriver $driverPath
    if ($LASTEXITCODE -ne 0) { throw 'adddriver failed' }
} elseif ($Action -eq 'Unregister' -and $findExit -eq 0) {
    & $tool removedriver $driverPath
    if ($LASTEXITCODE -ne 0) { throw 'removedriver failed' }
}
$after = @(& $tool finddriver myumiq)
$afterExit = $LASTEXITCODE
$expected = if ($Action -eq 'Register') { 0 } else { 1 }
if ($afterExit -ne $expected) { throw "Registration verification failed: $after" }
Write-Output "$Action verified for $driverPath"
