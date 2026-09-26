param(
    [string]$InputDirectory = (Join-Path $PSScriptRoot '../driver/resources/input'),
    [string]$ActionManifest
)
$ErrorActionPreference = 'Stop'
$profile = Get-Content -LiteralPath (Join-Path $InputDirectory 'controller_profile.json') -Raw | ConvertFrom-Json
$binding = Get-Content -LiteralPath (Join-Path $InputDirectory 'vrchat_bindings.json') -Raw | ConvertFrom-Json
if ($binding.controller_type -cne $profile.controller_type) { throw 'Controller type mismatch' }
if ($binding.app_key -cne 'steam.app.438100') { throw 'Unexpected application key' }
if ($profile.default_bindings.Count -ne 1 -or $profile.default_bindings[0].binding_url -cne 'vrchat_bindings.json') { throw 'Default binding reference mismatch' }
$poses = @($binding.bindings.'/actions/global'.poses)
foreach ($hand in @('left','right')) {
    if (@($poses | Where-Object { $_.path -ceq "/user/hand/$hand/pose/raw" -and $_.output -ceq '/actions/global/in/pose' }).Count -ne 1) { throw "Missing or duplicate $hand pose binding" }
}
$outputs = @($poses.output)
foreach ($set in $binding.bindings.PSObject.Properties) {
    foreach ($source in $set.Value.sources) {
        $relativePath = $source.path -replace '^/user/hand/(left|right)', ''
        if ($source.path -notmatch '^/user/hand/(left|right)/input/' -or $relativePath -notin $profile.input_source.PSObject.Properties.Name) { throw "Unsupported source $($source.path)" }
        $outputs += @($source.inputs.PSObject.Properties | ForEach-Object { $_.Value.output })
    }
}
if ($ActionManifest) {
    $manifest = Get-Content -LiteralPath $ActionManifest -Raw | ConvertFrom-Json
    foreach ($output in $outputs) {
        if ($output -notin $manifest.actions.name) { throw "Action absent from manifest: $output" }
    }
}
Write-Output 'Default binding references, controller sources and both hand poses passed.'
