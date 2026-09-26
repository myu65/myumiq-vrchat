param(
    [string]$Driver = (Join-Path $PSScriptRoot '../build/myumiq'),
    [ValidateRange(1, 30)][int]$Days = 7
)
# Read-only diagnostic. Emit JSON to stdout; redirect outside the repository.
$ErrorActionPreference = 'Stop'
$dll = (Resolve-Path -LiteralPath (Join-Path $Driver 'bin/win64/driver_myumiq.dll')).Path
$signature = Get-AuthenticodeSignature -LiteralPath $dll
$eventStatus = 'Available'
$events = @()
try {
    $events = @(Get-WinEvent -FilterHashtable @{
        LogName = 'Microsoft-Windows-CodeIntegrity/Operational'
        Id = 3077
        StartTime = (Get-Date).AddDays(-$Days)
    } -ErrorAction Stop | Where-Object {
        # Event file paths may use a device-volume prefix instead of a drive letter.
        $xml = [xml]$_.ToXml()
        $suffix = $dll.Substring([IO.Path]::GetPathRoot($dll).Length - 1)
        @($xml.Event.EventData.Data | Where-Object {
            $_.Name -in 'File Name', 'FileName' -and ([string]$_.'#text').Replace('\\', '\').EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase)
        }).Count -gt 0
    } | ForEach-Object {
        [ordered]@{ Time = $_.TimeCreated.ToString('o'); Id = $_.Id; Message = $_.Message }
    })
} catch {
    if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $eventStatus = 'No events in window' }
    else { $eventStatus = 'Unavailable: ' + $_.Exception.Message }
}
[ordered]@{
    File = $dll
    SHA256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
    SignatureStatus = [string]$signature.Status
    EventQueryStatus = $eventStatus
    HistoricalBlocksAtPath = $events
    Note = 'Historical events may refer to an older binary at this path. Signature status and absence of events do not prove the current file is permitted or that VR rendering works.'
} | ConvertTo-Json -Depth 5
