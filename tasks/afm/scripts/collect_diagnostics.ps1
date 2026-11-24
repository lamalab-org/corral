param(
    [Parameter(Mandatory=$true)][ValidateNotNullOrEmpty()][string]$MachineId,
    [Parameter(Mandatory=$true)][string]$ProcessName,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [ValidateRange(1,168)][int]$Hours = 4
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
[ordered]@{
    machine_id = $MachineId
    hostname = [System.Net.Dns]::GetHostName()
    collected_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    process_name = $ProcessName
    lookback_hours = $Hours
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "machine.json")
# Read-only collection. This does not restart/kill Nanosurf or configure dumps.
$name = [IO.Path]::GetFileNameWithoutExtension($ProcessName)
Get-Process -Name $name -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, StartTime, Path,
        @{Name="FileVersion";Expression={$_.MainModule.FileVersionInfo.FileVersion}} |
    ConvertTo-Json -Depth 4 |
    Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "processes.json")
$filter = @{LogName="Application"; Id=1000,1001; StartTime=(Get-Date).AddHours(-$Hours)}
try {
    # Preserve matching application events as evidence, not proof of a crash.
    $events = @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop |
        Where-Object { $_.Message -match [regex]::Escape($name) })
    $events | Select-Object TimeCreated, Id, ProviderName, Message |
        ConvertTo-Json -Depth 4 |
        Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "application-events.json")
    $events | ForEach-Object { $_.ToXml() } |
        Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "application-events.xml")
} catch {
    $_.ToString() | Set-Content -Encoding UTF8 (Join-Path $OutputDirectory "event-query-error.txt")
}
