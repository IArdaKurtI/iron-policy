$ErrorActionPreference = "Stop"

$launcherPath = Join-Path $PSScriptRoot "iron polcy v7.vbs"
$iconPath = Join-Path $PSScriptRoot "assets\tank_v7_icon.ico"
$projectShortcutPath = Join-Path $PSScriptRoot "iron polcy v7.lnk"
$desktopFolder = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopFolder "Iron Polcy v7.lnk"

$shell = New-Object -ComObject WScript.Shell

function Test-IsInsideOneDrive([string]$Path) {
    $candidate = [IO.Path]::GetFullPath($Path).TrimEnd([char[]]@('\', '/'))
    $roots = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique
    foreach ($rootValue in $roots) {
        $root = [IO.Path]::GetFullPath($rootValue).TrimEnd([char[]]@('\', '/'))
        if ($candidate.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
            $candidate.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function New-IronPolcyShortcut([string]$ShortcutPath) {
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $launcherPath
    $shortcut.WorkingDirectory = $PSScriptRoot
    $shortcut.IconLocation = "$iconPath,0"
    $shortcut.Description = "Iron Polcy v7 - yerel proje klasorunden baslat"
    $shortcut.Save()
}

New-IronPolcyShortcut $projectShortcutPath

# A backup copy under OneDrive must not replace an existing, valid local shortcut.
$writeDesktopShortcut = $true
if ((Test-IsInsideOneDrive $PSScriptRoot) -and (Test-Path -LiteralPath $desktopShortcutPath)) {
    $existing = $shell.CreateShortcut($desktopShortcutPath)
    if ((Test-Path -LiteralPath $existing.TargetPath) -and
        -not (Test-IsInsideOneDrive $existing.TargetPath)) {
        $writeDesktopShortcut = $false
    }
}
if ($writeDesktopShortcut) {
    New-IronPolcyShortcut $desktopShortcutPath
}
