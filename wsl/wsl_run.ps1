# wsl_run.ps1 - Run a .sh script inside WSL without quoting pain.
#
# Why: invoking `wsl bash -lc "<complex cmd>"` from PowerShell mangles nested
# quotes, $vars and non-ASCII paths (cp936 command-line conversion); wsl.exe
# re-parses its argument list through another shell layer, so inline bash with
# && / {} / nested quotes breaks unpredictably. This wrapper therefore:
#   1. passes ALL values through WSLENV environment variables (UTF-safe), and
#   2. keeps every wsl.exe argument a single-quoted simple word (no nesting),
#   3. delegates real logic to wslrun_launcher.sh (staged to /tmp).
#
# Usage (ExecutionPolicy Bypass needed on this machine):
#   powershell -ExecutionPolicy Bypass -File wsl_run.ps1 .\task.sh                 # foreground
#   powershell -ExecutionPolicy Bypass -File wsl_run.ps1 .\task.sh -Background     # setsid nohup, prints log path
#   powershell -ExecutionPolicy Bypass -File wsl_run.ps1 .\task.sh -Log /tmp/x.log # custom WSL-side log path
#   powershell -ExecutionPolicy Bypass -File wsl_run.ps1 /mnt/c/code/x.sh          # /mnt/x form also accepted
#
# Discipline (media_code): any non-trivial WSL work = write a .sh file, run it
# through this wrapper. Never inline complex quoting into `wsl bash -lc "..."`.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,                       # .sh path (Windows or /mnt/<drive>/ form)
    [string]$Log = "",                     # WSL-side log path (Background mode)
    [string]$Distro = "Ubuntu2404",
    [switch]$Background                    # detach via setsid nohup
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$p) {
    if ($p -match '^/mnt/[a-z]/') { return $p }
    $full = [System.IO.Path]::GetFullPath($p)
    if ($full -match '^([A-Za-z]):[\\/](.*)$') {
        $drive = $Matches[1].ToLower()
        $rest = ($Matches[2] -replace '\\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Cannot convert to WSL path: $p"
}

if (-not (Test-Path -LiteralPath $Script)) {
    throw "Script not found: $Script"
}
if ($Script -match "'") { throw "Script path must not contain single quotes: $Script" }

$launcher = Join-Path $PSScriptRoot "wslrun_launcher.sh"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Launcher not found: $launcher"
}

$wslSrc = Convert-ToWslPath $Script
$wslLauncher = Convert-ToWslPath $launcher
$base = [System.IO.Path]::GetFileNameWithoutExtension($Script)
$wslTmp = "/tmp/wslrun_$base.sh"
$wslLauncherTmp = "/tmp/wslrun_launcher.sh"
if ($Log -eq "") { $Log = "/tmp/wslrun_$base.log" }

# Pass values via WSLENV (UTF-safe, survives non-ASCII paths).
$env:WSLRUN_SRC = $wslSrc
$env:WSLRUN_TMP = $wslTmp
$env:WSLRUN_LOG = $Log
$env:WSLRUN_MODE = $(if ($Background) { "bg" } else { "fg" })
$prevWslenv = $env:WSLENV
$env:WSLENV = "WSLRUN_SRC:WSLRUN_TMP:WSLRUN_LOG:WSLRUN_MODE"

try {
    # Stage 1: copy script + launcher to /tmp (avoids 9p exec quirks), strip
    # CRLF, chmod. Every argument is a single-quoted word -> nothing for
    # wsl.exe to re-split.
    $stage1 = "cp '$wslSrc' '$wslTmp' && cp '$wslLauncher' '$wslLauncherTmp' && sed -i 's/\r$//' '$wslTmp' '$wslLauncherTmp' && chmod +x '$wslTmp' '$wslLauncherTmp'"
    & wsl -d $Distro -- bash -c $stage1
    if ($LASTEXITCODE -ne 0) { throw "wsl stage-1 (copy/sanitize) failed, exit=$LASTEXITCODE" }

    # Stage 2: run the launcher (simple word args only).
    & wsl -d $Distro -- bash "$wslLauncherTmp"
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "wsl run failed, exit=$code" }

    if ($Background) {
        Write-Output "WSL background job started."
        Write-Output "  log : $Log   (tail via: wsl -d $Distro -- tail -f $Log)"
        Write-Output "  stop: wsl -d $Distro -- pkill -f wslrun_$base.sh"
    }
    exit $code
}
finally {
    $env:WSLENV = $prevWslenv
}
