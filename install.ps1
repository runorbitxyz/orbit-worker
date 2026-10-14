# ORBIT operator installer for Windows: downloads the worker, pairs it with your account in the browser, and
# starts it at every login (a Startup item; no administrator rights needed).
#   irm https://api.runorbit.xyz/install.ps1 | iex
#   $env:ORBIT_PRIVATE=1; irm https://api.runorbit.xyz/install.ps1 | iex      # a rig that serves only you, free
# Remove it later with: py -3 $HOME\.orbit\orbit_worker.py --uninstall-service
$ErrorActionPreference = "Stop"
$OrbitUrl = if ($env:ORBIT_URL) { $env:ORBIT_URL } else { "https://api.runorbit.xyz" }
$Dir = Join-Path $HOME ".orbit"
$Py = $null
foreach ($candidate in @("py", "python", "python3")) {
  $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source -notlike "*WindowsApps*") { $Py = $cmd.Source; break }
}
if (-not $Py) { throw "ORBIT needs Python 3.9+. Install it from https://www.python.org/downloads/ and run this again." }
$PyArgs = @()
if ((Split-Path $Py -Leaf) -eq "py.exe") { $PyArgs = @("-3") }
& $Py @PyArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) { throw "ORBIT needs Python 3.9 or newer." }
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Write-Host "-> Downloading the ORBIT worker to $Dir"
$Worker = Join-Path $Dir "orbit_worker.py"
Invoke-WebRequest -UseBasicParsing -Uri "$OrbitUrl/worker/orbit_worker.py" -OutFile "$Worker.part"
Move-Item -Force "$Worker.part" $Worker
$Extra = @()
if ($env:ORBIT_PRIVATE) { $Extra += "--private" }
& $Py @PyArgs $Worker --orbit-url $OrbitUrl --setup @Extra
