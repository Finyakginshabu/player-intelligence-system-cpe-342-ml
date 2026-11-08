# Downloads and unzips competition data into ./data (requires `kaggle auth login` first).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$comp = 'player-intelligence-system-cpe-342-ml'
New-Item -ItemType Directory -Force "$root\data" | Out-Null
kaggle competitions download -c $comp -p "$root\data"
Expand-Archive -Force "$root\data\$comp.zip" "$root\data"
Get-ChildItem "$root\data" | Select-Object Name, Length
