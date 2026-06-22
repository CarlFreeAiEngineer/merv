param(
    [string]$Tag = $(if ($env:LLAMA_CPP_TAG) { $env:LLAMA_CPP_TAG } else { 'b9761' })
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$dest = Join-Path $root 'bin\llama.cpp'
$server = Join-Path $dest 'llama-server.exe'

if (Test-Path -LiteralPath $server) {
    exit 0
}

$asset = "llama-$Tag-bin-win-cpu-x64.zip"
$url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$asset"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) $asset

Write-Host "[merv] downloading llama.cpp $Tag Windows CPU server ..."
New-Item -ItemType Directory -Path $dest -Force | Out-Null
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
Expand-Archive -LiteralPath $tmp -DestinationPath $dest -Force

if (-not (Test-Path -LiteralPath $server)) {
    throw "Downloaded $asset, but llama-server.exe was not found after extraction."
}

Write-Host "[merv] llama-server ready: $server"
