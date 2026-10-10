param(
    [string]$CacheRoot = "$env:USERPROFILE\hf_qwen3_scale_cache",
    [string]$ArchivePath = "$env:USERPROFILE\qwen3_scale_models.tar"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheScript = Join-Path $ScriptRoot "cache_qwen3_assets.py"
$Expected = @(
    "hub\models--Qwen--Qwen3-0.6B-Base",
    "hub\models--Qwen--Qwen3-8B-Base"
)

if (-not (Test-Path -LiteralPath $CacheScript -PathType Leaf)) {
    throw "Missing cache script: $CacheScript"
}

$env:HF_HOME = $CacheRoot
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

& py -3.12 -u $CacheScript 0p6b 8b --models-only
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed with exit code $LASTEXITCODE"
}

foreach ($RelativePath in $Expected) {
    $FullPath = Join-Path $CacheRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Container)) {
        throw "Downloaded model cache is missing: $FullPath"
    }
}

if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

& tar -cf $ArchivePath -C $CacheRoot `
    "hub/models--Qwen--Qwen3-0.6B-Base" `
    "hub/models--Qwen--Qwen3-8B-Base"
if ($LASTEXITCODE -ne 0) {
    throw "Archive creation failed with exit code $LASTEXITCODE"
}

& tar -tf $ArchivePath | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Archive verification failed with exit code $LASTEXITCODE"
}

$Hash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLower()
$HashPath = "$ArchivePath.sha256"
"$Hash  $(Split-Path -Leaf $ArchivePath)" |
    Set-Content -LiteralPath $HashPath -Encoding ascii

Write-Host ""
Write-Host "Both Base models were downloaded and the archive passed tar verification."
Write-Host "Archive: $ArchivePath"
Write-Host "SHA256:  $Hash"
Write-Host "Hash file: $HashPath"
