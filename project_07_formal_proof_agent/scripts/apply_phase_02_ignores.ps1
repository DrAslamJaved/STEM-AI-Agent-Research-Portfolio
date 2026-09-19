$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$GitIgnore = Join-Path $ProjectRoot '.gitignore'

if (-not (Test-Path -LiteralPath $GitIgnore)) {
    throw "Missing .gitignore: $GitIgnore"
}

$RequiredLines = @(
    'benchmarks/minif2f/upstream/',
    'benchmarks/minif2f/upstream/_target/',
    'reports/phase_02/*.log'
)

$Current = Get-Content -LiteralPath $GitIgnore -Raw
$Missing = @($RequiredLines | Where-Object { $Current -notmatch [regex]::Escape($_) })

if ($Missing.Count -gt 0) {
    Add-Content -LiteralPath $GitIgnore -Value ("`n# Phase 02 local benchmark checkout and build logs`n" + ($Missing -join "`n") + "`n")
    Write-Host "Added Phase 02 ignore rules to $GitIgnore"
} else {
    Write-Host 'Phase 02 ignore rules already present.'
}
