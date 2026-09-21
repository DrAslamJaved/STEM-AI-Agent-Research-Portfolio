[CmdletBinding()]
param(
    [string]$Repo = 'G:\Research\STEM\Micro1_STEM_Agent_Portfolio',
    [string]$ProjectName = 'project_07_formal_proof_agent'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PinnedCommit = 'f0dcc8b59e630fba00ba9569ca6714700e0a8801'
$LeanToolchain = 'leanprover-community/lean:3.42.1'
$Repo = [System.IO.Path]::GetFullPath($Repo)
$Project = Join-Path $Repo $ProjectName
$CanonicalUpstream = Join-Path $Project 'benchmarks\minif2f\upstream'
$Elan = Join-Path $env:USERPROFILE '.elan\bin\elan.exe'

if (-not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) {
    throw "Canonical Git repository is missing: $Repo"
}
if (-not (Test-Path -LiteralPath $Project)) {
    throw "Canonical Project 07 directory is missing: $Project"
}
if (-not (Test-Path -LiteralPath $Elan)) {
    throw "elan.exe is missing: $Elan"
}

$CurrentBranch = (git -C $Repo branch --show-current).Trim()
if ($CurrentBranch -eq '') {
    throw 'Refusing detached HEAD for canonical runtime provisioning.'
}

if (Test-Path -LiteralPath $CanonicalUpstream) {
    if (-not (Test-Path -LiteralPath (Join-Path $CanonicalUpstream 'leanpkg.toml'))) {
        throw "Canonical miniF2F directory is not a valid checkout: $CanonicalUpstream"
    }
    $Observed = ((git -C $CanonicalUpstream rev-parse HEAD) -join '').Trim()
    if ($Observed -ne $PinnedCommit) {
        throw "Canonical miniF2F commit mismatch. Expected $PinnedCommit; found $Observed"
    }
    Write-Host "Using existing canonical miniF2F checkout: $CanonicalUpstream"
}
else {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CanonicalUpstream) | Out-Null
    git clone https://github.com/openai/miniF2F.git $CanonicalUpstream
    if ($LASTEXITCODE -ne 0) { throw 'miniF2F clone failed.' }
    git -C $CanonicalUpstream checkout --detach $PinnedCommit
    if ($LASTEXITCODE -ne 0) { throw "Pinned miniF2F checkout failed: $PinnedCommit" }
    $Observed = ((git -C $CanonicalUpstream rev-parse HEAD) -join '').Trim()
    if ($Observed -ne $PinnedCommit) {
        throw "Canonical miniF2F commit mismatch after clone. Found $Observed"
    }
}

$LeanVersion = ((& $Elan run $LeanToolchain lean --version) -join "`n").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Pinned Lean toolchain verification failed.' }

Push-Location -LiteralPath $CanonicalUpstream
try {
    & $Elan run $LeanToolchain leanpkg configure
    if ($LASTEXITCODE -ne 0) { throw 'miniF2F leanpkg configure failed.' }
    & $Elan run $LeanToolchain leanpkg build
    if ($LASTEXITCODE -ne 0) { throw 'miniF2F leanpkg build failed.' }
}
finally {
    Pop-Location
}

$RuntimeRoot = Join-Path $Project 'runs\phase_09\canonical_runtime'
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$Evidence = [ordered]@{
    schema_version = 1
    phase = '09'
    canonical_project_root = $Project
    canonical_minif2f_root = $CanonicalUpstream
    benchmark_commit = $PinnedCommit
    lean_toolchain = $LeanToolchain
    lean_version = $LeanVersion
    configure_status = 'passed'
    build_status = 'passed'
    legacy_worktree_modified = $false
    validated_at_utc = [DateTime]::UtcNow.ToString('o')
}
$Evidence | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RuntimeRoot 'canonical_runtime_validation.json') -Encoding utf8

Write-Host ''
Write-Host 'PASS: Project 07 miniF2F/Lean runtime is provisioned in the canonical project directory.'
Write-Host "Canonical upstream: $CanonicalUpstream"
Write-Host 'The legacy Phase 02 worktree has not been changed or removed.'
