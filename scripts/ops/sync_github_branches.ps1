param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $repoRoot
try {
    $codeRef = 'refs/heads/vdt/ops/b200-portable'
    $logRef = 'refs/heads/log'
    $codeCommit = (& git rev-parse --verify $codeRef).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Missing local B200 branch' }
    $logCommit = (& git rev-parse --verify $logRef).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Missing local log branch' }
    $roots = @(& git rev-list --max-parents=0 $logRef)
    if ($LASTEXITCODE -ne 0 -or $roots.Count -ne 1) { throw 'Expected one independent log history root' }
    & git merge-base $codeRef $logRef *> $null
    if ($LASTEXITCODE -eq 0) { throw 'Log branch shares code history; expected orphan history' }
    if ($LASTEXITCODE -ne 1) { throw 'Cannot verify independent histories' }
    # Push immutable snapshots; never force and never change remote/default branch configuration.
    foreach ($url in @('https://github.com/sontungkieu/SimCT.git', 'https://github.com/ihbkaiser/cross-tok-opd.git')) {
        $pushArgs = @('push', '--atomic')
        if ($DryRun) { $pushArgs += '--dry-run' }
        & git @pushArgs $url "${codeCommit}:$codeRef" "${logCommit}:$logRef"
        if ($LASTEXITCODE -ne 0) { throw "Sync failed for $url; earlier repo may already be updated. Re-run after resolving divergence/access." }
        if (-not $DryRun) {
            $remoteRows = @(& git ls-remote $url $codeRef $logRef)
            if ($LASTEXITCODE -ne 0) { throw "Cannot verify $url" }
            $remoteRefs = @{}
            foreach ($row in $remoteRows) { $parts = $row -split '\s+'; $remoteRefs[$parts[1]] = $parts[0] }
            if ($remoteRefs[$codeRef] -ne $codeCommit -or $remoteRefs[$logRef] -ne $logCommit) {
                throw "Remote verification mismatch for $url"
            }
        }
    }
    Write-Output "SYNC_PASS code=$codeCommit log=$logCommit dry_run=$DryRun"
} finally { Pop-Location }
