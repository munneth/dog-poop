# PowerShell script to commit files one at a time and sync each commit
# Excludes .mp4 files from commits

Write-Host "🚀 Starting individual file commit and sync process..." -ForegroundColor Green
Write-Host "📋 Excluding .mp4 files from commits" -ForegroundColor Yellow

# Function to run git commands
function Invoke-GitCommand {
    param(
        [string]$Command,
        [bool]$Check = $true
    )
    
    try {
        $result = Invoke-Expression "git $Command" 2>&1
        return $result
    }
    catch {
        Write-Host "Error running: git $Command" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        return $null
    }
}

# Get all modified/untracked files
Write-Host "📁 Scanning for files to commit..." -ForegroundColor Cyan
$gitStatus = git status --porcelain
$filesToCommit = @()

foreach ($line in $gitStatus) {
    if ($line.Trim()) {
        $status = $line.Substring(0, 2)
        $filename = $line.Substring(3)
        
        # Only include files that are modified, added, or untracked
        if ($status -match '^[MA]' -or $status -match '^\?\?') {
            # Exclude .mp4 files
            if (-not $filename.ToLower().EndsWith('.mp4')) {
                $filesToCommit += $filename
            }
        }
    }
}

# Remove duplicates
$filesToCommit = $filesToCommit | Sort-Object -Unique

if ($filesToCommit.Count -eq 0) {
    Write-Host "✅ No files to commit (excluding .mp4 files)" -ForegroundColor Green
    exit 0
}

Write-Host "📁 Found $($filesToCommit.Count) files to commit:" -ForegroundColor Green
for ($i = 0; $i -lt $filesToCommit.Count; $i++) {
    Write-Host "  $($i + 1). $($filesToCommit[$i])" -ForegroundColor White
}

# Reset any staged changes to start fresh
Write-Host "`n🔄 Resetting staged changes to start fresh..." -ForegroundColor Yellow
$resetResult = Invoke-GitCommand "reset HEAD"
if ($resetResult) {
    Write-Host "✅ Reset staged changes" -ForegroundColor Green
}

# Commit each file individually
$successfulCommits = 0
$failedCommits = 0

for ($i = 0; $i -lt $filesToCommit.Count; $i++) {
    $filename = $filesToCommit[$i]
    
    Write-Host "`n" + "="*60 -ForegroundColor Blue
    Write-Host "📦 Processing file $($i + 1)/$($filesToCommit.Count): $filename" -ForegroundColor Cyan
    Write-Host "="*60 -ForegroundColor Blue
    
    # Stage the specific file
    Write-Host "🔄 Staging: $filename" -ForegroundColor Yellow
    $addResult = Invoke-GitCommand "add `"$filename`""
    
    if ($addResult -and $LASTEXITCODE -eq 0) {
        # Commit the file
        $commitMsg = "Add $(Split-Path $filename -Leaf)"
        Write-Host "🔄 Committing: $filename" -ForegroundColor Yellow
        $commitResult = Invoke-GitCommand "commit -m `"$commitMsg`""
        
        if ($commitResult -and $LASTEXITCODE -eq 0) {
            Write-Host "✅ Committed: $filename" -ForegroundColor Green
            
            # Sync to remote
            Write-Host "🔄 Syncing commit to remote..." -ForegroundColor Yellow
            $pushResult = Invoke-GitCommand "push origin main"
            
            if ($pushResult -and $LASTEXITCODE -eq 0) {
                Write-Host "✅ Synced to remote successfully" -ForegroundColor Green
                $successfulCommits++
                Write-Host "✅ Successfully committed and synced: $filename" -ForegroundColor Green
            } else {
                Write-Host "❌ Failed to sync: $filename" -ForegroundColor Red
                $failedCommits++
                # Try alternative push method
                Write-Host "🔄 Trying alternative push method..." -ForegroundColor Yellow
                $altPushResult = Invoke-GitCommand "push --verbose origin main"
                if ($altPushResult -and $LASTEXITCODE -eq 0) {
                    Write-Host "✅ Alternative push successful" -ForegroundColor Green
                    $successfulCommits++
                } else {
                    Write-Host "❌ Alternative push also failed" -ForegroundColor Red
                }
            }
        } else {
            Write-Host "❌ Failed to commit: $filename" -ForegroundColor Red
            $failedCommits++
        }
    } else {
        Write-Host "❌ Failed to stage: $filename" -ForegroundColor Red
        $failedCommits++
    }
    
    # Small delay between commits
    Start-Sleep -Seconds 1
}

# Summary
Write-Host "`n" + "="*60 -ForegroundColor Blue
Write-Host "📊 SUMMARY" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Blue
Write-Host "✅ Successful commits: $successfulCommits" -ForegroundColor Green
Write-Host "❌ Failed commits: $failedCommits" -ForegroundColor Red
Write-Host "📁 Total files processed: $($filesToCommit.Count)" -ForegroundColor White

if ($failedCommits -gt 0) {
    Write-Host "`n⚠️  $failedCommits commits failed. Check the output above for details." -ForegroundColor Yellow
} else {
    Write-Host "`n🎉 All $successfulCommits files committed and synced successfully!" -ForegroundColor Green
} 