#!/usr/bin/env python3
import subprocess
import os
import sys
import time

def run_command(command, check=True):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode

def get_staged_files():
    """Get list of staged files"""
    stdout, stderr, code = run_command("git diff --cached --name-only")
    if code != 0:
        print(f"Error getting staged files: {stderr}")
        return []
    return [f.strip() for f in stdout.split('\n') if f.strip()]

def get_unstaged_files():
    """Get list of unstaged files"""
    stdout, stderr, code = run_command("git status --porcelain")
    if code != 0:
        print(f"Error getting unstaged files: {stderr}")
        return []
    
    files = []
    for line in stdout.split('\n'):
        if line.strip():
            status = line[:2]
            filename = line[3:]
            if status.startswith('M') or status.startswith('A') or status.startswith('??'):
                files.append(filename)
    return files

def commit_file(filename):
    """Commit a single file"""
    print(f"\n🔄 Committing: {filename}")
    
    # Stage the specific file
    stdout, stderr, code = run_command(f'git add "{filename}"')
    if code != 0:
        print(f"❌ Error staging {filename}: {stderr}")
        return False
    
    # Commit the file
    commit_msg = f"Add {os.path.basename(filename)}"
    stdout, stderr, code = run_command(f'git commit -m "{commit_msg}"')
    if code != 0:
        print(f"❌ Error committing {filename}: {stderr}")
        return False
    
    print(f"✅ Committed: {filename}")
    return True

def sync_commit():
    """Push the commit to remote"""
    print("🔄 Syncing commit to remote...")
    
    # Try push with increased timeout
    stdout, stderr, code = run_command("git push origin main")
    if code != 0:
        print(f"❌ Push failed: {stderr}")
        # Try alternative push method
        stdout, stderr, code = run_command("git push --verbose origin main")
        if code != 0:
            print(f"❌ Alternative push also failed: {stderr}")
            return False
    
    print("✅ Synced to remote successfully")
    return True

def main():
    print("🚀 Starting individual file commit and sync process...")
    print("📋 Excluding .mp4 files from commits")
    
    # Get all files that need to be committed
    staged_files = get_staged_files()
    unstaged_files = get_unstaged_files()
    
    # Filter out .mp4 files
    all_files = []
    for file in staged_files + unstaged_files:
        if not file.lower().endswith('.mp4'):
            all_files.append(file)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in all_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    
    if not unique_files:
        print("✅ No files to commit (excluding .mp4 files)")
        return
    
    print(f"📁 Found {len(unique_files)} files to commit:")
    for i, file in enumerate(unique_files, 1):
        print(f"  {i}. {file}")
    
    # Reset any staged changes to start fresh
    print("\n🔄 Resetting staged changes to start fresh...")
    stdout, stderr, code = run_command("git reset HEAD")
    if code != 0:
        print(f"Warning: Could not reset staged changes: {stderr}")
    
    # Commit each file individually
    successful_commits = 0
    failed_commits = 0
    
    for i, filename in enumerate(unique_files, 1):
        print(f"\n{'='*60}")
        print(f"📦 Processing file {i}/{len(unique_files)}: {filename}")
        print(f"{'='*60}")
        
        if commit_file(filename):
            if sync_commit():
                successful_commits += 1
                print(f"✅ Successfully committed and synced: {filename}")
            else:
                failed_commits += 1
                print(f"❌ Failed to sync: {filename}")
                # Continue with next file even if sync fails
        else:
            failed_commits += 1
            print(f"❌ Failed to commit: {filename}")
        
        # Small delay between commits
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful commits: {successful_commits}")
    print(f"❌ Failed commits: {failed_commits}")
    print(f"📁 Total files processed: {len(unique_files)}")
    
    if failed_commits > 0:
        print(f"\n⚠️  {failed_commits} commits failed. Check the output above for details.")
    else:
        print(f"\n🎉 All {successful_commits} files committed and synced successfully!")

if __name__ == "__main__":
    main() 