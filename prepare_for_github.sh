#!/bin/bash
# Script to prepare repository for GitHub push

echo "=========================================="
echo "Preparing EZKL CNN Benchmark for GitHub"
echo "=========================================="
echo

# Step 1: Clean temporary files
echo "Step 1: Cleaning temporary files..."
rm -rf temp_* temp/
rm -f *.log
rm -f experiment_run_*.log
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name ".DS_Store" -delete
echo "✓ Temporary files cleaned"
echo

# Step 2: Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Step 2: Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "Step 2: Git repository already initialized"
fi
echo

# Step 3: Add all files
echo "Step 3: Staging files..."
git add .
echo "✓ Files staged"
echo

# Step 4: Show status
echo "Step 4: Repository status:"
git status
echo

# Step 5: Check repository size
echo "Step 5: Checking repository size..."
REPO_SIZE=$(du -sh . | cut -f1)
RESULTS_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
echo "  Total repository size: $REPO_SIZE"
echo "  Results directory size: $RESULTS_SIZE"
echo

# Step 6: Provide next steps
echo "=========================================="
echo "Repository prepared!"
echo "=========================================="
echo
echo "Next steps:"
echo "  1. Review staged files: git status"
echo "  2. Commit changes:"
echo "     git commit -m \"Initial commit: EZKL CNN benchmarking framework\""
echo "  3. Create GitHub repository at:"
echo "     https://github.com/new"
echo "  4. Add remote:"
echo "     git remote add origin https://github.com/georgeakor/ezkl-cnn-benchmark.git"
echo "  5. Push to GitHub:"
echo "     git branch -M main"
echo "     git push -u origin main"
echo
echo "Note: If repository size > 100MB, consider using Git LFS for large files"
echo "      GitHub free tier limit is 1GB per repository"
echo
