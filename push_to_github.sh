#!/bin/bash
# Push AgriShield Flask to GitHub

echo "============================================"
echo "  Pushing to GitHub Repository"
echo "============================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed!"
    echo "Please install Git first: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi
echo ""

# Add remote (will update if exists)
echo "Setting up remote repository..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/xCaliyPsO/AgriShield_Flask.git
echo "✅ Remote repository configured"
echo ""

# Add all files
echo "Adding files..."
git add .
echo "✅ Files added"
echo ""

# Commit
echo "Committing changes..."
git commit -m "Initial commit: AgriShield Flask ML API - Pest Detection, Forecasting, and Training Systems"
if [ $? -ne 0 ]; then
    echo "⚠️  Nothing to commit or commit failed"
else
    echo "✅ Changes committed"
fi
echo ""

# Set branch to main
git branch -M main
echo ""

# Push to GitHub
echo "Pushing to GitHub..."
echo ""
echo "⚠️  You may need to authenticate:"
echo "   - Use your GitHub username"
echo "   - Use Personal Access Token as password"
echo "   - Get token from: GitHub Settings > Developer settings > Personal access tokens"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "  ✅ Successfully pushed to GitHub!"
    echo "============================================"
    echo ""
    echo "Repository: https://github.com/xCaliyPsO/AgriShield_Flask"
    echo ""
else
    echo ""
    echo "============================================"
    echo "  ❌ Push failed"
    echo "============================================"
    echo ""
    echo "Possible issues:"
    echo "  1. Authentication required (use Personal Access Token)"
    echo "  2. Repository doesn't exist or no access"
    echo "  3. Network connection issue"
    echo ""
    echo "Try manually:"
    echo "  git push -u origin main"
    echo ""
fi

