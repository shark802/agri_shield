@echo off
REM Push AgriShield Flask to GitHub
echo ============================================
echo   Pushing to GitHub Repository
echo ============================================
echo.

cd /d "%~dp0"

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git is not installed!
    echo Please install Git first: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo ✅ Git is installed
echo.

REM Initialize git if needed
if not exist .git (
    echo Initializing git repository...
    git init
    echo ✅ Git repository initialized
) else (
    echo ✅ Git repository already exists
)
echo.

REM Add remote (will update if exists)
echo Setting up remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/xCaliyPsO/AgriShield_Flask.git
echo ✅ Remote repository configured
echo.

REM Add all files
echo Adding files...
git add .
echo ✅ Files added
echo.

REM Commit
echo Committing changes...
git commit -m "Initial commit: AgriShield Flask ML API - Pest Detection, Forecasting, and Training Systems"
if %errorlevel% neq 0 (
    echo ⚠️  Nothing to commit or commit failed
) else (
    echo ✅ Changes committed
)
echo.

REM Set branch to main
git branch -M main
echo.

REM Push to GitHub
echo Pushing to GitHub...
echo.
echo ⚠️  You may need to authenticate:
echo    - Use your GitHub username
echo    - Use Personal Access Token as password
echo    - Get token from: GitHub Settings ^> Developer settings ^> Personal access tokens
echo.
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   ✅ Successfully pushed to GitHub!
    echo ============================================
    echo.
    echo Repository: https://github.com/xCaliyPsO/AgriShield_Flask
    echo.
) else (
    echo.
    echo ============================================
    echo   ❌ Push failed
    echo ============================================
    echo.
    echo Possible issues:
    echo   1. Authentication required (use Personal Access Token)
    echo   2. Repository doesn't exist or no access
    echo   3. Network connection issue
    echo.
    echo Try manually:
    echo   git push -u origin main
    echo.
)

pause

