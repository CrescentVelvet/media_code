@echo off
REM One-click batch face detection & cropping (MediaPipe) on Windows.
REM Reads:  C:\code\target_c\*.tif  ->  Writes: C:\code\target_c_faces\*_face1.jpg ...
REM Resumable: re-run skips already-processed files (see faces_log.csv).
setlocal
cd /d "%~dp0"
python crop_faces.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] crop_faces.py exited with code %errorlevel%
    pause
    exit /b %errorlevel%
)
endlocal
