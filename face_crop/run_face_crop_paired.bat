@echo off
REM One-click paired HQ/LQ face-crop dataset builder for SR (Windows).
REM HQ RAW:  C:\baidunetdiskdownload\PPR0K_all_files_11161_zip\raw_zips\raw
REM LQ tif:  C:\baidunetdiskdownload\PPR0K_all_files_11161_zip\train_val_images_tif_360p\source
REM Output:  C:\code\ppr10k_faces\{hq,lq}\*_faceN.png  (paired)
REM Resumable: re-run skips pairs already in faces_paired_log.csv.
setlocal
cd /d "%~dp0"
python crop_faces_paired.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] crop_faces_paired.py exited with code %errorlevel%
    pause
    exit /b %errorlevel%
)
endlocal
