@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Project virtual environment not found: ..\.venv\Scripts\python.exe
    exit /b 1
)

"%PYTHON%" -m pip install -r need.txt
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install -r build-requirements.txt
if errorlevel 1 exit /b 1

if not exist "models\bearing_condition_v2.npz" (
    echo Trained bearing CNN model not found. Run train_bearing_cnn.py first.
    exit /b 1
)

"%PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed --name "RailwayVehicleBrakeAnalyzerV2" --add-data "..\models\bearing_condition_v2.npz;models" --distpath "dist" --workpath "build\work" --specpath "build" "main.py"
if errorlevel 1 exit /b 1

if not exist "dist\examples" mkdir "dist\examples"
copy /Y "examples\braking_cases.csv" "dist\examples\braking_cases.csv" >nul
if errorlevel 1 exit /b %errorlevel%
copy /Y "examples\车型_*.csv" "dist\examples\" >nul
if errorlevel 1 exit /b %errorlevel%
copy /Y "操作说明书.txt" "dist\操作说明书.txt" >nul
if errorlevel 1 exit /b %errorlevel%
copy /Y "轴承CNN训练报告.txt" "dist\轴承CNN训练报告.txt" >nul
if errorlevel 1 exit /b %errorlevel%
copy /Y "data\mendeley_bearing_v2\ATTRIBUTION.txt" "dist\轴承数据集来源与许可.txt" >nul
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete: dist\RailwayVehicleBrakeAnalyzerV2.exe
echo Sample cases:   dist\examples\braking_cases.csv
echo User manual:    dist\操作说明书.txt
echo Bearing CNN:    embedded in the EXE
