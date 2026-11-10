<#
Build strata-slim for Windows: the engine and the picture reader, a private Python, the server and the prep tools,
in one folder and one zip. The same build as the Dockerfile, for Windows.

    .\scripts\build_windows.ps1                                              # RTX 30 (sm_86), upstream's code
    .\scripts\build_windows.ps1 -CudaArchitectures "86;89" -Patches all -BuildJobs 8

CudaArchitectures: 86 = RTX 30, 89 = RTX 40, 120 = RTX 50 (the last one also gets PTX). Patches: "", "all" or "0001,0002"
(patches/README.md). BuildJobs: each nvcc job can take several GB of RAM.

Needs: Visual Studio 2022 Build Tools (workload "Desktop development with C++"), the CUDA toolkit (its installer sets
CUDA_PATH), Git for Windows (its patch.exe applies patches/), and Python 3 (the py launcher) for CMake and Ninja.

Work files stay in build\windows\ between builds, so a second build compiles only what changed. The result is
build\windows\strata-slim\ and build\windows\strata-slim-<tag>-windows-x64.zip.
#>
param(
    [string]$CudaArchitectures = "86",
    [int]$BuildJobs = 4,
    [string]$Patches = "",
    [string]$PythonVersion = "3.12.10",
    [switch]$NoVision
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"    # Invoke-WebRequest is many times slower while it draws a progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12     # Windows PowerShell 5.1's default is older

$Repo = Split-Path -Parent $PSScriptRoot
$Vendor = Join-Path $Repo "vendor"
$Work = Join-Path $Repo "build\windows"
$Src = Join-Path $Work "src"
$Stage = Join-Path $Work "strata-slim"

function Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Fail([string]$Text) {
    Write-Host "error: $Text" -ForegroundColor Red
    exit 1
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$(Split-Path -Leaf $Program) stopped (exit $LASTEXITCODE)"
    }
}

# robocopy keeps the files' times, so a second build sees only what changed. Exit codes below 8 are success.
function Copy-Mirror([string]$From, [string]$To) {
    robocopy $From $To /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Fail "copying $From failed (robocopy exit $LASTEXITCODE)"
    }
}

function Copy-Files([string]$From, [string]$To, [string[]]$Names) {
    robocopy $From $To @Names /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Fail "copying $($Names -join ', ') from $From failed (robocopy exit $LASTEXITCODE)"
    }
}

# CMake reads backslashes as escapes in some places: give it forward slashes
function CMakePath([string]$Path) {
    return $Path -replace '\\', '/'
}

$upstream = @{}
foreach ($line in Get-Content (Join-Path $Vendor "UPSTREAM")) {
    if ($line -match '^([^=]+)=(.*)$') {
        $upstream[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}
$tag = $upstream["tag"]

# ---- 1. the build tools ------------------------------------------------------------------------------------------
Step "finding the build tools"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Fail "Visual Studio was not found: install Visual Studio 2022 Build Tools with 'Desktop development with C++'"
}
$vs = & $vswhere -latest -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    Fail "Visual Studio 2022 with the C++ tools was not found: install Visual Studio 2022 Build Tools with 'Desktop development with C++'"
}
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
# run vcvars64.bat and take the environment it sets into this PowerShell
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
    }
}
Write-Host "Visual Studio: $vs"

if (-not $env:CUDA_PATH) {
    Fail "CUDA_PATH is not set: install the CUDA toolkit (13.0, as the Docker image uses)"
}
$nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    Fail "no nvcc.exe in $env:CUDA_PATH\bin"
}
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
Write-Host "CUDA: $env:CUDA_PATH"

$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Fail "git was not found: install Git for Windows (its patch.exe applies patches/)"
}
$gitRoot = Split-Path -Parent (Split-Path -Parent $git.Source)       # ...\Git\cmd\git.exe -> ...\Git
$gitTools = Join-Path $gitRoot "usr\bin"
if (-not (Test-Path (Join-Path $gitTools "patch.exe"))) {
    Fail "no patch.exe in $gitTools"
}
# at the end of PATH: Git's usr\bin also has a link.exe, which must not hide MSVC's linker
$env:PATH = "$env:PATH;$gitTools"

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Fail "the Python launcher (py) was not found: install Python 3 from python.org"
}
$toolsEnv = Join-Path $Work "tools"
$toolsPython = Join-Path $toolsEnv "Scripts\python.exe"
if (-not (Test-Path $toolsPython)) {
    Invoke-Checked "py" @("-3", "-m", "venv", $toolsEnv)
}
Invoke-Checked $toolsPython @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "-r", (Join-Path $Vendor "requirements-build.txt"))
$cmake = Join-Path $toolsEnv "Scripts\cmake.exe"
$ninja = Join-Path $toolsEnv "Scripts\ninja.exe"

# ---- 2. the sources, with the patches ----------------------------------------------------------------------------
Step "copying the engine's sources from vendor\ (upstream $tag)"

New-Item -ItemType Directory -Force -Path $Src | Out-Null
Copy-Files $Vendor $Src @("CMakeLists.txt", "UPSTREAM")
foreach ($folder in @("cmake", "include", "src", "third_party", "bench", "tests", "tools\vision", "tools\trellis")) {
    Copy-Mirror (Join-Path $Vendor $folder) (Join-Path $Src $folder)
}
Invoke-Checked $toolsPython @((Join-Path $Repo "scripts\apply_patches.py"), (Join-Path $Repo "patches"), $Src, $Patches)

$llamaCommit = ((Get-Content (Join-Path $Vendor "third_party\ggml\VERSION.txt") -TotalCount 1) -split '\s+')[0]
$llama = Join-Path $Work "llama.cpp-$llamaCommit"
if (-not (Test-Path $llama)) {
    Step "downloading llama.cpp at upstream's pinned commit $llamaCommit"
    $zip = Join-Path $Work "llama.cpp.zip"
    Invoke-WebRequest -UseBasicParsing "https://github.com/ggml-org/llama.cpp/archive/$llamaCommit.zip" -OutFile $zip
    Invoke-Checked "tar" @("-xf", $zip, "-C", $Work)
    Remove-Item $zip
}

# ---- 3. the engine and the picture reader ------------------------------------------------------------------------
# every architecture as real code, plus PTX for the last one; anything before sm_75 needs upstream's SM60 path
$archs = @($CudaArchitectures -split '[;,\s]+' | Where-Object { $_ })
$cudaArchs = (@($archs | ForEach-Object { "$_-real" }) + "$($archs[-1])-virtual") -join ";"
$oldestArch = ($archs | ForEach-Object { [int]$_ } | Measure-Object -Minimum).Minimum
$sm60 = "OFF"
if ($oldestArch -lt 75) {
    $sm60 = "ON"
}

$cudaDefinitions = @(
    "-G", "Ninja",
    "-DCMAKE_MAKE_PROGRAM=$(CMakePath $ninja)",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_ARCHITECTURES=$cudaArchs",
    "-DCMAKE_CUDA_COMPILER=$(CMakePath $nvcc)",
    "-DCUDAToolkit_ROOT=$(CMakePath $env:CUDA_PATH)",
    "-DSTRATA_PORTABLE=ON"
)

# CUDA 13.0's ptxas now and then fails on a PTX file it just wrote; the same build gets past it the second time
function Build-Target([string]$BuildDir, [string[]]$Targets) {
    $arguments = @("--build", $BuildDir, "--target") + $Targets + @("-j", "$BuildJobs")
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "(the build stopped: trying it once more)" -ForegroundColor Yellow
        Invoke-Checked $cmake $arguments
    }
}

Step "building the engine for $CudaArchitectures (the first build takes a long while)"
$engineBuild = Join-Path $Work "build-engine"
Invoke-Checked $cmake (@("-S", $Src, "-B", $engineBuild) + $cudaDefinitions + @(
    "-DSTRATA_ENABLE_CUDA=ON",
    "-DSTRATA_BUILD_TESTS=OFF",
    "-DSTRATA_EXPERIMENTAL_SM60=$sm60",
    "-DSTRATA_NATIVE_EXPERTS=ON",
    "-DSTRATA_Q6K_EXPERTS=ON",
    "-DSTRATA_MMQ_KQUANTS=ON",
    "-DSTRATA_GGML_DIR=$(CMakePath $llama)"
))
Build-Target $engineBuild @("strata", "strata_kq")

$visionBuild = Join-Path $Work "build-vision"
if (-not $NoVision) {
    Step "building the picture reader"
    Invoke-Checked $cmake (@("-S", (Join-Path $Src "tools\vision"), "-B", $visionBuild) + $cudaDefinitions + @(
        "-DLLAMA_DIR=$(CMakePath $llama)",
        "-DSTRATA_VISION_CUDA=ON",
        "-DGGML_CUDA_NCCL=OFF"
    ))
    Build-Target $visionBuild @("strata-vision")
}

# ---- 4. the folder -----------------------------------------------------------------------------------------------
Step "putting together $Stage"

$engine = Join-Path $Stage "engine"
if (Test-Path $engine) {
    Remove-Item -Recurse -Force $engine
}
New-Item -ItemType Directory -Force -Path $engine | Out-Null
Copy-Item (Join-Path $engineBuild "strata.exe") $engine
$kq = Join-Path $engineBuild "strata_kq.dll"
if (Test-Path $kq) {
    Copy-Item $kq $engine
}
$visionExe = Join-Path $visionBuild "bin\strata-vision.exe"
if (-not $NoVision) {
    Copy-Item $visionExe $engine
}
Invoke-Checked $toolsPython @((Join-Path $Repo "scripts\build_info.py"), "cuda", $CudaArchitectures, $Src, $engine)

# the CUDA libraries and the MSVC runtime pieces the programs import (the NVIDIA driver brings nvcuda.dll itself)
$dllSearch = @((Join-Path $env:CUDA_PATH "bin\x64"), (Join-Path $env:CUDA_PATH "bin"))
if ($env:VCToolsRedistDir) {
    $dllSearch += Get-ChildItem -Directory (Join-Path $env:VCToolsRedistDir "x64") | ForEach-Object { $_.FullName }
}
Invoke-Checked $toolsPython (@((Join-Path $Repo "scripts\windows_dlls.py"), $engine) + $dllSearch)

# a private Python: the nuget package is a complete, movable Python (the embeddable zip cannot run the tools as they are)
$python = Join-Path $Stage "python"
$pythonExe = Join-Path $python "python.exe"
$pythonStamp = Join-Path $python ".version"
$pythonCurrent = ""
if (Test-Path $pythonStamp) {
    $pythonCurrent = (Get-Content $pythonStamp -TotalCount 1).Trim()
}
if ($pythonCurrent -ne $PythonVersion) {
    Write-Host "Python $PythonVersion"
    if (Test-Path $python) {
        Remove-Item -Recurse -Force $python
    }
    $nupkg = Join-Path $Work "python-$PythonVersion.zip"
    if (-not (Test-Path $nupkg)) {
        Invoke-WebRequest -UseBasicParsing "https://www.nuget.org/api/v2/package/python/$PythonVersion" -OutFile $nupkg
    }
    $unpacked = Join-Path $Work "python-$PythonVersion"
    if (Test-Path $unpacked) {
        Remove-Item -Recurse -Force $unpacked
    }
    New-Item -ItemType Directory -Force -Path $unpacked | Out-Null
    Invoke-Checked "tar" @("-xf", $nupkg, "-C", $unpacked)
    Move-Item (Join-Path $unpacked "tools") $python
    Remove-Item -Recurse -Force $unpacked
    Set-Content -Path $pythonStamp -Value $PythonVersion
}
Invoke-Checked $pythonExe @("-m", "ensurepip", "--upgrade")
Invoke-Checked $pythonExe @("-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-warn-script-location",
    "-r", (Join-Path $Vendor "requirements.txt"), "-r", (Join-Path $Repo "requirements.txt"))

# the same files the Dockerfile puts in /opt/strata
foreach ($folder in @("serve", "strata_control", "strata_cli", "data")) {
    Copy-Mirror (Join-Path $Vendor $folder) (Join-Path $Stage $folder)
}
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "tools") | Out-Null
Copy-Files (Join-Path $Vendor "tools") (Join-Path $Stage "tools") @("*.py")
Copy-Mirror (Join-Path $llama "gguf-py") (Join-Path $Stage "third_party\llama.cpp\gguf-py")
Copy-Files $Vendor $Stage @("UPSTREAM", ".config-keys", "LICENSE")
Copy-Files $Repo $Stage @("launch.py", "prepare.py", "hf.py", "common.py", "settings.py", "start.bat", "example.env")

# the same check as the image's: the server starts
Step "checking that the server starts"
Push-Location $Stage
try {
    & $pythonExe -m serve.server --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "serve.server --help stopped (exit $LASTEXITCODE)"
    }
}
finally {
    Pop-Location
}

# ---- 5. the zip --------------------------------------------------------------------------------------------------
$zipName = "strata-slim-$tag-windows-x64.zip"
$zipPath = Join-Path $Work $zipName
Step "packing $zipName"
if (Test-Path $zipPath) {
    Remove-Item $zipPath
}
# tar (Windows' own bsdtar) writes a zip without Compress-Archive's 2 GB limit, and much faster. What a test run in the
# folder made stays out: the models, the settings (they may hold HF_TOKEN / API_KEY), the config and the logs.
Invoke-Checked "tar" @("-a", "-cf", $zipPath, "-C", $Work,
    "--exclude", "strata-slim/models", "--exclude", "strata-slim/.env", "--exclude", "strata-slim/run",
    "--exclude", "strata-slim/logs", "--exclude", "__pycache__", "strata-slim")

Write-Host ""
Write-Host "done: $Stage" -ForegroundColor Green
Write-Host "      $zipPath" -ForegroundColor Green
