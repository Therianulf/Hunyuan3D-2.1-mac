# Build the two native extensions the texture pipeline needs on Windows/CUDA.
#
#   hy3dpaint/custom_rasterizer       -> custom_rasterizer_kernel  (CUDA, needs nvcc + MSVC)
#   hy3dpaint/DifferentiableRenderer  -> mesh_inpaint_processor    (C++/pybind11, needs MSVC)
#
# Neither has a Windows build path in the repo: custom_rasterizer ships setup.py
# (Linux-oriented) plus setup_macos.py, and the inpaint processor ships only
# compile_mesh_painter.sh. setup_windows.py covers the second one.
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File build_extensions_windows.ps1

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
Set-Location $repo

$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "venv not found at $py" }

# --- locate the toolchain ---------------------------------------------------
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -products * -latest -property installationPath
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) { throw "vcvarsall.bat not found under $vsPath" }

$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
$cudaHome = (Get-ChildItem $cudaRoot -ErrorAction Stop | Sort-Object Name -Descending | Select-Object -First 1).FullName
Write-Host "VS:   $vsPath"
Write-Host "CUDA: $cudaHome"

# --- import the MSVC environment -------------------------------------------
# Pinned to toolset 14.38. CUDA 12.4's nvcc hard-rejects MSVC 19.40+, and the
# Build Tools default is 14.44 (19.44), so an unpinned vcvarsall fails with
# "unsupported Microsoft Visual Studio version". 14.38 is cl 19.38.
cmd /c "`"$vcvars`" x64 -vcvars_ver=14.38 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
Write-Host ("cl:   " + (Get-Command cl.exe).Source)

$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$env:PATH = "$cudaHome\bin;$env:PATH"
$env:DISTUTILS_USE_SDK = '1'          # required: tells torch's cpp_extension to trust this vcvars env
$env:TORCH_CUDA_ARCH_LIST = '7.5'     # RTX 2070 is Turing; building only sm_75 keeps nvcc fast
$env:PYTHONUTF8 = '1'

# --- 1. custom_rasterizer (CUDA) -------------------------------------------
Write-Host "`n=== building custom_rasterizer ===" -ForegroundColor Cyan
Push-Location (Join-Path $repo 'hy3dpaint\custom_rasterizer')
# --no-build-isolation: setup.py imports torch at build time, so it must run
# against the venv rather than a clean isolated build env.
& $py -m pip install . --no-build-isolation
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "custom_rasterizer build failed" }
Pop-Location

# --- 2. mesh_inpaint_processor (C++) ---------------------------------------
Write-Host "`n=== building mesh_inpaint_processor ===" -ForegroundColor Cyan
Push-Location (Join-Path $repo 'hy3dpaint\DifferentiableRenderer')
& $py setup_windows.py build_ext --inplace
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "mesh_inpaint_processor build failed" }
Pop-Location

# --- verify -----------------------------------------------------------------
Write-Host "`n=== verifying ===" -ForegroundColor Cyan
& $py -c @"
import torch  # MUST come first: Python 3.8+ ignores PATH for extension DLLs, and
              # importing torch registers torch/lib (which ships cudart64_12.dll)
              # as a DLL search directory. Without it the next line raises
              # 'DLL load failed while importing custom_rasterizer_kernel'.
import sys
sys.path.insert(0, 'hy3dpaint')
import custom_rasterizer_kernel
print('custom_rasterizer_kernel: OK')
from DifferentiableRenderer.mesh_inpaint_processor import meshVerticeInpaint
print('mesh_inpaint_processor:   OK')
import custom_rasterizer
print('custom_rasterizer HAS_CUDA_RASTERIZER:', custom_rasterizer.render.HAS_CUDA_RASTERIZER)
"@
