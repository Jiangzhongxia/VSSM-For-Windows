# Mamba-SSM & Causal-Conv1d Windows 编译与部署指南

本指南详细记录了在 Windows (x64) 环境下，手动编译、打包并安装 `mamba_ssm` 和 `causal-conv1d` 的完整流程。

**主要解决的问题：**
* ❌ **C1060**: 编译器堆空间不足 (内存溢出)。
* ❌ **C1083**: 无法打开包括文件 `stddef.h` (VS 环境未加载)。
* ❌ **Sm_90 Error**: 默认编译 H100 架构导致内存耗尽。
* ❌ **BackendUnavailable**: Pip 版本过低导致构建失败。

---

## 📋 1. 环境前置要求 (Prerequisites)

在开始之前，必须确保系统满足以下条件：

* **操作系统**: Windows 10/11 x64
* **编译器**: **Visual Studio 2022** (安装时需勾选 "使用 C++ 的桌面开发")
* **CUDA**: CUDA Toolkit 12.x (推荐 12.1 或 12.4)
* **Python 环境**: Anaconda 或 Miniconda (Python 3.10)
* **虚拟内存 (关键)**: 
    * 由于编译极为消耗内存，建议物理内存 32GB+。
    * 如果内存不足，**务必**手动设置 Windows 虚拟内存：初始大小 **32GB (32000MB)**，最大大小 **64GB**。

---

## ⚙️ 2. 基础依赖安装

打开普通终端 (CMD/PowerShell)，激活 Conda 环境并安装基础库：

```bash
# 1. 激活环境
conda activate mamba

# 2. 安装 PyTorch (根据你的 CUDA 版本，这里以 12.4 为例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# 3. 核心：升级构建工具 (解决 BackendUnavailable / TomlError)
python -m pip install --upgrade pip setuptools wheel packaging

# 4. 安装运行时依赖
pip install transformers einops pytest

# 5.DOS
REM === 1. 准备工作 ===
REM 激活 Conda 环境 (修改为你的实际路径)
call E:\MiniConda\Scripts\activate.bat mamba
REM 进入源码目录
cd /d D:\Segmenatation\mamba-for-windows-main\causal-conv1d-1.4.0

REM === 2. 彻底清理缓存 (防止 Sm_90 错误) ===
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q causal_conv1d.egg-info
rmdir /s /q build\temp.win-amd64-3.10

REM === 3. 设置环境变量 ===
set MAX_JOBS=1
set TORCH_CUDA_ARCH_LIST=8.6;8.9
set DISTUTILS_USE_SDK=1
set CAUSAL_CONV1D_FORCE_BUILD=TRUE

REM === 4. 打包成 Wheel 文件 ===
python setup.py bdist_wheel

REM === 5. 本机安装 (可选) ===
REM 如果你想直接装在当前机器，可以使用生成的 whl
pip install dist\causal_conv1d-1.4.0-cp310-cp310-win_amd64.whl

REM === 1. 进入源码目录 ===
cd /d D:\Segmenatation\mamba-for-windows-main\mamba-2.2.2

REM === 2. 清理缓存 ===
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q mamba_ssm.egg-info

REM === 3. 环境变量 (沿用之前的，但为保险可再设一次) ===
set MAX_JOBS=1
set TORCH_CUDA_ARCH_LIST=8.6;8.9
set DISTUTILS_USE_SDK=1

REM === 4. 打包 ===
python setup.py bdist_wheel

REM === 5. 本机安装 (可选) ===
pip install dist\mamba_ssm-2.2.2-cp310-cp310-win_amd64.whl
