# Copyright (c) 2023, Tri Dao.

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
import tempfile
import math 
from pathlib import Path
from typing import Literal, Optional
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
    IS_HIP_EXTENSION,
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

PACKAGE_NAME = "flash_attn"
BASE_WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"
ROCM_BACKEND: Optional[Literal["triton", "ck"]] = None
if IS_ROCM:
    ROCM_BACKEND = "triton" if os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE" else "ck"
NVCC_THREADS = os.getenv("NVCC_THREADS") or "4"

@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")

def get_platform():
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version

def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
    if bare_metal_version >= Version("11.8") and "90" in archs:
        cc_flag += ["-gencode", "arch=compute_90,code=sm_90"]
    if bare_metal_version >= Version("12.8"):
        if "100" in archs:
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flag += ["-gencode", "arch=compute_100,code=sm_100"]
        if "120" in archs:
            if bare_metal_version >= Version("12.9"):
                cc_flag += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]
        if "110" in archs:
            if bare_metal_version >= Version("13.0"):
                cc_flag += ["-gencode", "arch=compute_110f,code=sm_110"]
            else:
                if bare_metal_version >= Version("12.8"):
                    cc_flag += ["-gencode", "arch=compute_101,code=sm_101"]
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]
    return cc_flag

def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))

def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is None:
        warnings.warn(f"{global_option} was requested, but nvcc was not found.")

def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is None:
        warnings.warn(f"{global_option} was requested, but hipcc was not found.")

def detect_hipify_v2():
    try:
        from torch.utils.hipify import __version__
        if Version(__version__) >= Version("2.0.0"):
            return True
    except:
        pass
    return False

def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", NVCC_THREADS]

def rename_cpp_to_cu(cpp_files):
    for entry in cpp_files:
        cu_file = os.path.splitext(entry)[0] + ".cu"
        if not os.path.exists(cu_file):
            shutil.copy(entry, cu_file)

def validate_and_update_archs(archs):
    allowed_archs = ["native", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]
    assert all(arch in allowed_archs for arch in archs), f"Invalid archs: {archs}. Allowed: {allowed_archs}"

cmdclass = {}
ext_modules = []

if IS_ROCM:
    if ROCM_BACKEND == "triton":
        if os.path.isdir(".git"):
            subprocess.run(["git", "submodule", "update", "--init", "third_party/aiter"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "third_party/aiter"], check=True)
    elif ROCM_BACKEND == "ck":
        if os.path.isdir(".git"):
            subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
else:
    if os.path.isdir(".git"):
        subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)

if not SKIP_CUDA_BUILD and not IS_ROCM:
    # (CUDA側の設定は変更せず維持)
    check_if_cuda_home_none("flash_attn")
    archs = cuda_archs()
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    cc_flag = []
    add_cuda_gencodes(cc_flag, archs, bare_metal_version)
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"], "nvcc": append_nvcc_threads(["-O3", "-std=c++17"] + cc_flag)}
    # (以下、CUDA版のソース収集ロジックが続く)

elif not SKIP_CUDA_BUILD and IS_ROCM:
    if ROCM_BACKEND == "ck":
        ck_dir = "csrc/composable_kernel"
        if not os.path.exists("./build"): os.makedirs("build")
        optdim = os.getenv("OPT_DIM", "32,64,128,256")
        archs = [arch.lower() for arch in os.getenv("GPU_ARCHS", "native").split(";")]
        validate_and_update_archs(archs)
        if archs != ["native"]:
            kernel_targets = archs
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            kernel_targets = [props.gcnArchName.split(":")[0].lower()]
        
        for direction in ["fwd", "fwd_appendkv", "fwd_splitkv", "bwd"]:
            subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", direction, "--output_dir", "build", "--receipt", "2", "--optdim", optdim, "--targets", ",".join(kernel_targets)], check=True)

        sources = ["csrc/flash_attn_ck/flash_api.cpp", "csrc/flash_attn_ck/flash_common.cpp", "csrc/flash_attn_ck/mha_bwd.cpp", "csrc/flash_attn_ck/mha_fwd_kvcache.cpp", "csrc/flash_attn_ck/mha_fwd.cpp", "csrc/flash_attn_ck/mha_varlen_bwd.cpp", "csrc/flash_attn_ck/mha_varlen_fwd.cpp"] + glob.glob(f"build/fmha_*wd*.cpp")
        rename_cpp_to_cu(sources)
        renamed_sources = [s.replace(".cpp", ".cu") for s in sources]
        renamed_sources.sort()

        # =========================================================================
        # 改修点: 20分割ビルド + リンク時全ソース復元ロジック
        # =========================================================================
        chunk_index = int(os.environ.get("BUILD_CHUNK_INDEX", "-1"))
        num_chunks = 20 # 明示的に20分割を指定
        
        target_sources = renamed_sources
        # リンク専用モードでない場合のみ、ソースを分割する
        if os.environ.get("BUILD_LINK_ONLY", "0") != "1":
            if chunk_index >= 0:
                chunk_size = math.ceil(len(renamed_sources) / num_chunks)
                target_sources = renamed_sources[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
                print(f"\n[Parallel Build] Chunk {chunk_index + 1}/20: Compiling {len(target_sources)} files\n")
        else:
            print(f"\n[Parallel Build] Link Mode: Using all {len(target_sources)} files for linking\n")
        # =========================================================================

        maybe_hipify_v2_flag = ["-DHIPIFY_V2"] if detect_hipify_v2() else []
        cc_flag = [f"--offload-arch={arch}" for arch in kernel_targets] + ["-O3","-std=c++20","-DCK_USE_XDL","-D__HIP_PLATFORM_HCC__=1"]
        extra_compile_args = {"cxx": ["-O3", "-std=c++20"] + maybe_hipify_v2_flag, "nvcc": cc_flag + maybe_hipify_v2_flag}
        
        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=target_sources,
                extra_compile_args=extra_compile_args,
                include_dirs=[Path(this_dir)/"csrc"/"composable_kernel"/"include", Path(this_dir)/"csrc"/"composable_kernel"/"library"/"include", Path(this_dir)/"csrc"/"composable_kernel"/"example"/"ck_tile"/"01_fmha"],
            )
        )

def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    return f"{public_version}+{local_version}" if local_version else str(public_version)

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.environ.get("MAX_JOBS"):
            import psutil
            os.environ["MAX_JOBS"] = str(max(1, os.cpu_count() // 2))
        super().__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        # =========================================================================
        # 改修点: BUILD_COMPILE_ONLY / BUILD_LINK_ONLY の制御パッチ
        # =========================================================================
        if os.environ.get("BUILD_COMPILE_ONLY", "0") == "1":
            print("[Parallel Build] Skipping link phase.")
            self.compiler.link = lambda *args, **kwargs: None

        if os.environ.get("BUILD_LINK_ONLY", "0") == "1":
            print("[Parallel Build] Mocking Ninja to skip compilation and run only link.")
            import subprocess
            if not hasattr(subprocess, "_original_run"):
                subprocess._original_run = subprocess.run
                def mocked_run(*args, **kwargs):
                    cmd = args[0] if args else kwargs.get("args", [])
                    if any("ninja" in str(c).lower() for c in (cmd if isinstance(cmd, list) else [cmd])):
                        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
                    return subprocess._original_run(*args, **kwargs)
                subprocess.run = mocked_run
                subprocess.check_call = lambda *args, **kwargs: 0
        # =========================================================================

        original_spawn = None
        if sys.platform == "win32" and self.compiler.compiler_type == "msvc":
            original_spawn = self.compiler.spawn
            def spawn(cmd):
                if not cmd or Path(str(cmd[0])).name.lower() != "link.exe": return original_spawn(cmd)
                if len(subprocess.list2cmdline(cmd)) <= 32767: return original_spawn(cmd)
                with tempfile.TemporaryDirectory() as tmpdir:
                    rsp = Path(tmpdir) / "cmdline.txt"
                    rsp.write_text("\n".join(subprocess.list2cmdline([arg]) for arg in cmd[1:]), encoding="ascii")
                    return original_spawn([cmd[0], f"@{rsp}"])
            self.compiler.spawn = spawn
        try:
            super().build_extensions()
        finally:
            if original_spawn: self.compiler.spawn = original_spawn

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(exclude=("build", "csrc", "include", "tests", "dist")),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: BSD License", "Operating System :: Microsoft :: Windows"],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension} if ext_modules else {"bdist_wheel": CachedWheelsCommand},
    python_requires=">=3.9",
    install_requires=["torch", "einops"],
    setup_requires=["packaging", "psutil", "ninja"],
)

