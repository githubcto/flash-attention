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
import math # 追加: 分割計算用
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

BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/{tag_name}/{wheel_name}"
)

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
    if CUDA_HOME is not None:
        return
    warnings.warn(f"{global_option} was requested, but nvcc was not found.")


def check_if_rocm_home_none(global_option: str) -> None:
    if ROCM_HOME is not None:
        return
    warnings.warn(f"{global_option} was requested, but hipcc was not found.")


def detect_hipify_v2():
    try:
        from torch.utils.hipify import __version__
        from packaging.version import Version
        if Version(__version__) >= Version("2.0.0"):
            return True
    except Exception as e:
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
    if "native" in archs and len(archs) > 1:
        raise ValueError("'native' cannot be combined with explicit archs.")


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
    # --- 省略（変更なしの部分なので元のまま使用）---
    # CUDA側のコード（既存のまま）
    pass

elif not SKIP_CUDA_BUILD and IS_ROCM:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    if ROCM_BACKEND == "ck":
        ck_dir = "csrc/composable_kernel"

        if not os.path.exists("./build"):
            os.makedirs("build")

        optdim = os.getenv("OPT_DIM", "32,64,128,256")
        archs = [arch.lower() for arch in os.getenv("GPU_ARCHS", "native").split(";")]
        validate_and_update_archs(archs)

        if archs != ["native"]:
            kernel_targets = archs
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gcn_arch = getattr(props, "gcnArchName", None)
            detected_arch = gcn_arch.split(":")[0]
            kernel_targets = [detected_arch.lower()]
            validate_and_update_archs(kernel_targets)

        targets_arg = ",".join(kernel_targets)
        for direction in ["fwd", "fwd_appendkv", "fwd_splitkv", "bwd"]:
            subprocess.run([sys.executable, f"{ck_dir}/example/ck_tile/01_fmha/generate.py", "-d", direction, "--output_dir", "build", "--receipt", "2", "--optdim", optdim, "--targets", targets_arg], check=True)

        generator_flag = []
        torch_dir = torch.__path__[0]
        if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
            generator_flag = ["-DOLD_GENERATOR_PATH"]

        check_if_rocm_home_none("flash_attn")
        cc_flag = [f"--offload-arch={arch}" for arch in kernel_targets]

        if FORCE_CXX11_ABI:
            torch._C._GLIBCXX_USE_CXX11_ABI = True

        sources = ["csrc/flash_attn_ck/flash_api.cpp",
                "csrc/flash_attn_ck/flash_common.cpp",
                "csrc/flash_attn_ck/mha_bwd.cpp",
                "csrc/flash_attn_ck/mha_fwd_kvcache.cpp",
                "csrc/flash_attn_ck/mha_fwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_bwd.cpp",
                "csrc/flash_attn_ck/mha_varlen_fwd.cpp"] + glob.glob(
            f"build/fmha_*wd*.cpp"
        )

        maybe_hipify_v2_flag = []
        if detect_hipify_v2():
            maybe_hipify_v2_flag = ["-DHIPIFY_V2"]

        rename_cpp_to_cu(sources)

        renamed_sources = ["csrc/flash_attn_ck/flash_api.cu",
                        "csrc/flash_attn_ck/flash_common.cu",
                        "csrc/flash_attn_ck/mha_bwd.cu",
                        "csrc/flash_attn_ck/mha_fwd_kvcache.cu",
                        "csrc/flash_attn_ck/mha_fwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_bwd.cu",
                        "csrc/flash_attn_ck/mha_varlen_fwd.cu"] + glob.glob(f"build/fmha_*wd*.cu")

        # =========================================================================
        # 追加: 分割ビルド用ソース切り出しロジック
        # =========================================================================
        chunk_index = int(os.environ.get("BUILD_CHUNK_INDEX", "-1"))
        num_chunks = int(os.environ.get("BUILD_NUM_CHUNKS", "20"))

        target_sources = renamed_sources
        if chunk_index >= 0 and num_chunks > 0:
            chunk_size = math.ceil(len(renamed_sources) / num_chunks)
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, len(renamed_sources))
            target_sources = renamed_sources[start_idx:end_idx]
            print(f"\n[Parallel Build] Building Chunk {chunk_index + 1}/{num_chunks}")
            print(f"[Parallel Build] Compiling {len(target_sources)} files out of {len(renamed_sources)}\n")
        # =========================================================================

        cc_flag += ["-O3","-std=c++20",
                    "-Wno-unknown-warning-option",
                    "-fbracket-depth=1024",
                    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                    "-fgpu-flush-denormals-to-zero",
                    "-DCK_ENABLE_BF16",
                    "-DCK_ENABLE_BF8",
                    "-DCK_ENABLE_FP16",
                    "-DCK_ENABLE_FP32",
                    "-DCK_ENABLE_FP64",
                    "-DCK_ENABLE_FP8",
                    "-DCK_ENABLE_INT8",
                    "-DCK_USE_XDL",
                    "-DUSE_PROF_API=1",
                    "-D__HIP_PLATFORM_HCC__=1"]

        ck_tile_float_to_bfloat16_default = os.environ.get("CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT")
        if ck_tile_float_to_bfloat16_default is None:
            has_gfx11_target = any(arch.startswith("gfx11") for arch in kernel_targets)
            ck_tile_float_to_bfloat16_default = "0" if has_gfx11_target else "3"
        cc_flag += [f"-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT={ck_tile_float_to_bfloat16_default}"]

        hip_version = get_hip_version()
        if hip_version > Version('5.5.00000'):
            cc_flag += ["-mllvm", "--lsr-drop-solution=1"]
        if hip_version > Version('5.7.23302'):
            cc_flag += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            cc_flag += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            cc_flag += ["-mllvm", "-amdgpu-early-inline-all=true",
                        "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            cc_flag += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

        extra_compile_args = {
            "cxx": ["-O3", "-std=c++20"] + generator_flag + maybe_hipify_v2_flag,
            "nvcc": cc_flag + generator_flag + maybe_hipify_v2_flag,
        }

        include_dirs = [
            Path(this_dir) / "csrc" / "composable_kernel" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "library" / "include",
            Path(this_dir) / "csrc" / "composable_kernel" / "example" / "ck_tile" / "01_fmha",
        ]

        ext_modules.append(
            CUDAExtension(
                name="flash_attn_2_cuda",
                sources=target_sources,  # renamed_sources から target_sources に変更
                extra_compile_args=extra_compile_args,
                include_dirs=include_dirs,
            )
        )


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    # 省略（既存のまま）
    return "", ""

class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if FORCE_BUILD:
            return super().run()
        super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.environ.get("MAX_JOBS"):
            import psutil
            nvcc_threads = max(1, int(NVCC_THREADS))
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            max_num_jobs_memory = max(1, int(free_memory_gb / (5 * nvcc_threads)))
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)
        super().__init__(*args, **kwargs)

    def build_extensions(self) -> None:
        # =========================================================================
        # 追加: コンパイル専用モードの場合、リンクエラーを回避するためにlinkを無効化
        # =========================================================================
        if os.environ.get("BUILD_COMPILE_ONLY", "0") == "1":
            print("[Parallel Build] Skipping link phase because BUILD_COMPILE_ONLY=1")
            self.compiler.link = lambda *args, **kwargs: None

        # =========================================================================
        # 追加: リンク専用モードの場合、Ninja(コンパイル)の実行をモック化して完全にスキップ
        # =========================================================================
        if os.environ.get("BUILD_LINK_ONLY", "0") == "1":
            print("[Parallel Build] Skipping compile phase because BUILD_LINK_ONLY=1")
            import subprocess
            
            # 元の関数をバックアップ（多重パッチ防止）
            if not hasattr(subprocess, "_original_run"):
                subprocess._original_run = subprocess.run
                subprocess._original_check_call = subprocess.check_call
                subprocess._original_check_output = subprocess.check_output

            def is_ninja_cmd(cmd):
                if isinstance(cmd, (list, tuple)) and len(cmd) > 0:
                    return "ninja" in str(cmd[0]).lower()
                elif isinstance(cmd, str):
                    return "ninja" in cmd.lower()
                return False

            def mocked_run(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                if is_ninja_cmd(cmd):
                    print(f"[Parallel Build] Mocking ninja run: {cmd}")
                    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
                return subprocess._original_run(*args, **kwargs)

            def mocked_check_call(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                if is_ninja_cmd(cmd):
                    print(f"[Parallel Build] Mocking ninja check_call: {cmd}")
                    return 0  # 正常終了を返す
                return subprocess._original_check_call(*args, **kwargs)

            def mocked_check_output(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                if is_ninja_cmd(cmd):
                    print(f"[Parallel Build] Mocking ninja check_output: {cmd}")
                    return b""
                return subprocess._original_check_output(*args, **kwargs)

            # subprocessの関数をすり替え
            subprocess.run = mocked_run
            subprocess.check_call = mocked_check_call
            subprocess.check_output = mocked_check_output
        # =========================================================================

        original_spawn = None
        if sys.platform == "win32" and self.compiler.compiler_type == "msvc":
            original_spawn = self.compiler.spawn

            def spawn(cmd):
                if not cmd or Path(str(cmd[0])).name.lower() != "link.exe":
                    return original_spawn(cmd)
                cmd = [str(arg) for arg in cmd]
                if len(subprocess.list2cmdline(cmd)) <= 32767:
                    return original_spawn(cmd)
                with tempfile.TemporaryDirectory() as tmpdir:
                    rsp_path = Path(tmpdir) / "cmdline.txt"
                    rsp_path.write_text(
                        "\n".join(subprocess.list2cmdline([arg]) for arg in cmd[1:]) + "\n",
                        encoding="ascii",
                    )
                    return original_spawn([cmd[0], f"@{rsp_path}"])

            self.compiler.spawn = spawn

        try:
            super().build_extensions()
        finally:
            if original_spawn is not None:
                self.compiler.spawn = original_spawn


if ROCM_BACKEND == "triton":
    install_requires = ["einops", "triton==3.5.1"]
else:
    install_requires = ["torch", "einops"]

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks", "flash_attn.egg-info", "flash_attn.cute", "flash_attn.cute.*")),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: BSD License", "Operating System :: Unix"],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension} if ext_modules else {"bdist_wheel": CachedWheelsCommand},
    python_requires=">=3.9",
    install_requires=install_requires,
    setup_requires=["packaging", "psutil", "ninja"],
)

