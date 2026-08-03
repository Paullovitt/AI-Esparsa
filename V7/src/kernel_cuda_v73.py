"""Carregamento opcional do kernel CUDA fundido da V7.3.

O binário é compilado no cache do PyTorch e nunca entra no checkpoint. A
função mantém o erro de compilação disponível para auditoria e permite que o
runtime use um caminho PyTorch seguro quando o toolchain não estiver presente.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import ModuleType

import torch


_MODULO: ModuleType | None = None
_ERRO_CARREGAMENTO: str | None = None


def _caminho_curto_windows(caminho: Path) -> str:
    """Evita ambiguidades de aspas do ``cmd /c`` ao chamar o vcvars."""

    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    tamanho = ctypes.windll.kernel32.GetShortPathNameW(
        str(caminho),
        buffer,
        len(buffer),
    )
    if tamanho == 0 or tamanho >= len(buffer):
        return str(caminho)
    return buffer.value


def _preparar_msvc() -> None:
    """Importa o ambiente x64 do Build Tools para o processo Python atual."""

    if os.name != "nt" or "VCToolsInstallDir" in os.environ:
        return
    vswhere = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio"
        r"\Installer\vswhere.exe"
    )
    if not vswhere.exists():
        return
    instalacao = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not instalacao:
        return
    vcvars = Path(instalacao) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    comando_vcvars = _caminho_curto_windows(vcvars)
    ambiente = subprocess.check_output(
        ["cmd.exe", "/d", "/c", f"call {comando_vcvars} >nul && set"],
        text=True,
        encoding="mbcs",
    )
    for linha in ambiente.splitlines():
        if "=" in linha:
            chave, valor = linha.split("=", 1)
            os.environ[chave] = valor

    ninja = (
        Path(instalacao)
        / "Common7"
        / "IDE"
        / "CommonExtensions"
        / "Microsoft"
        / "CMake"
        / "Ninja"
    )
    if ninja.exists():
        os.environ["PATH"] = f"{ninja}{os.pathsep}{os.environ['PATH']}"


def carregar_kernel_cuda_v73(*, obrigatorio: bool = False) -> ModuleType | None:
    """Compila uma vez e recarrega o operador fundido do cache local."""

    global _MODULO, _ERRO_CARREGAMENTO
    if _MODULO is not None:
        return _MODULO
    if _ERRO_CARREGAMENTO is not None:
        if obrigatorio:
            raise RuntimeError(_ERRO_CARREGAMENTO)
        return None
    if not torch.cuda.is_available():
        _ERRO_CARREGAMENTO = "CUDA indisponivel para o kernel V7.3"
        if obrigatorio:
            raise RuntimeError(_ERRO_CARREGAMENTO)
        return None

    try:
        _preparar_msvc()
        from torch.utils.cpp_extension import load

        raiz = Path(__file__).resolve().parent / "kernels_v73"
        _MODULO = load(
            name="ai_esparsa_v73_cuda",
            sources=[
                str(raiz / "v73_cuda.cpp"),
                str(raiz / "v73_cuda_kernel.cu"),
            ],
            extra_cflags=["/O2"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-allow-unsupported-compiler",
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                "-lineinfo",
            ],
            verbose=False,
        )
    except Exception as erro:  # pragma: no cover - depende do toolchain local
        _ERRO_CARREGAMENTO = (
            "nao foi possivel compilar/carregar o kernel CUDA V7.3: "
            f"{type(erro).__name__}: {erro}"
        )
        if obrigatorio:
            raise RuntimeError(_ERRO_CARREGAMENTO) from erro
        return None
    return _MODULO


def erro_kernel_cuda_v73() -> str | None:
    """Expõe a causa de fallback sem iniciar uma compilação implicitamente."""

    return _ERRO_CARREGAMENTO
