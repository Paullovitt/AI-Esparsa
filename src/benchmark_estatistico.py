"""Executa benchmarks repetidos e registra dispersão e estado da GPU.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import statistics
import subprocess
from collections.abc import Callable
from typing import Any


def estado_gpu() -> dict[str, str] | None:
    """Consulta clocks, energia e P-state quando o driver oferece os campos."""

    campos = (
        "name,driver_version,pstate,clocks.current.sm,"
        "clocks.current.memory,power.draw,power.limit"
    )
    try:
        processo = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={campos}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    valores = [valor.strip() for valor in processo.stdout.splitlines()[0].split(",")]
    nomes = campos.split(",")
    if len(valores) != len(nomes):
        return None
    return dict(zip(nomes, valores))


def _estatisticas(valores: list[float]) -> dict[str, float]:
    return {
        "media": statistics.fmean(valores),
        "mediana": statistics.median(valores),
        "desvio_padrao": (
            statistics.stdev(valores) if len(valores) > 1 else 0.0
        ),
        "minimo": min(valores),
        "maximo": max(valores),
    }


def executar_benchmark_estatistico(
    medir: Callable[[], dict[str, Any]],
    *,
    repeticoes: int = 5,
    aquecimentos: int = 2,
) -> dict[str, Any]:
    """Usa a mediana como valor representativo e preserva cada repetição."""

    if repeticoes < 2:
        raise ValueError("o benchmark estatístico exige ao menos 2 repetições")
    if aquecimentos < 1:
        raise ValueError("o benchmark exige ao menos 1 aquecimento")
    for _ in range(aquecimentos):
        medir()
    gpu_antes = estado_gpu()
    amostras = [medir() for _ in range(repeticoes)]
    gpu_depois = estado_gpu()
    chaves_numericas = {
        chave
        for amostra in amostras
        for chave, valor in amostra.items()
        if isinstance(valor, (int, float)) and not isinstance(valor, bool)
    }
    estatisticas = {
        chave: _estatisticas(
            [float(amostra[chave]) for amostra in amostras if chave in amostra]
        )
        for chave in sorted(chaves_numericas)
    }
    representativo = dict(amostras[0])
    for chave, resumo in estatisticas.items():
        representativo[chave] = resumo["mediana"]
    representativo["protocolo_estatistico"] = {
        "aquecimentos": aquecimentos,
        "repeticoes": repeticoes,
        "agregacao_principal": "mediana",
        "estatisticas": estatisticas,
        "amostras": amostras,
        "gpu_antes": gpu_antes,
        "gpu_depois": gpu_depois,
    }
    return representativo
