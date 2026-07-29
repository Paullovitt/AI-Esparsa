"""Benchmarks básicos de forward e geração autorregressiva.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import time

import torch

from .decodificador_gerador import ConfiguracaoDecodificacao, gerar_controlado
from .tokenizador_palavras import TokenizadorPalavras
from .validacao_gerador import FRASES_GERACAO, extrair_continuacao

@torch.inference_mode()
def benchmark(
    modelo: ModeloGeradorEsparso,
    vocabulario: int,
    dispositivo: torch.device,
) -> dict[str, float | str]:
    """Mede somente o forward paralelo de forma sincronizada na GPU."""

    modelo.eval()
    entradas = torch.randint(
        4,
        vocabulario,
        (16, modelo.configuracao.maximo_contexto),
        device=dispositivo,
    )
    for _ in range(3):
        modelo(entradas)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeticoes = 12
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        modelo(entradas)
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    return {
        "tipo": "forward_paralelo_lote_16_contexto_completo",
        "tokens_por_segundo": entradas.numel() * repeticoes / duracao,
        "latencia_ms": duracao * 1000.0 / repeticoes,
        "vram_pico_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
    }


@torch.inference_mode()
def benchmark_autorregressivo(
    modelo: ModeloGeradorEsparso,
    tokenizador: TokenizadorPalavras,
    prompt: str,
    dispositivo: torch.device,
) -> dict[str, float | str]:
    """Mede uma geração completa, incluindo latência do primeiro token."""

    metricas: dict[str, float] = {}
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dispositivo)
    texto = gerar_controlado(
        modelo,
        tokenizador,
        prompt,
        dispositivo,
        configuracao=ConfiguracaoDecodificacao(
            maximo_tokens=500,
            minimo_tokens_frase=300,
            ngrama_bloqueado=0,
            penalidade_repeticao=0.0,
            janela_penalidade=14,
            temperatura=0.58,
            top_k_amostragem=5,
            parar_apos_frase=True,
            frases_para_encerrar=FRASES_GERACAO,
            bloquear_trigrama_adicional=False,
        ),
        amostrar=False,
        metricas_desempenho=metricas,
    )
    continuacao = extrair_continuacao(texto)
    return {
        "tipo": "geracao_autorregressiva_relato_completo",
        **metricas,
        "vram_pico_mib": (
            torch.cuda.max_memory_allocated(dispositivo) / 1024**2
            if dispositivo.type == "cuda"
            else 0.0
        ),
        "caracteres_gerados": float(len(continuacao)),
        "frases_geradas": float(
            sum(
                token in {".", "!", "?"}
                for token in tokenizador.tokenizar(continuacao)
            )
        ),
    }
