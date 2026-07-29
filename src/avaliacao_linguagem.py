"""Avaliação de perda, perplexidade e acurácia token a token.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .dados_gerador import criar_lotes


SEMENTE = 20260728

@torch.inference_mode()
def avaliar_linguagem(
    modelo: ModeloGeradorEsparso,
    codificados: list[tuple[list[int], list[float]]],
    pad_id: int,
    tamanho_lote: int,
    dispositivo: torch.device,
) -> dict[str, float]:
    """Mede a perda sem os pesos extras usados apenas durante o treino."""

    modelo.eval()
    perda_total = 0.0
    tokens_total = 0
    corretos = 0
    for entradas, alvos, _ in criar_lotes(
        codificados,
        pad_id,
        tamanho_lote,
        dispositivo,
        embaralhar=False,
        semente=SEMENTE,
    ):
        logits, _ = modelo(entradas)
        mascara = alvos.ne(pad_id)
        perda_total += float(
            F.cross_entropy(
                logits[mascara],
                alvos[mascara],
                reduction="sum",
            )
        )
        tokens_total += int(mascara.sum())
        corretos += int(
            logits.argmax(dim=-1)[mascara].eq(alvos[mascara]).sum()
        )
    perda = perda_total / tokens_total
    return {
        "perda": perda,
        "ppl": math.exp(perda),
        "acuracia_token": corretos / tokens_total,
        "tokens": float(tokens_total),
    }
