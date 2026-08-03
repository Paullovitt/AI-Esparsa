"""Identidade e validação da topologia COO dos checkpoints V7.3."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import torch
from torch import nn

from .camada_linear_esparsa import LinearBlocoEsparsa


FORMATO_TOPOLOGIA = "coo-v1"


def metadados_topologia(modelo: nn.Module) -> dict[str, Any]:
    """Calcula uma identidade estável para todas as conexões COO do modelo."""

    resumo: list[dict[str, int | str]] = []
    digest = hashlib.sha256()
    for nome, modulo in modelo.named_modules():
        if not isinstance(modulo, LinearBlocoEsparsa):
            continue
        indices = modulo.indices.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        nome_bytes = nome.encode("utf-8")
        digest.update(struct.pack("<I", len(nome_bytes)))
        digest.update(nome_bytes)
        digest.update(
            struct.pack(
                "<QQQ",
                int(modulo.entradas),
                int(modulo.saidas),
                int(modulo.conexoes),
            )
        )
        digest.update(indices.numpy().tobytes(order="C"))
        resumo.append(
            {
                "nome": nome,
                "entradas": int(modulo.entradas),
                "saidas": int(modulo.saidas),
                "conexoes": int(modulo.conexoes),
            }
        )
    if not resumo:
        raise ValueError("modelo nao possui topologia COO para identificar")
    return {
        "formato": FORMATO_TOPOLOGIA,
        "sha256": digest.hexdigest(),
        "camadas": resumo,
    }


def validar_topologia_checkpoint(
    checkpoint: dict[str, Any],
    modelo: nn.Module,
    *,
    permitir_legado: bool = False,
) -> dict[str, Any]:
    """Rejeita pesos associados a conexões diferentes das usadas ao salvar."""

    registrada = checkpoint.get("topologia")
    if registrada is None:
        if permitir_legado:
            return metadados_topologia(modelo)
        raise ValueError(
            "checkpoint sem identidade de topologia; migre-o ou use "
            "--permitir-checkpoint-legado conscientemente"
        )
    if not isinstance(registrada, dict):
        raise ValueError("metadados de topologia invalidos")
    if registrada.get("formato") != FORMATO_TOPOLOGIA:
        raise ValueError("formato de topologia nao suportado")
    atual = metadados_topologia(modelo)
    if registrada.get("sha256") != atual["sha256"]:
        raise ValueError(
            "topologia COO do checkpoint diverge do codigo/configuracao atual"
        )
    if registrada.get("camadas") != atual["camadas"]:
        raise ValueError("resumo da topologia COO diverge do checkpoint")
    return atual
