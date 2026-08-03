"""Decodificacao autoregressiva e metricas de repeticao da V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConfiguracaoDecodificacaoV73:
    """Parametros reproduziveis de greedy ou amostragem controlada."""

    modo: str = "amostragem"
    temperatura: float = 0.8
    top_k: int = 40
    top_p: float = 0.90
    penalidade_repeticao: float = 1.15
    ngrama_sem_repeticao: int = 3
    maximo_repeticao_consecutiva: int = 3
    semente: int = 20260802

    def validar(self) -> None:
        if self.modo not in {"greedy", "amostragem"}:
            raise ValueError("modo deve ser greedy ou amostragem")
        if self.temperatura <= 0:
            raise ValueError("temperatura deve ser positiva")
        if self.top_k < 0:
            raise ValueError("top_k nao pode ser negativo")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p deve ficar em (0, 1]")
        if self.penalidade_repeticao < 1:
            raise ValueError("penalidade de repeticao deve ser >= 1")
        if self.ngrama_sem_repeticao < 0:
            raise ValueError("ngrama sem repeticao nao pode ser negativo")
        if self.maximo_repeticao_consecutiva < 0:
            raise ValueError("limite consecutivo nao pode ser negativo")

    @classmethod
    def greedy_puro(cls) -> "ConfiguracaoDecodificacaoV73":
        return cls(
            modo="greedy",
            temperatura=1.0,
            top_k=0,
            top_p=1.0,
            penalidade_repeticao=1.0,
            ngrama_sem_repeticao=0,
            maximo_repeticao_consecutiva=0,
        )


def _aplicar_penalidade_repeticao(
    logits: Tensor,
    historico: list[int],
    penalidade: float,
) -> None:
    if penalidade == 1.0 or not historico:
        return
    ids = torch.tensor(
        sorted(set(historico)),
        dtype=torch.long,
        device=logits.device,
    )
    valores = logits[ids]
    logits[ids] = torch.where(
        valores < 0,
        valores * penalidade,
        valores / penalidade,
    )


def _tokens_bloqueados_por_ngrama(
    historico: list[int],
    tamanho: int,
) -> set[int]:
    if tamanho <= 0 or len(historico) + 1 < tamanho:
        return set()
    prefixo = tuple(historico[-(tamanho - 1):]) if tamanho > 1 else ()
    bloqueados: set[int] = set()
    limite = len(historico) - tamanho + 1
    for inicio in range(max(0, limite + 1)):
        trecho = historico[inicio : inicio + tamanho - 1]
        if tuple(trecho) == prefixo:
            proximo = inicio + tamanho - 1
            if proximo < len(historico):
                bloqueados.add(historico[proximo])
    return bloqueados


def _aplicar_limites_repeticao(
    logits: Tensor,
    historico: list[int],
    configuracao: ConfiguracaoDecodificacaoV73,
) -> None:
    bloqueados = _tokens_bloqueados_por_ngrama(
        historico,
        configuracao.ngrama_sem_repeticao,
    )
    limite = configuracao.maximo_repeticao_consecutiva
    if limite > 0 and len(historico) >= limite:
        ultimo = historico[-1]
        if all(token == ultimo for token in historico[-limite:]):
            bloqueados.add(ultimo)
    if bloqueados:
        ids = torch.tensor(
            sorted(bloqueados),
            dtype=torch.long,
            device=logits.device,
        )
        logits[ids] = -torch.inf


def _filtrar_top_k_top_p(logits: Tensor, top_k: int, top_p: float) -> Tensor:
    filtrados = logits.clone()
    if 0 < top_k < filtrados.numel():
        limiar = torch.topk(filtrados, top_k).values[-1]
        filtrados[filtrados < limiar] = -torch.inf
    if top_p < 1.0:
        ordenados, indices = torch.sort(filtrados, descending=True)
        probabilidades = torch.softmax(ordenados, dim=-1)
        acumuladas = probabilidades.cumsum(dim=-1)
        remover = acumuladas > top_p
        remover[1:] = remover[:-1].clone()
        remover[0] = False
        ordenados[remover] = -torch.inf
        filtrados = torch.full_like(filtrados, -torch.inf)
        filtrados.scatter_(0, indices, ordenados)
    return filtrados


def metricas_diversidade_tokens(
    tokens: list[int],
    eos_id: int,
) -> dict[str, float | bool]:
    """Resume diversidade sem depender da forma textual do tokenizador."""

    uteis = list(tokens)
    eos_emitido = eos_id in uteis
    if eos_emitido:
        uteis = uteis[: uteis.index(eos_id)]
    quantidade = len(uteis)
    resultado: dict[str, float | bool] = {
        "eos_emitido": eos_emitido,
        "tokens_uteis": float(quantidade),
        "fracao_tokens_repetidos": (
            1.0 - len(set(uteis)) / quantidade if quantidade else 0.0
        ),
    }
    for n in (1, 2, 3):
        total = max(quantidade - n + 1, 0)
        ngramas = {
            tuple(uteis[inicio : inicio + n])
            for inicio in range(total)
        }
        resultado[f"distinct_{n}"] = len(ngramas) / total if total else 0.0
    maior = atual = 0
    anterior: int | None = None
    for token in uteis:
        atual = atual + 1 if token == anterior else 1
        maior = max(maior, atual)
        anterior = token
    resultado["maior_repeticao_consecutiva"] = float(maior)
    return resultado


def resumir_metricas_geracao(amostras: list[dict[str, Any]]) -> dict[str, float]:
    """Agrega somente campos numericos comparaveis entre amostras."""

    chaves = (
        "tokens_por_segundo",
        "latencia_primeiro_token_ms",
        "tokens_gerados",
        "entropia_media",
        "margem_top2_media",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "fracao_tokens_repetidos",
        "maior_repeticao_consecutiva",
    )
    resumo = {
        f"{chave}_media": statistics.fmean(
            float(amostra[chave]) for amostra in amostras
        )
        for chave in chaves
        if amostras and all(chave in amostra for amostra in amostras)
    }
    resumo["taxa_eos"] = (
        statistics.fmean(float(bool(item.get("eos_emitido"))) for item in amostras)
        if amostras
        else 0.0
    )
    return resumo


@torch.inference_mode()
def gerar_v73(
    modelo,
    tokens: Tensor,
    maximo_novos_tokens: int,
    eos_id: int,
    configuracao: ConfiguracaoDecodificacaoV73,
) -> tuple[Tensor, dict[str, float | bool]]:
    """Gera um exemplo com cache causal, filtros e auditoria probabilistica."""

    configuracao.validar()
    if maximo_novos_tokens <= 0:
        raise ValueError("maximo_novos_tokens deve ser positivo")
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("geracao V7.3 aceita um exemplo por chamada")

    inicio = time.perf_counter()
    logits, cache = modelo.iniciar_cache_geracao(tokens)
    primeiro = None
    gerados = tokens.clone()
    historico: list[int] = []
    entropias: list[float] = []
    margens: list[float] = []
    gerador = torch.Generator(device=tokens.device)
    gerador.manual_seed(configuracao.semente)

    for _ in range(maximo_novos_tokens):
        brutos = logits[0, -1].float()
        probabilidades_brutas = torch.softmax(brutos, dim=-1)
        entropias.append(
            float(
                -(probabilidades_brutas * probabilidades_brutas.clamp_min(1e-12).log()).sum()
            )
        )
        dois = torch.topk(probabilidades_brutas, min(2, probabilidades_brutas.numel())).values
        margens.append(float(dois[0] - dois[1]) if dois.numel() == 2 else float(dois[0]))

        candidatos = brutos.clone()
        _aplicar_penalidade_repeticao(
            candidatos,
            historico,
            configuracao.penalidade_repeticao,
        )
        _aplicar_limites_repeticao(candidatos, historico, configuracao)
        if not bool(torch.isfinite(candidatos).any()):
            candidatos[eos_id] = 0.0

        if configuracao.modo == "greedy":
            proximo_id = int(candidatos.argmax())
        else:
            candidatos = candidatos / configuracao.temperatura
            candidatos = _filtrar_top_k_top_p(
                candidatos,
                configuracao.top_k,
                configuracao.top_p,
            )
            probabilidades = torch.softmax(candidatos, dim=-1)
            if not bool(torch.isfinite(probabilidades).all()) or float(probabilidades.sum()) <= 0:
                proximo_id = int(brutos.argmax())
            else:
                proximo_id = int(
                    torch.multinomial(
                        probabilidades,
                        1,
                        generator=gerador,
                    )
                )

        proximo = torch.tensor(
            [[proximo_id]],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        gerados = torch.cat((gerados, proximo), dim=1)
        historico.append(proximo_id)
        if primeiro is None:
            primeiro = time.perf_counter()
        if proximo_id == eos_id:
            break
        logits, cache = modelo.avancar_cache_geracao(proximo, cache)

    fim = time.perf_counter()
    duracao = max(fim - inicio, 1e-9)
    diversidade = metricas_diversidade_tokens(historico, eos_id)
    return gerados, {
        "tokens_por_segundo": len(historico) / duracao,
        "latencia_primeiro_token_ms": ((primeiro or fim) - inicio) * 1000.0,
        "tokens_gerados": float(len(historico)),
        "entropia_media": statistics.fmean(entropias) if entropias else 0.0,
        "margem_top2_media": statistics.fmean(margens) if margens else 0.0,
        **diversidade,
    }


def gerar_greedy_v73(
    modelo,
    tokens: Tensor,
    maximo_novos_tokens: int,
    eos_id: int,
) -> tuple[Tensor, dict[str, float | bool]]:
    """Mantem a API greedy legada para comparacoes estritas."""

    return gerar_v73(
        modelo,
        tokens,
        maximo_novos_tokens,
        eos_id,
        ConfiguracaoDecodificacaoV73.greedy_puro(),
    )


def configuracao_serializavel(
    configuracao: ConfiguracaoDecodificacaoV73,
) -> dict[str, Any]:
    return asdict(configuracao)