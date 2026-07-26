"""Treina o modelo-base V6 em tres sementes.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.atencao_causal_texto import (
    CodigoEsparsoCausal,
    ConfiguracaoTextoCausal,
    DadosTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    preparar_candidatos,
    selecionar_candidatos_causais,
)
from src.modelo_v6 import (
    CONFIGURACAO_FFN_TEXTO,
    ModeloV6,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"
SEMENTES = (20260726, 20260727, 20260728)
CONFIGURACAO_TEXTO = ConfiguracaoTextoCausal()
TAMANHO_LOTE = 128


def configurar_semente(semente: int) -> None:
    random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def criar_otimizadores(
    modelo: ModeloV6,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    denso = torch.optim.AdamW(
        modelo.parametros_otimizador_denso(),
        lr=0.004,
        weight_decay=1e-4,
    )
    qk = torch.optim.SparseAdam(
        modelo.parametros_qk_esparsos(), lr=0.02
    )
    return denso, qk


def forward_indices(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritores: Tensor,
    indices: Tensor,
    *,
    usar_ffn: bool = True,
) -> tuple[Tensor, Tensor]:
    return modelo(
        dados.entradas[indices],
        descritores[indices],
        dados.candidatos[indices],
        dados.candidatos_validos[indices],
        usar_ffn=usar_ffn,
    )


def treinar_epoca(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritores: Tensor,
    otimizador: torch.optim.Optimizer,
    otimizador_qk: torch.optim.Optimizer,
) -> float:
    modelo.train()
    perda_total = 0.0
    peso_total = 0.0
    permutacao = torch.randperm(
        len(dados.tokens), device=dados.tokens.device
    )
    for indices in permutacao.split(TAMANHO_LOTE):
        logits, _ = forward_indices(
            modelo, dados, descritores, indices
        )
        alvos = dados.alvos[indices]
        perdas = F.cross_entropy(
            logits.flatten(0, 1),
            alvos.flatten(),
            reduction="none",
        ).view_as(alvos)
        pesos = (
            1.0
            + 5.0
            * dados.mascara_resposta_local[indices].to(perdas.dtype)
        )
        perda = (perdas * pesos).sum() / pesos.sum()
        otimizador.zero_grad(set_to_none=True)
        otimizador_qk.zero_grad(set_to_none=True)
        perda.backward()
        nn.utils.clip_grad_norm_(
            modelo.parametros_otimizador_denso(), max_norm=1.0
        )
        otimizador.step()
        otimizador_qk.step()
        perda_total += float((perdas * pesos).sum())
        peso_total += float(pesos.sum())
    return perda_total / peso_total


@torch.inference_mode()
def avaliar(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritores: Tensor,
    *,
    usar_ffn: bool = True,
    tamanho_lote: int = 256,
) -> dict[str, float]:
    modelo.eval()
    perda_total = 0.0
    tokens_total = 0
    tokens_corretos = 0
    respostas = 0
    respostas_corretas = 0
    recuperacoes = 0
    for inicio in range(0, len(dados.tokens), tamanho_lote):
        indices = torch.arange(
            inicio,
            min(inicio + tamanho_lote, len(dados.tokens)),
            device=dados.tokens.device,
        )
        logits, posicoes = forward_indices(
            modelo,
            dados,
            descritores,
            indices,
            usar_ffn=usar_ffn,
        )
        alvos = dados.alvos[indices]
        perda_total += float(
            F.cross_entropy(
                logits.flatten(0, 1),
                alvos.flatten(),
                reduction="sum",
            )
        )
        previsoes = logits.argmax(dim=-1)
        tokens_total += alvos.numel()
        tokens_corretos += int(previsoes.eq(alvos).sum())
        mascara = dados.mascara_resposta_local[indices]
        respostas += int(mascara.sum())
        respostas_corretas += int(
            (previsoes.eq(alvos) & mascara).sum()
        )
        origens = dados.posicao_fato_origem[indices]
        recuperacoes += int((posicoes.eq(origens) & mascara).sum())
    perda = perda_total / tokens_total
    return {
        "perda": perda,
        "ppl": math.exp(perda),
        "acuracia_token": tokens_corretos / tokens_total,
        "acuracia_local_resposta": respostas_corretas / respostas,
        "recuperacao_fato": recuperacoes / respostas,
    }


@torch.inference_mode()
def gerar_primeiro_local(
    modelo: ModeloV6,
    codigo: CodigoEsparsoCausal,
    prefixo: Tensor,
) -> int:
    descritores = codigo.descritores(prefixo)
    candidatos, validos = selecionar_candidatos_causais(
        descritores, CONFIGURACAO_TEXTO.top_k
    )
    logits, _ = modelo(
        prefixo, descritores, candidatos, validos
    )
    return int(logits[:, -1].argmax(dim=-1))


@torch.inference_mode()
def avaliar_geracao(
    modelo: ModeloV6,
    codigo: CodigoEsparsoCausal,
    dados: DadosTextoCausal,
    quantidade: int = 72,
) -> dict[str, float | int]:
    modelo.eval()
    corretos = 0
    for linha in range(quantidade):
        posicao = int(
            dados.mascara_resposta_local[linha].nonzero().flatten()[-1]
        )
        esperado = int(dados.alvos[linha, posicao])
        prefixo = dados.tokens[
            linha : linha + 1, : posicao + 1
        ].contiguous()
        previsto = gerar_primeiro_local(modelo, codigo, prefixo)
        corretos += int(previsto == esperado)
    return {
        "prompts": quantidade,
        "acuracia_primeiro_local": corretos / quantidade,
        "erros": quantidade - corretos,
    }


def medir(
    funcao: Callable[[], Any],
    tokens_lote: int,
    repeticoes: int,
) -> dict[str, float]:
    for _ in range(20):
        funcao()
    torch.cuda.synchronize()
    memoria_base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        funcao()
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    return {
        "latencia_ms": duracao * 1_000 / repeticoes,
        "tokens_por_segundo": tokens_lote * repeticoes / duracao,
        "vram_temporaria_mib": max(
            0, torch.cuda.max_memory_allocated() - memoria_base
        )
        / (1024**2),
    }


def expandir_tokens(
    base: Tensor,
    comprimento: int,
    lote: int,
) -> Tensor:
    tokens = base
    if len(tokens) < lote:
        tokens = tokens.repeat(math.ceil(lote / len(tokens)), 1)
    tokens = tokens[:lote]
    return tokens.repeat(
        1, math.ceil(comprimento / tokens.shape[1])
    )[:, :comprimento].contiguous()


@torch.inference_mode()
def benchmark_modelo(
    modelo: ModeloV6,
    codigo: CodigoEsparsoCausal,
    tokens_base: Tensor,
    comprimento: int,
    lote: int,
) -> dict[str, float]:
    tokens = expandir_tokens(tokens_base, comprimento, lote)
    descritores = codigo.descritores(tokens)
    candidatos, validos = selecionar_candidatos_causais(
        descritores, CONFIGURACAO_TEXTO.top_k
    )
    repeticoes = 100 if comprimento == 73 else 50
    return medir(
        lambda: modelo(
            tokens, descritores, candidatos, validos
        ),
        tokens.numel(),
        repeticoes,
    )


def agregar(
    execucoes: list[dict[str, Any]],
) -> dict[str, float]:
    itens = execucoes
    caminhos = {
        "ppl": [item["teste"]["ppl"] for item in itens],
        "acuracia_token": [
            item["teste"]["acuracia_token"] for item in itens
        ],
        "acuracia_local": [
            item["teste"]["acuracia_local_resposta"] for item in itens
        ],
        "recuperacao": [
            item["teste"]["recuperacao_fato"] for item in itens
        ],
        "geracao": [
            item["geracao"]["acuracia_primeiro_local"] for item in itens
        ],
        "ppl_sem_ffn": [
            item["ablacao_sem_ffn"]["ppl"] for item in itens
        ],
        "tempo_treino_s": [
            item["tempo_treino_segundos"] for item in itens
        ],
    }
    saida = {}
    for nome, valores in caminhos.items():
        saida[f"{nome}_media"] = statistics.fmean(valores)
        saida[f"{nome}_desvio"] = statistics.pstdev(valores)
    return saida


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria")
    dispositivo = torch.device("cuda")
    vocabulario = criar_vocabulario()
    codigo = CodigoEsparsoCausal(
        vocabulario, CONFIGURACAO_TEXTO
    ).to(dispositivo)
    treino = gerar_corpus_texto_causal(
        4000,
        101,
        vocabulario,
        CONFIGURACAO_TEXTO,
        usar_combinacoes_ineditas=False,
    ).para(dispositivo)
    validacao = gerar_corpus_texto_causal(
        1000,
        202,
        vocabulario,
        CONFIGURACAO_TEXTO,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    teste = gerar_corpus_texto_causal(
        2000,
        303,
        vocabulario,
        CONFIGURACAO_TEXTO,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    treino, descritores_treino = preparar_candidatos(treino, codigo)
    validacao, descritores_validacao = preparar_candidatos(
        validacao, codigo
    )
    teste, descritores_teste = preparar_candidatos(teste, codigo)

    diretorio = (
        RESULTADOS
        / f"v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    diretorio.mkdir(parents=True, exist_ok=False)
    execucoes = []
    benchmarks = []

    for semente in SEMENTES:
        configurar_semente(semente)
        modelo = ModeloV6(
            vocabulario,
            CONFIGURACAO_TEXTO,
            semente_topologia=semente,
        ).to(dispositivo)
        otimizador, otimizador_qk = criar_otimizadores(modelo)
        historico = []
        inicio_treino = time.perf_counter()
        for epoca in range(1, 6):
            perda_treino = treinar_epoca(
                modelo,
                treino,
                descritores_treino,
                otimizador,
                otimizador_qk,
            )
            metricas = avaliar(
                modelo, validacao, descritores_validacao
            )
            historico.append(
                {
                    "epoca": epoca,
                    "perda_treino_ponderada": perda_treino,
                    **metricas,
                }
            )
            torch.save(
                {
                    "estado_modelo": modelo.state_dict(),
                    "modelo": "V6",
                    "semente": semente,
                    "epoca": epoca,
                    "configuracao_texto": asdict(
                        CONFIGURACAO_TEXTO
                    ),
                    "configuracao_ffn": asdict(
                        CONFIGURACAO_FFN_TEXTO
                    ),
                    "metricas_validacao": metricas,
                },
                diretorio
                / f"v6_semente{semente}_epoca{epoca}.pt",
            )
            print(
                f"[V6 s={semente}] epoca {epoca}/5 "
                f"PPL={metricas['ppl']:.4f} "
                f"local={metricas['acuracia_local_resposta']:.2%} "
                f"recall={metricas['recuperacao_fato']:.2%}",
                flush=True,
            )

        teste_final = avaliar(
            modelo, teste, descritores_teste
        )
        ablacao = avaliar(
            modelo,
            teste,
            descritores_teste,
            usar_ffn=False,
        )
        geracao = avaliar_geracao(
            modelo, codigo, teste
        )
        execucoes.append(
            {
                "modelo": "V6",
                "semente": semente,
                "arquitetura": modelo.auditoria(),
                "historico": historico,
                "teste": teste_final,
                "ablacao_sem_ffn": ablacao,
                "geracao": geracao,
                "tempo_treino_segundos": (
                    time.perf_counter() - inicio_treino
                ),
            }
        )
        for comprimento, lote in ((73, 64), (512, 16)):
            benchmarks.append(
                {
                    "modelo": "V6",
                    "semente": semente,
                    "comprimento": comprimento,
                    "lote": lote,
                    "metricas": benchmark_modelo(
                        modelo,
                        codigo,
                        teste.entradas,
                        comprimento,
                        lote,
                    ),
                }
            )
        del modelo
        torch.cuda.empty_cache()

    relatorio = {
        "experimento": "modelo_base_v6",
        "nome": "V6 Base",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "versao_modelo": "V6",
        "checkpoint_rollback": "modelos/v6_rollback.pt",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "sementes": list(SEMENTES),
        "epocas": 5,
        "checkpoints": 15,
        "configuracao_texto": asdict(CONFIGURACAO_TEXTO),
        "configuracao_ffn": asdict(CONFIGURACAO_FFN_TEXTO),
        "execucoes": execucoes,
        "agregado": agregar(execucoes),
        "benchmark": benchmarks,
        "bytes_checkpoints": sum(
            caminho.stat().st_size
            for caminho in diretorio.glob("v6_*.pt")
        ),
        "observacao": (
            "A FFN textual usa topologia aleatoria balanceada sobre as 48 "
            "dimensoes; nao recebe campos, regras ou semantica do dataset."
        ),
    }
    serializado = json.dumps(
        relatorio, ensure_ascii=False, indent=2
    ) + "\n"
    caminho = diretorio / "relatorio.json"
    caminho.write_text(serializado, encoding="utf-8")
    (RESULTADOS / "v6_ultimo.json").write_text(
        serializado, encoding="utf-8"
    )
    print(json.dumps(relatorio["agregado"], ensure_ascii=False, indent=2))
    print(f"RELATORIO={caminho}")


if __name__ == "__main__":
    main()
