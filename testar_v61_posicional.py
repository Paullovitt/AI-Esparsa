"""Treina e audita a V6.1 posicional sem alterar a V6 canonica.

O experimento usa cinco epocas, salva cada epoca e mede qualidade,
geracao livre, pipeline completo e VRAM contra o mesmo checkpoint V6.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from executar_v6 import CHECKPOINT_PADRAO, carregar_v6
from src.atencao_causal_texto import (
    ConfiguracaoTextoCausal,
    DadosTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    selecionar_candidatos_causais,
)
from src.modelo_v61_experimental import (
    DIMENSAO_FEATURES_BASE,
    CodigoTemporalEsparso,
    ModeloV61Posicional,
    TOKENS_POR_FATO,
    TOKENS_POR_PERGUNTA_RESPOSTA,
)
from testar_v6_isolado import (
    avaliar_geracao_livre,
    benchmark_pipeline,
    recall_candidatos,
)
from treinar_v6 import (
    CONFIGURACAO_TEXTO,
    TAMANHO_LOTE,
    avaliar,
    configurar_semente,
    criar_otimizadores,
    forward_indices,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"
SEMENTE = 20260727
CONFIGURACAO_V61 = replace(
    CONFIGURACAO_TEXTO,
    dimensao_features=(
        DIMENSAO_FEATURES_BASE
        + CONFIGURACAO_TEXTO.fatos_por_historia
        + 2
    ),
)


def preparar_dados(
    dados: DadosTextoCausal,
    codigo: CodigoTemporalEsparso,
) -> tuple[DadosTextoCausal, Tensor]:
    descritores = codigo.descritores(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores,
        codigo.configuracao.top_k,
    )
    return dados.com_candidatos(candidatos, validos), descritores


def mascara_conteudo_temporal(
    dados: DadosTextoCausal,
    configuracao: ConfiguracaoTextoCausal,
) -> Tensor:
    """Marca transicoes e conteudo que dependem da ordem dos fatos."""

    comprimento = dados.alvos.shape[1]
    mascara = torch.zeros(
        comprimento,
        dtype=torch.bool,
        device=dados.tokens.device,
    )
    inicio_perguntas = (
        1
        + TOKENS_POR_FATO
        * configuracao.fatos_por_historia
    )
    # Os indices sao posicoes do token-alvo; a perda fica uma posicao antes.
    offsets_alvo = (0, 3, 4, 6, 8, 9, 10, 12, 15)
    for slot in range(configuracao.fatos_por_historia):
        inicio = (
            inicio_perguntas
            + slot * TOKENS_POR_PERGUNTA_RESPOSTA
        )
        for offset in offsets_alvo:
            posicao_perda = inicio + offset - 1
            if 0 <= posicao_perda < comprimento:
                mascara[posicao_perda] = True
    return mascara.view(1, -1)


def treinar_epoca_v61(
    modelo: ModeloV61Posicional,
    dados: DadosTextoCausal,
    descritores: Tensor,
    otimizador: torch.optim.Optimizer,
    otimizador_qk: torch.optim.Optimizer,
) -> float:
    """Treina linguagem e reforca apenas tokens dependentes de ordem."""

    modelo.train()
    mascara_conteudo = mascara_conteudo_temporal(
        dados,
        modelo.configuracao_texto,
    )
    perda_total = 0.0
    peso_total = 0.0
    permutacao = torch.randperm(
        len(dados.tokens),
        device=dados.tokens.device,
    )
    for indices in permutacao.split(TAMANHO_LOTE):
        logits, _ = forward_indices(
            modelo,
            dados,
            descritores,
            indices,
        )
        alvos = dados.alvos[indices]
        perdas = F.cross_entropy(
            logits.flatten(0, 1),
            alvos.flatten(),
            reduction="none",
        ).view_as(alvos)
        pesos = (
            1.0
            + 2.0 * mascara_conteudo.to(perdas.dtype)
            + 5.0
            * dados.mascara_resposta_local[indices].to(perdas.dtype)
        )
        pesos = pesos.expand_as(perdas)
        perda = (perdas * pesos).sum() / pesos.sum()
        otimizador.zero_grad(set_to_none=True)
        otimizador_qk.zero_grad(set_to_none=True)
        perda.backward()
        nn.utils.clip_grad_norm_(
            modelo.parametros_otimizador_denso(),
            max_norm=1.0,
        )
        otimizador.step()
        otimizador_qk.step()
        perda_total += float((perdas * pesos).sum())
        peso_total += float(pesos.sum())
    return perda_total / peso_total


def _preparar_base(
    dados: DadosTextoCausal,
    descritor,
    top_k: int,
) -> tuple[DadosTextoCausal, Tensor]:
    descritores = descritor(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores,
        top_k,
    )
    return dados.com_candidatos(candidatos, validos), descritores


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para o experimento")
    dispositivo = torch.device("cuda")
    configurar_semente(SEMENTE)
    vocabulario = criar_vocabulario()

    treino = gerar_corpus_texto_causal(
        4000,
        101,
        vocabulario,
        CONFIGURACAO_V61,
        usar_combinacoes_ineditas=False,
    ).para(dispositivo)
    validacao = gerar_corpus_texto_causal(
        1000,
        202,
        vocabulario,
        CONFIGURACAO_V61,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    teste = gerar_corpus_texto_causal(
        2000,
        303,
        vocabulario,
        CONFIGURACAO_V61,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)

    codigo = CodigoTemporalEsparso(
        vocabulario,
        CONFIGURACAO_V61,
        semente=SEMENTE,
    ).to(dispositivo)
    treino, descritores_treino = preparar_dados(treino, codigo)
    validacao, descritores_validacao = preparar_dados(
        validacao,
        codigo,
    )
    teste, descritores_teste = preparar_dados(teste, codigo)

    modelo = ModeloV61Posicional(
        vocabulario,
        CONFIGURACAO_V61,
        semente_topologia=SEMENTE,
    ).to(dispositivo)
    otimizador, otimizador_qk = criar_otimizadores(modelo)
    diretorio = RESULTADOS / (
        "teste_v61_posicional_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    diretorio.mkdir(parents=True, exist_ok=False)

    historico = []
    inicio_treino = time.perf_counter()
    for epoca in range(1, 6):
        perda_treino = treinar_epoca_v61(
            modelo,
            treino,
            descritores_treino,
            otimizador,
            otimizador_qk,
        )
        metricas = avaliar(
            modelo,
            validacao,
            descritores_validacao,
        )
        geracao = avaliar_geracao_livre(
            modelo,
            validacao,
            codigo.descritores,
            CONFIGURACAO_V61,
            vocabulario,
            quantidade=24,
        )
        registro = {
            "epoca": epoca,
            "perda_treino_ponderada": perda_treino,
            "validacao": metricas,
            "geracao_livre_24": geracao,
        }
        historico.append(registro)
        torch.save(
            {
                "estado_modelo": modelo.state_dict(),
                "modelo": "V6.1-posicional-experimental",
                "semente": SEMENTE,
                "epoca": epoca,
                "configuracao_texto": asdict(CONFIGURACAO_V61),
                "auditoria_modelo": modelo.auditoria(),
                "auditoria_codigo": codigo.auditoria(),
                "metricas_validacao": metricas,
                "geracao_livre_24": geracao,
            },
            diretorio / f"v61_epoca{epoca}.pt",
        )
        print(
            f"[V6.1] epoca {epoca}/5 "
            f"PPL={metricas['ppl']:.4f} "
            f"local={metricas['acuracia_local_resposta']:.2%} "
            f"livre={geracao['sequencias_exatas']:.2%} "
            f"tokens={geracao['acuracia_tokens_alinhados']:.2%}",
            flush=True,
        )

    qualidade_v61 = avaliar(
        modelo,
        teste,
        descritores_teste,
    )
    geracao_v61 = avaliar_geracao_livre(
        modelo,
        teste,
        codigo.descritores,
        CONFIGURACAO_V61,
        vocabulario,
    )
    recall_v61 = recall_candidatos(teste)

    modelo_base, codigo_base, checkpoint_base = carregar_v6(
        CHECKPOINT_PADRAO,
        dispositivo,
    )
    teste_base, descritores_base = _preparar_base(
        teste,
        codigo_base.descritores,
        modelo_base.configuracao_texto.top_k,
    )
    qualidade_base = avaliar(
        modelo_base,
        teste_base,
        descritores_base,
    )
    geracao_base = avaliar_geracao_livre(
        modelo_base,
        teste_base,
        codigo_base.descritores,
        modelo_base.configuracao_texto,
        vocabulario,
    )

    benchmark = {"v6_base": {}, "v61_posicional": {}}
    for comprimento, lote in ((73, 64), (512, 16)):
        benchmark["v6_base"][str(comprimento)] = benchmark_pipeline(
            modelo_base,
            teste.entradas,
            codigo_base.descritores,
            modelo_base.configuracao_texto,
            comprimento,
            lote,
        )
        benchmark["v61_posicional"][str(comprimento)] = (
            benchmark_pipeline(
                modelo,
                teste.entradas,
                codigo.descritores,
                CONFIGURACAO_V61,
                comprimento,
                lote,
            )
        )

    criterios = {
        "ppl_ate_5_porcento_da_v6": (
            qualidade_v61["ppl"] <= qualidade_base["ppl"] * 1.05
        ),
        "geracao_livre_exata_minimo_50": (
            geracao_v61["sequencias_exatas"] >= 0.50
        ),
        "tokens_livres_20_pontos_acima_da_v6": (
            geracao_v61["acuracia_tokens_alinhados"]
            >= geracao_base["acuracia_tokens_alinhados"] + 0.20
        ),
        "local_minimo_99": (
            qualidade_v61["acuracia_local_resposta"] >= 0.99
        ),
        "pipeline_minimo_75_porcento_da_v6": all(
            benchmark["v61_posicional"][comprimento][
                "pipeline_completo"
            ]["tokens_por_segundo"]
            >= benchmark["v6_base"][comprimento][
                "pipeline_completo"
            ]["tokens_por_segundo"]
            * 0.75
            for comprimento in ("73", "512")
        ),
        "sem_qk_denso_e_ffn_sem_linear": (
            modelo.auditoria()["parametros_qk_densos"] == 0
            and not modelo.auditoria()["ffn_esparsa"][
                "usa_nn_linear"
            ]
        ),
    }
    relatorio = {
        "experimento": "v61_posicional_isolada",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "checkpoint_v6": str(CHECKPOINT_PADRAO),
        "checkpoint_v6_semente": checkpoint_base["semente"],
        "checkpoint_v6_epoca": checkpoint_base["epoca"],
        "v6_modificada": False,
        "semente_v61": SEMENTE,
        "epocas": 5,
        "configuracao_v61": asdict(CONFIGURACAO_V61),
        "auditoria_modelo": modelo.auditoria(),
        "auditoria_codigo": codigo.auditoria(),
        "historico": historico,
        "tempo_treino_segundos": time.perf_counter() - inicio_treino,
        "v6_base": {
            "qualidade": qualidade_base,
            "geracao_livre": geracao_base,
        },
        "v61_posicional": {
            "qualidade": qualidade_v61,
            "geracao_livre": geracao_v61,
            "recall_candidatos": recall_v61,
        },
        "benchmark": benchmark,
        "criterios": criterios,
        "decisao": (
            "candidata_a_promocao"
            if all(criterios.values())
            else "nao_promover_manter_v6_base"
        ),
        "interpretacao": (
            "A posicao senoidal preserva ordem sem pesos densos. O codigo "
            "temporal acrescenta uma feature de slot e uma de papel do "
            "corpus controlado; deve ser revalidado em sequencias variaveis."
        ),
    }
    serializado = json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    (diretorio / "relatorio.json").write_text(
        serializado,
        encoding="utf-8",
    )
    (RESULTADOS / "teste_v61_posicional_ultimo.json").write_text(
        serializado,
        encoding="utf-8",
    )
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    print(f"RELATORIO={diretorio / 'relatorio.json'}")


if __name__ == "__main__":
    main()
