"""Valida a candidata V6.1 em varias amostras e medicoes.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from executar_v6 import CHECKPOINT_PADRAO, carregar_v6
from executar_v61 import CHECKPOINT_V61, carregar_v61
from src.atencao_causal_texto import (
    DadosTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    selecionar_candidatos_causais,
)
from testar_v6_isolado import (
    avaliar_geracao_livre,
    benchmark_pipeline,
)
from treinar_v6 import avaliar


RAIZ = Path(__file__).resolve().parent
RESULTADO = RAIZ / "resultados" / "v61_candidata_validacao.json"
SEMENTES_DADOS = (303, 404, 505, 606)


def preparar(
    dados: DadosTextoCausal,
    descritor,
    top_k: int,
) -> tuple[DadosTextoCausal, torch.Tensor]:
    descritores = descritor(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores,
        top_k,
    )
    return dados.com_candidatos(candidatos, validos), descritores


def agregar(
    resultados: list[dict[str, object]],
    grupo: str,
    metrica: str,
) -> dict[str, float]:
    valores = [
        float(resultado[grupo][metrica])
        for resultado in resultados
    ]
    return {
        "media": statistics.fmean(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "desvio": statistics.pstdev(valores),
    }


def agregar_benchmark(
    rodadas: list[dict[str, object]],
    modelo: str,
    comprimento: str,
    metrica: str,
) -> dict[str, float]:
    valores = [
        float(
            rodada[modelo][comprimento]["pipeline_completo"][
                metrica
            ]
        )
        for rodada in rodadas
    ]
    return {
        "media": statistics.fmean(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "desvio": statistics.pstdev(valores),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para validar desempenho")
    dispositivo = torch.device("cuda")
    vocabulario = criar_vocabulario()
    modelo, codigo, checkpoint = carregar_v61(
        CHECKPOINT_V61,
        dispositivo,
    )
    modelo_base, codigo_base, checkpoint_base = carregar_v6(
        CHECKPOINT_PADRAO,
        dispositivo,
    )

    resultados = []
    tokens_benchmark = None
    dados_base_303 = None
    descritores_base_303 = None
    for semente in SEMENTES_DADOS:
        dados = gerar_corpus_texto_causal(
            2000,
            semente,
            vocabulario,
            modelo.configuracao_texto,
            usar_combinacoes_ineditas=True,
        ).para(dispositivo)
        dados, descritores = preparar(
            dados,
            codigo.descritores,
            modelo.configuracao_texto.top_k,
        )
        qualidade = avaliar(modelo, dados, descritores)
        geracao = avaliar_geracao_livre(
            modelo,
            dados,
            codigo.descritores,
            modelo.configuracao_texto,
            vocabulario,
        )
        resultados.append(
            {
                "semente_dados": semente,
                "qualidade": qualidade,
                "geracao_livre": geracao,
            }
        )
        print(
            f"[robustez {semente}] "
            f"PPL={qualidade['ppl']:.4f} "
            f"livre={geracao['sequencias_exatas']:.2%} "
            f"tokens={geracao['acuracia_tokens_alinhados']:.2%}",
            flush=True,
        )
        if semente == 303:
            tokens_benchmark = dados.entradas
            dados_base_303, descritores_base_303 = preparar(
                dados,
                codigo_base.descritores,
                modelo_base.configuracao_texto.top_k,
            )

    if (
        tokens_benchmark is None
        or dados_base_303 is None
        or descritores_base_303 is None
    ):
        raise RuntimeError("a semente de controle 303 nao foi avaliada")

    controle_base = {
        "qualidade": avaliar(
            modelo_base,
            dados_base_303,
            descritores_base_303,
        ),
        "geracao_livre": avaliar_geracao_livre(
            modelo_base,
            dados_base_303,
            codigo_base.descritores,
            modelo_base.configuracao_texto,
            vocabulario,
        ),
    }
    rodadas_benchmark = []
    for rodada in range(1, 4):
        registro = {"rodada": rodada, "v6_base": {}, "v61": {}}
        for comprimento, lote in ((73, 64), (512, 16)):
            registro["v6_base"][str(comprimento)] = (
                benchmark_pipeline(
                    modelo_base,
                    tokens_benchmark,
                    codigo_base.descritores,
                    modelo_base.configuracao_texto,
                    comprimento,
                    lote,
                )
            )
            registro["v61"][str(comprimento)] = benchmark_pipeline(
                modelo,
                tokens_benchmark,
                codigo.descritores,
                modelo.configuracao_texto,
                comprimento,
                lote,
            )
        rodadas_benchmark.append(registro)
        print(f"[benchmark] rodada {rodada}/3", flush=True)

    agregado = {
        "ppl": agregar(resultados, "qualidade", "ppl"),
        "acuracia_token": agregar(
            resultados,
            "qualidade",
            "acuracia_token",
        ),
        "local": agregar(
            resultados,
            "qualidade",
            "acuracia_local_resposta",
        ),
        "recuperacao": agregar(
            resultados,
            "qualidade",
            "recuperacao_fato",
        ),
        "geracao_exata": agregar(
            resultados,
            "geracao_livre",
            "sequencias_exatas",
        ),
        "eos": agregar(
            resultados,
            "geracao_livre",
            "taxa_eos",
        ),
        "tokens_livres": agregar(
            resultados,
            "geracao_livre",
            "acuracia_tokens_alinhados",
        ),
        "locais_livres": agregar(
            resultados,
            "geracao_livre",
            "acuracia_locais_alinhados",
        ),
    }
    benchmark_agregado = {}
    for comprimento in ("73", "512"):
        base_tps = agregar_benchmark(
            rodadas_benchmark,
            "v6_base",
            comprimento,
            "tokens_por_segundo",
        )
        v61_tps = agregar_benchmark(
            rodadas_benchmark,
            "v61",
            comprimento,
            "tokens_por_segundo",
        )
        base_vram = agregar_benchmark(
            rodadas_benchmark,
            "v6_base",
            comprimento,
            "vram_temporaria_mib",
        )
        v61_vram = agregar_benchmark(
            rodadas_benchmark,
            "v61",
            comprimento,
            "vram_temporaria_mib",
        )
        benchmark_agregado[comprimento] = {
            "v6_base_tokens_por_segundo": base_tps,
            "v61_tokens_por_segundo": v61_tps,
            "razao_velocidade_media": (
                v61_tps["media"] / base_tps["media"]
            ),
            "v6_base_vram_mib": base_vram,
            "v61_vram_mib": v61_vram,
        }

    criterios = {
        "ppl_maxima_1_60": agregado["ppl"]["maximo"] <= 1.60,
        "geracao_exata_minima_80": (
            agregado["geracao_exata"]["minimo"] >= 0.80
        ),
        "tokens_livres_media_98": (
            agregado["tokens_livres"]["media"] >= 0.98
        ),
        "locais_livres_minimo_99": (
            agregado["locais_livres"]["minimo"] >= 0.99
        ),
        "pipeline_medio_minimo_75_porcento": all(
            benchmark_agregado[comprimento][
                "razao_velocidade_media"
            ]
            >= 0.75
            for comprimento in ("73", "512")
        ),
        "vram_media_ate_5_porcento": all(
            benchmark_agregado[comprimento][
                "v61_vram_mib"
            ]["media"]
            <= benchmark_agregado[comprimento][
                "v6_base_vram_mib"
            ]["media"]
            * 1.05
            for comprimento in ("73", "512")
        ),
    }
    relatorio = {
        "experimento": "validacao_robusta_v61_candidata",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "checkpoint_v61": str(CHECKPOINT_V61),
        "checkpoint_v61_semente": checkpoint["semente"],
        "checkpoint_v61_epoca": checkpoint["epoca"],
        "checkpoint_v6": str(CHECKPOINT_PADRAO),
        "checkpoint_v6_semente": checkpoint_base["semente"],
        "checkpoint_v6_epoca": checkpoint_base["epoca"],
        "v6_modificada": False,
        "sementes_dados": list(SEMENTES_DADOS),
        "resultados": resultados,
        "agregado": agregado,
        "controle_v6_semente_dados_303": controle_base,
        "rodadas_benchmark": rodadas_benchmark,
        "benchmark_agregado": benchmark_agregado,
        "criterios": criterios,
        "decisao": (
            "aprovada_como_candidata_v61"
            if all(criterios.values())
            else "reprovada_manter_v6"
        ),
        "limite": (
            "O slot e os papeis temporais usam a estrutura fixa do corpus. "
            "A validacao nao comprova textos de tamanho ou formato livre."
        ),
    }
    RESULTADO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "agregado": agregado,
            "benchmark_agregado": benchmark_agregado,
            "criterios": criterios,
            "decisao": relatorio["decisao"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"RELATORIO={RESULTADO}")


if __name__ == "__main__":
    main()
