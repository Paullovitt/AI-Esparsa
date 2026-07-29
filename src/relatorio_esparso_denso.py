"""Contrato e relatório da comparação esparsa contra densa.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math

from .versao import VERSAO_PROJETO


SEMENTE = 20260728
QUANTIDADES_PADRAO = (50_000, 1_000, 1_000)
EPOCAS = 5
TAMANHO_LOTE = 100
PASSOS_POR_EPOCA = 500
TAXA_APRENDIZADO = 0.0018
FRASES_GERACAO = 24
MINIMO_CARACTERES_GERACAO = 2_000

def _protocolo() -> dict[str, object]:
    return {
        "semente": SEMENTE,
        "dados": {
            "treino": QUANTIDADES_PADRAO[0],
            "validacao": QUANTIDADES_PADRAO[1],
            "teste": QUANTIDADES_PADRAO[2],
        },
        "epocas": EPOCAS,
        "lote": TAMANHO_LOTE,
        "passos_por_epoca": PASSOS_POR_EPOCA,
        "passos_totais": EPOCAS * PASSOS_POR_EPOCA,
        "otimizador": "AdamW",
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "betas": [0.9, 0.95],
        "weight_decay": 1e-4,
        "clip_grad_norm": 1.0,
        "agenda": "aquecimento_mais_cosseno",
        "frases_geracao": FRASES_GERACAO,
        "minimo_caracteres": MINIMO_CARACTERES_GERACAO,
    }


def _comparar_metrica(
    esparso: float,
    denso: float,
    *,
    maior_melhor: bool,
) -> dict[str, float | str]:
    if math.isclose(esparso, denso, rel_tol=1e-9, abs_tol=1e-12):
        vencedor = "empate"
    elif (esparso > denso) == maior_melhor:
        vencedor = "esparso"
    else:
        vencedor = "denso"
    if denso == 0.0:
        vantagem = 0.0
    elif maior_melhor:
        vantagem = (esparso / denso - 1.0) * 100.0
    else:
        vantagem = (1.0 - esparso / denso) * 100.0
    return {
        "esparso": esparso,
        "denso": denso,
        "vencedor": vencedor,
        "vantagem_percentual_esparso": vantagem,
    }


def construir_comparacao(
    esparso: dict[str, object],
    denso: dict[str, object],
) -> dict[str, object]:
    """Consolida métricas sem transformar um resultado misto em marketing."""

    arquitetura_e = esparso["arquitetura"]
    arquitetura_d = denso["arquitetura"]
    avaliacao_e = esparso["avaliacao_final"]
    avaliacao_d = denso["avaliacao_final"]
    geracao_e = esparso["geracao_livre"]
    geracao_d = denso["geracao_livre"]
    forward_e = esparso["desempenho_forward"]
    forward_d = denso["desempenho_forward"]
    auto_e = esparso["desempenho_autorregressivo"]
    auto_d = denso["desempenho_autorregressivo"]

    metricas = {
        "parametros": _comparar_metrica(
            float(arquitetura_e["parametros"]),
            float(arquitetura_d["parametros"]),
            maior_melhor=False,
        ),
        "ppl": _comparar_metrica(
            float(avaliacao_e["ppl"]),
            float(avaliacao_d["ppl"]),
            maior_melhor=False,
        ),
        "acuracia_token": _comparar_metrica(
            float(avaliacao_e["acuracia_token"]),
            float(avaliacao_d["acuracia_token"]),
            maior_melhor=True,
        ),
        "aprovacao_geracao": _comparar_metrica(
            float(geracao_e["taxa_aprovacao"]),
            float(geracao_d["taxa_aprovacao"]),
            maior_melhor=True,
        ),
        "recuperacao_campos": _comparar_metrica(
            float(geracao_e["recuperacao_campos_pedido"]),
            float(geracao_d["recuperacao_campos_pedido"]),
            maior_melhor=True,
        ),
        "repeticao_trigramas": _comparar_metrica(
            float(geracao_e["repeticao_trigramas_media"]),
            float(geracao_d["repeticao_trigramas_media"]),
            maior_melhor=False,
        ),
        "taxa_retentativa": _comparar_metrica(
            float(geracao_e["taxa_uso_retentativa"]),
            float(geracao_d["taxa_uso_retentativa"]),
            maior_melhor=False,
        ),
        "tempo_treino_segundos": _comparar_metrica(
            float(esparso["tempo_treino_total_segundos"]),
            float(denso["tempo_treino_total_segundos"]),
            maior_melhor=False,
        ),
        "vram_treino_mib": _comparar_metrica(
            float(esparso["vram_treino_pico_mib"]),
            float(denso["vram_treino_pico_mib"]),
            maior_melhor=False,
        ),
        "forward_tokens_s": _comparar_metrica(
            float(forward_e["tokens_por_segundo"]),
            float(forward_d["tokens_por_segundo"]),
            maior_melhor=True,
        ),
        "vram_forward_mib": _comparar_metrica(
            float(forward_e["vram_pico_mib"]),
            float(forward_d["vram_pico_mib"]),
            maior_melhor=False,
        ),
        "geracao_tokens_s": _comparar_metrica(
            float(auto_e["tokens_por_segundo"]),
            float(auto_d["tokens_por_segundo"]),
            maior_melhor=True,
        ),
        "primeiro_token_ms": _comparar_metrica(
            float(auto_e["latencia_primeiro_token_ms"]),
            float(auto_d["latencia_primeiro_token_ms"]),
            maior_melhor=False,
        ),
        "checkpoint_bytes": _comparar_metrica(
            float(esparso["checkpoint_final_bytes"]),
            float(denso["checkpoint_final_bytes"]),
            maior_melhor=False,
        ),
    }
    eficiencia = [
        "tempo_treino_segundos",
        "vram_treino_mib",
        "forward_tokens_s",
        "vram_forward_mib",
        "geracao_tokens_s",
        "primeiro_token_ms",
        "checkpoint_bytes",
    ]
    qualidade_preservada = (
        float(avaliacao_e["ppl"]) <= float(avaliacao_d["ppl"]) * 1.01
        and float(geracao_e["taxa_aprovacao"])
        >= float(geracao_d["taxa_aprovacao"])
    )
    vantagens_eficiencia = sum(
        metricas[nome]["vencedor"] == "esparso"
        for nome in eficiencia
    )
    resumo_numerico = {
        "aceleracao_treino_denso": (
            float(esparso["tempo_treino_total_segundos"])
            / float(denso["tempo_treino_total_segundos"])
        ),
        "reducao_vram_treino_denso_percentual": (
            1.0
            - float(denso["vram_treino_pico_mib"])
            / float(esparso["vram_treino_pico_mib"])
        )
        * 100.0,
        "aceleracao_forward_denso": (
            float(forward_d["tokens_por_segundo"])
            / float(forward_e["tokens_por_segundo"])
        ),
        "aceleracao_geracao_densa": (
            float(auto_d["tokens_por_segundo"])
            / float(auto_e["tokens_por_segundo"])
        ),
        "vencedor_vram_forward": (
            "esparso"
            if float(forward_e["vram_pico_mib"])
            <= float(forward_d["vram_pico_mib"])
            else "denso"
        ),
        "reducao_vram_forward_vencedor_percentual": (
            (
                1.0
                - min(
                    float(forward_e["vram_pico_mib"]),
                    float(forward_d["vram_pico_mib"]),
                )
                / max(
                    float(forward_e["vram_pico_mib"]),
                    float(forward_d["vram_pico_mib"]),
                )
            )
            * 100.0
        ),
        "diferenca_ppl_denso_menos_esparso": (
            float(avaliacao_d["ppl"]) - float(avaliacao_e["ppl"])
        ),
    }
    vantagem_pratica = (
        qualidade_preservada
        and vantagens_eficiencia > len(eficiencia) / 2
    )
    return {
        "experimento": "comparacao_esparso_denso_parametros_equivalentes",
        "versao": VERSAO_PROJETO,
        "autor": "Paulo Augusto",
        "ano": 2026,
        "protocolo": _protocolo(),
        "ambiente": esparso["ambiente"],
        "justica": {
            "mesmos_dados": True,
            "mesma_ordem_de_lotes": True,
            "mesmos_passos": True,
            "mesmo_otimizador_e_agenda": True,
            "mesmo_decodificador": True,
            "treinos_na_mesma_execucao": False,
            "diferenca_parametros_percentual": abs(
                float(arquitetura_e["parametros"])
                / float(arquitetura_d["parametros"])
                - 1.0
            )
            * 100.0,
            "observacao": (
                "A baseline troca largura por conectividade densa. Isso "
                "iguala o orçamento de parâmetros, não a dimensão interna. "
                "O esparso usa o treino oficial já concluído sob o mesmo "
                "protocolo; os tempos de treino não são simultâneos."
            ),
        },
        "metricas": metricas,
        "resumo_numerico": resumo_numerico,
        "qualidade_esparsa_preservada": qualidade_preservada,
        "vantagens_esparsas_de_eficiencia": vantagens_eficiencia,
        "vantagens_de_eficiencia_avaliadas": len(eficiencia),
        "vantagem_pratica_esparsa_sustentada": vantagem_pratica,
        "modelo_superior_neste_protocolo": (
            "esparso" if vantagem_pratica else "denso"
        ),
        "relatorios": {
            "esparso": "esparso/relatorio.json",
            "denso": "denso/relatorio.json",
        },
        "limite": (
            "O resultado vale para este domínio, hardware e orçamento. "
            "Uma única configuração não prova superioridade universal."
        ),
    }


def _numero_pt(valor: float, casas: int = 2) -> str:
    if casas == 0:
        return f"{int(round(valor)):,}".replace(",", ".")
    inteiro, decimal = f"{valor:.{casas}f}".split(".")
    milhares = f"{int(inteiro):,}".replace(",", ".")
    return f"{milhares},{decimal}"


def _valor_markdown(chave: str, valor: float) -> str:
    if chave in {"parametros", "checkpoint_bytes"}:
        return _numero_pt(valor, 0)
    if chave in {
        "acuracia_token",
        "aprovacao_geracao",
        "recuperacao_campos",
        "repeticao_trigramas",
        "taxa_retentativa",
    }:
        return f"{_numero_pt(valor * 100.0, 2)}%"
    if chave == "ppl":
        return _numero_pt(valor, 6)
    return _numero_pt(valor, 2)


def gerar_markdown_comparacao(
    comparacao: dict[str, object],
) -> str:
    metricas = comparacao["metricas"]
    rotulos = {
        "parametros": "Parâmetros",
        "ppl": "PPL de teste",
        "acuracia_token": "Acurácia de token",
        "aprovacao_geracao": "Aprovação de geração",
        "recuperacao_campos": "Recuperação dos campos",
        "repeticao_trigramas": "Repetição média de trigramas",
        "taxa_retentativa": "Taxa de retentativa",
        "tempo_treino_segundos": "Tempo de treino (s)",
        "vram_treino_mib": "Pico de VRAM no treino (MiB)",
        "forward_tokens_s": "Forward (tokens/s)",
        "vram_forward_mib": "Pico de VRAM no forward (MiB)",
        "geracao_tokens_s": "Geração real (tokens/s)",
        "primeiro_token_ms": "Primeiro token (ms)",
        "checkpoint_bytes": "Checkpoint (bytes)",
    }
    linhas = [
        "# Comparação justa: esparso × denso",
        "",
        "Autor: Paulo Augusto",
        "Ano: 2026",
        "",
        "Mesmos dados, ordem de lotes, cinco épocas, 500 passos por época, "
        "lote 100, AdamW, agenda, semente e decodificador.",
        "",
        "| Métrica | Esparso | Denso | Vencedor |",
        "|---|---:|---:|---|",
    ]
    for chave, rotulo in rotulos.items():
        metrica = metricas[chave]
        linhas.append(
            f"| {rotulo} | "
            f"{_valor_markdown(chave, float(metrica['esparso']))} | "
            f"{_valor_markdown(chave, float(metrica['denso']))} | "
            f"{metrica['vencedor']} |"
        )
    sustentada = comparacao["vantagem_pratica_esparsa_sustentada"]
    resumo = comparacao["resumo_numerico"]
    linhas.extend(
        [
            "",
            "## Conclusão automática",
            "",
            (
                "A vantagem prática esparsa foi sustentada neste protocolo."
                if sustentada
                else (
                    "A vantagem prática esparsa não foi sustentada de forma "
                    "majoritária neste protocolo."
                )
            ),
            "",
            (
                "A baseline densa treinou "
                f"{_numero_pt(float(resumo['aceleracao_treino_denso']))}× "
                "mais "
                "rápido, executou o forward "
                f"{_numero_pt(float(resumo['aceleracao_forward_denso']))}× "
                "mais "
                "rápido e gerou texto "
                f"{_numero_pt(float(resumo['aceleracao_geracao_densa']))}× "
                "mais "
                "rápido."
            ),
            (
                f"O {resumo['vencedor_vram_forward']} usou "
                f"{_numero_pt(float(resumo['reducao_vram_forward_vencedor_percentual']))}% "
                "menos VRAM no forward; o denso usou "
                f"{_numero_pt(float(resumo['reducao_vram_treino_denso_percentual']))}% "
                "menos VRAM no treino."
            ),
            "",
            str(comparacao["limite"]),
            "",
        ]
    )
    return "\n".join(linhas)
