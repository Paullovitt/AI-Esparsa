"""Avalia separadamente o modelo puro e o decodificador controlado.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import torch

from .decodificador_gerador import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from .tokenizador_palavras import TokenizadorPalavras
from .validacao_gerador import (
    SEMENTE,
    _medir_continuacao,
    extrair_continuacao,
)


def _configuracao_greedy(*, restricoes: bool) -> ConfiguracaoDecodificacao:
    """Reproduz o relato longo alterando somente as restrições causais."""

    return ConfiguracaoDecodificacao(
        maximo_tokens=500,
        minimo_tokens_frase=300,
        ngrama_bloqueado=0,
        penalidade_repeticao=0.0,
        janela_penalidade=14,
        temperatura=0.58,
        top_k_amostragem=5,
        parar_apos_frase=True,
        frases_para_encerrar=24,
        bloquear_trigrama_adicional=False,
        aplicar_controle_de_repeticao=restricoes,
        aplicar_restricoes_de_dominio=restricoes,
    )


def _aprovado(medidas: dict[str, object]) -> bool:
    return (
        bool(medidas["completo"])
        and float(medidas["cobertura_palavras_chave"]) == 1.0
        and float(medidas["trigramas_repetidos"]) <= 0.05
        and bool(medidas["local_do_objeto_consistente"])
        and bool(medidas["acoes_consistentes"])
        and bool(medidas["concordancia_genero"])
    )


def _resumir(
    resultados: list[dict[str, object]],
    *,
    restricoes: bool,
) -> dict[str, object]:
    """Consolida as mesmas métricas para os dois modos de geração."""

    if not resultados:
        raise ValueError("a avaliação greedy exige ao menos uma amostra")
    total = len(resultados)
    chaves_booleanas = {
        "taxa_completude": "completo",
        "taxa_aprovacao": "aprovado",
        "taxa_vazamento_pergunta": "vazou_bloco_pergunta",
        "consistencia_objeto_local": "local_do_objeto_consistente",
        "consistencia_acoes": "acoes_consistentes",
        "concordancia_genero": "concordancia_genero",
    }
    resumo: dict[str, object] = {
        "modo": (
            "greedy_com_restricoes"
            if restricoes
            else "greedy_modelo_puro"
        ),
        "restricoes_decodificador_ativas": restricoes,
        "amostras_avaliadas": total,
        "tokens_medios": (
            sum(int(item["tokens"]) for item in resultados) / total
        ),
        "caracteres_minimos": min(
            int(item["caracteres"]) for item in resultados
        ),
        "caracteres_medios": (
            sum(int(item["caracteres"]) for item in resultados) / total
        ),
        "frases_medias": (
            sum(int(item["frases"]) for item in resultados) / total
        ),
        "recuperacao_campos_pedido": (
            sum(
                float(item["cobertura_palavras_chave"])
                for item in resultados
            )
            / total
        ),
        "repeticao_trigramas_media": (
            sum(float(item["trigramas_repetidos"]) for item in resultados)
            / total
        ),
        "frequencia_maxima_trigrama": max(
            int(item["frequencia_maxima_trigrama"])
            for item in resultados
        ),
        "exemplos": resultados,
        "falhas": [
            item for item in resultados if not bool(item["aprovado"])
        ],
    }
    for metrica, chave in chaves_booleanas.items():
        resumo[metrica] = (
            sum(bool(item[chave]) for item in resultados) / total
        )

    por_regra: dict[str, dict[str, int]] = {}
    for item in resultados:
        metricas = item["intervencoes_decodificador"]
        if not isinstance(metricas, dict):
            continue
        regras = metricas.get("por_regra", {})
        if not isinstance(regras, dict):
            continue
        for nome, dados in regras.items():
            if not isinstance(dados, dict):
                continue
            acumulado = por_regra.setdefault(
                str(nome),
                {
                    "aplicacoes": 0,
                    "tokens_bloqueados": 0,
                    "logits_alterados": 0,
                    "mudancas_argmax": 0,
                },
            )
            for chave in acumulado:
                acumulado[chave] += int(dados.get(chave, 0))
    resumo["intervencoes_decodificador"] = {
        "tokens_bloqueados": sum(
            dados["tokens_bloqueados"] for dados in por_regra.values()
        ),
        "logits_alterados": sum(
            dados["logits_alterados"] for dados in por_regra.values()
        ),
        "mudancas_argmax": sum(
            dados["mudancas_argmax"] for dados in por_regra.values()
        ),
        "por_regra": por_regra,
    }
    return resumo


@torch.inference_mode()
def avaliar_geracao_greedy(
    modelo: torch.nn.Module,
    tokenizador: TokenizadorPalavras,
    registros: list[dict[str, object]],
    dispositivo: torch.device,
    *,
    quantidade: int = 24,
    restricoes: bool,
) -> dict[str, object]:
    """Gera uma vez por prompt, sem amostragem nem retentativa."""

    configuracao = _configuracao_greedy(restricoes=restricoes)
    resultados: list[dict[str, object]] = []
    for indice, registro in enumerate(registros[:quantidade]):
        palavras_chave = [
            str(valor).lower() for valor in registro["palavras_chave"]
        ]
        intervencoes: dict[str, object] = {}
        texto = gerar_controlado(
            modelo,
            tokenizador,
            str(registro["pedido"]),
            dispositivo,
            configuracao=configuracao,
            amostrar=False,
            semente=SEMENTE + indice,
            metricas_decodificacao=intervencoes,
        )
        continuacao = extrair_continuacao(texto)
        medidas = _medir_continuacao(
            tokenizador,
            continuacao,
            palavras_chave,
        )
        resultados.append(
            {
                "pedido": registro["pedido"],
                "texto": continuacao,
                **medidas,
                "aprovado": _aprovado(medidas),
                "retentativas": 0,
                "intervencoes_decodificador": intervencoes,
            }
        )
    return _resumir(resultados, restricoes=restricoes)
