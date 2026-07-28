"""Revalida o checkpoint final do gerador sem executar novo treino.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from executar_gerador_esparso import (
    carregar_gerador,
)
from treinar_gerador_esparso import (
    RESULTADOS_PADRAO,
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    codificar_registros,
    gerar_divisoes_gerador,
)


RELATORIO = RESULTADOS_PADRAO / "relatorio.json"
CHECKPOINT_VALIDACAO = RESULTADOS_PADRAO / "epoca_05.pt"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para o benchmark")
    if not CHECKPOINT_VALIDACAO.exists() or not RELATORIO.exists():
        raise FileNotFoundError("checkpoint final ou relatorio ausente")

    dispositivo = torch.device("cuda")
    modelo, tokenizador, checkpoint = carregar_gerador(
        CHECKPOINT_VALIDACAO,
        dispositivo,
    )
    _, _, teste = gerar_divisoes_gerador()
    codificados_teste = codificar_registros(teste, tokenizador)
    linguagem = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        32,
        dispositivo,
    )
    geracao = avaliar_geracao_livre(
        modelo,
        tokenizador,
        teste,
        dispositivo,
    )
    desempenho = benchmark(
        modelo,
        tokenizador.tamanho,
        dispositivo,
    )
    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    relatorio["avaliacao_final"] = linguagem
    relatorio["geracao_livre"] = geracao
    relatorio["recuperacao"] = {
        "campos_do_pedido": geracao["recuperacao_campos_pedido"],
    }
    relatorio["desempenho"] = desempenho
    relatorio["revalidacao"] = {
        "checkpoint_epoca": int(checkpoint["epoca"]),
        "weights_only": True,
        "criterios": {
            "ppl_maxima_1_20": linguagem["ppl"] <= 1.20,
            "completude_100": geracao["taxa_completude"] == 1.0,
            "recuperacao_100": (
                geracao["recuperacao_campos_pedido"] == 1.0
            ),
            "consistencia_objeto_local_100": (
                geracao["consistencia_objeto_local"] == 1.0
            ),
            "consistencia_acoes_100": (
                geracao["consistencia_acoes"] == 1.0
            ),
            "repeticao_trigramas_media_ate_1_5_porcento": (
                geracao["repeticao_trigramas_media"] <= 0.015
            ),
            "nenhum_trigrama_repetido_mais_de_duas_vezes": (
                geracao["frequencia_maxima_trigrama"] <= 2
            ),
            "todas_geracoes_com_2_mil_caracteres": (
                geracao["caracteres_minimos"] >= 2_000
            ),
            "sem_vazamento_pergunta": (
                geracao["taxa_vazamento_pergunta"] == 0.0
            ),
        },
    }
    relatorio["revalidacao"]["aprovado"] = all(
        relatorio["revalidacao"]["criterios"].values()
    )
    RELATORIO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "avaliacao_final": linguagem,
                "geracao_livre": {
                    chave: valor
                    for chave, valor in geracao.items()
                    if chave != "exemplos"
                },
                "criterios": relatorio["revalidacao"]["criterios"],
                "aprovado": relatorio["revalidacao"]["aprovado"],
                "exemplos": geracao["exemplos"][:3],
                "relatorio": str(Path(RELATORIO).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
