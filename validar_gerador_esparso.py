"""Revalida o checkpoint final do gerador sem executar novo treino.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from executar_gerador_esparso import (
    carregar_gerador,
)
from src.documentacao_metricas import sincronizar_metricas_documentacao
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from treinar_gerador_esparso import (
    RESULTADOS_PADRAO,
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    benchmark_autorregressivo,
    codificar_registros,
    gerar_divisoes_gerador,
    local_do_objeto_consistente,
    validar_prompt_publico,
)


RELATORIO = RESULTADOS_PADRAO / "relatorio.json"
CHECKPOINT_VALIDACAO = RESULTADOS_PADRAO / "epoca_05.pt"


def avaliar_contratos_adversariais(
    tokenizador: TokenizadorPalavras,
) -> dict[str, bool]:
    """Executa casos que já escaparam do validador ou degradaram a entrada."""

    contradicao = (
        "o livro foi levado para o deposito, mas o livro continuava "
        "na sala."
    )
    distrator_valido = (
        "o documento foi levado para o escritorio, sem interferir no "
        "local reservado para o relatorio. o relatorio ficou na cozinha."
    )
    prompt_oov = (
        "Pedido: escreva um relato sobre joão, com ajuda de pedro, para "
        "lançar um foguete. Inclua o telescópio no laboratório e o "
        "problema um vazamento de gás. Texto:"
    )
    oov_rejeitado = False
    formato_rejeitado = False
    ordem_rejeitada = False
    try:
        validar_prompt_publico(prompt_oov, tokenizador)
    except ValueError:
        oov_rejeitado = True
    try:
        validar_prompt_publico("texto sem campos obrigatorios", tokenizador)
    except ValueError:
        formato_rejeitado = True
    try:
        validar_prompt_publico(
            (
                "Inclua a caixa na sala. Pedido: escreva um relato sobre "
                "bruno, com ajuda de tiago, para organizar a atividade. "
                "Texto:"
            ),
            tokenizador,
        )
    except ValueError:
        ordem_rejeitada = True
    return {
        "contradicao_livro_deposito_rejeitada": not (
            local_do_objeto_consistente("livro", "sala", contradicao)
        ),
        "distrator_de_outro_objeto_aceito": (
            local_do_objeto_consistente(
                "relatorio",
                "cozinha",
                distrator_valido,
            )
        ),
        "prompt_oov_rejeitado": oov_rejeitado,
        "prompt_sem_campos_rejeitado": formato_rejeitado,
        "prompt_com_ordem_invalida_rejeitado": ordem_rejeitada,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para o benchmark")
    if not CHECKPOINT_VALIDACAO.exists() or not RELATORIO.exists():
        raise FileNotFoundError("checkpoint final ou relatorio ausente")

    dispositivo = torch.device("cuda")
    modelo, tokenizador, checkpoint = carregar_gerador(
        CHECKPOINT_VALIDACAO,
        dispositivo,
        classe_modelo=ModeloGeradorEsparsoV62,
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
    desempenho_autorregressivo = benchmark_autorregressivo(
        modelo,
        tokenizador,
        str(teste[0]["pedido"]),
        dispositivo,
    )
    adversarial = avaliar_contratos_adversariais(tokenizador)
    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    relatorio["versao"] = VERSAO_PROJETO
    relatorio["configuracao"] = asdict(modelo.configuracao)
    if "arquitetura" in relatorio:
        relatorio["arquitetura_treino"] = relatorio.pop("arquitetura")
    relatorio["arquitetura_runtime"] = modelo.auditoria()
    relatorio["ambiente_revalidacao"] = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(dispositivo),
    }
    relatorio["avaliacao_final"] = linguagem
    relatorio["geracao_livre"] = geracao
    relatorio["recuperacao"] = {
        "campos_do_pedido": geracao["recuperacao_campos_pedido"],
    }
    relatorio.pop("desempenho", None)
    relatorio["desempenho_forward"] = desempenho
    relatorio["desempenho_autorregressivo"] = (
        desempenho_autorregressivo
    )
    relatorio["avaliacao_adversarial"] = adversarial
    relatorio["revalidacao"] = {
        "versao_validador": "2.0.0",
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
            "todas_as_24_geracoes_salvas": (
                len(geracao["exemplos"])
                == int(geracao["amostras_avaliadas"])
                == 24
            ),
            "contratos_adversariais_aprovados": all(
                adversarial.values()
            ),
        },
    }
    relatorio["revalidacao"]["aprovado"] = all(
        relatorio["revalidacao"]["criterios"].values()
    )
    if "promocao" in relatorio:
        relatorio["promocao"]["revalidacao_atual_aprovada"] = (
            relatorio["revalidacao"]["aprovado"]
        )
    RELATORIO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # O relatório é a fonte única para medições que variam entre execuções.
    sincronizar_metricas_documentacao(relatorio, Path(__file__).parent)
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
