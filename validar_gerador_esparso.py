"""Revalida o checkpoint final do gerador sem executar novo treino.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from executar_gerador_esparso import (
    carregar_gerador,
)
from src.modelo_gerador_esparso import ModeloGeradorEsparso
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_gerador_esparso import (
    RESULTADOS_PADRAO,
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    codificar_registros,
    gerar_divisoes_gerador,
    local_do_objeto_consistente,
    problema_recuperado,
    taxa_trigramas_repetidos,
    termo_recuperado,
)


RELATORIO = RESULTADOS_PADRAO / "relatorio.json"
CHECKPOINT_VALIDACAO = RESULTADOS_PADRAO / "epoca_05.pt"
CHECKPOINT_CONTEXTUAL = (
    Path(__file__).resolve().parent
    / "resultados"
    / "camada_contextual_texto_longo"
    / "epoca_05.pt"
)
AMOSTRA_CONTEXTUAL = (
    CHECKPOINT_CONTEXTUAL.parent / "amostra_gerada.json"
)


@torch.inference_mode()
def _benchmark_pareado(
    modelo: torch.nn.Module,
    vocabulario: int,
    dispositivo: torch.device,
) -> dict[str, float]:
    """Mede ambos os modelos com a mesma entrada 8 x 200."""

    entradas = torch.randint(
        4,
        vocabulario,
        (8, 200),
        device=dispositivo,
    )
    for _ in range(3):
        modelo(entradas)
    torch.cuda.synchronize()
    memoria_base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    repeticoes = 8
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        modelo(entradas)
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    return {
        "tokens_por_segundo": entradas.numel() * repeticoes / duracao,
        "latencia_ms": duracao * 1000.0 / repeticoes,
        "vram_temporaria_mib": (
            torch.cuda.max_memory_allocated() - memoria_base
        )
        / 1024**2,
    }


def comparar_contextual_anterior(
    modelo_novo: ModeloGeradorEsparso,
    vocabulario_novo: int,
    dispositivo: torch.device,
    geracao_nova: dict[str, object],
) -> dict[str, object]:
    """Compara qualidade e desempenho sem alterar o checkpoint anterior."""

    # As dependências históricas só são carregadas quando seus artefatos
    # opcionais estão presentes no workspace.
    from src.modelo_contextual_longo_experimental import (
        ConfiguracaoModeloContextualLongo,
        ModeloContextualLongoExperimental,
    )
    from treinar_camada_contextual_rapido import TokenizadorControlado
    from treinar_camada_contextual_texto_longo import (
        gerar_divisoes as gerar_divisoes_contextuais,
    )

    treino, validacao, teste = gerar_divisoes_contextuais()
    tokenizador = TokenizadorControlado(
        [
            str(registro["texto"])
            for registro in treino + validacao + teste
        ]
    )
    checkpoint = torch.load(
        CHECKPOINT_CONTEXTUAL,
        map_location="cpu",
        weights_only=True,
    )
    configuracao = ConfiguracaoModeloContextualLongo(
        **checkpoint["configuracao"]
    )
    modelo_anterior = ModeloContextualLongoExperimental(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo_anterior.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )
    modelo_anterior.eval()
    desempenho_anterior = _benchmark_pareado(
        modelo_anterior,
        tokenizador.tamanho,
        dispositivo,
    )
    del modelo_anterior
    torch.cuda.empty_cache()
    desempenho_novo = _benchmark_pareado(
        modelo_novo,
        vocabulario_novo,
        dispositivo,
    )

    amostra = json.loads(AMOSTRA_CONTEXTUAL.read_text(encoding="utf-8"))
    continuacao = str(amostra["texto_gerado"])
    palavras_chave = [
        str(valor).lower() for valor in teste[0]["palavras_chave"]
    ]
    recuperados = [
        termo_recuperado(valor, continuacao)
        for valor in palavras_chave[:-1]
    ]
    recuperados.append(
        problema_recuperado(palavras_chave[-1], continuacao)
    )
    repeticao_anterior = taxa_trigramas_repetidos(
        TokenizadorPalavras.tokenizar(continuacao)
    )
    return {
        "entrada_benchmark": "lote 8 x 200 tokens, 8 repeticoes",
        "contextual_anterior": {
            "ppl_validacao": float(
                checkpoint["metricas_validacao"]["ppl"]
            ),
            "acuracia_token": float(
                checkpoint["metricas_validacao"]["acuracia_token"]
            ),
            "geracao_amostras": 1,
            "recuperacao_campos_amostra": sum(recuperados)
            / len(recuperados),
            "consistencia_objeto_local_amostra": (
                local_do_objeto_consistente(
                    palavras_chave[2],
                    palavras_chave[3],
                    continuacao,
                )
            ),
            "repeticao_trigramas_amostra": repeticao_anterior,
            "desempenho": desempenho_anterior,
        },
        "gerador_novo": {
            "geracao_amostras": int(
                geracao_nova["amostras_avaliadas"]
            ),
            "recuperacao_campos": float(
                geracao_nova["recuperacao_campos_pedido"]
            ),
            "consistencia_objeto_local": float(
                geracao_nova["consistencia_objeto_local"]
            ),
            "repeticao_trigramas_media": float(
                geracao_nova["repeticao_trigramas_media"]
            ),
            "desempenho": desempenho_novo,
        },
        "razao_velocidade_novo_anterior": (
            desempenho_novo["tokens_por_segundo"]
            / desempenho_anterior["tokens_por_segundo"]
        ),
        "observacao": (
            "A geracao antiga possui somente uma amostra salva; qualidade "
            "livre nao e uma comparacao estatistica pareada."
        ),
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
    comparacao = (
        comparar_contextual_anterior(
            modelo,
            tokenizador.tamanho,
            dispositivo,
            geracao,
        )
        if CHECKPOINT_CONTEXTUAL.exists() and AMOSTRA_CONTEXTUAL.exists()
        else {
            "disponivel": False,
            "motivo": "artefatos contextuais históricos ausentes",
        }
    )

    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    relatorio["avaliacao_final"] = linguagem
    relatorio["geracao_livre"] = geracao
    relatorio["recuperacao"] = {
        "campos_do_pedido": geracao["recuperacao_campos_pedido"],
    }
    relatorio["desempenho"] = desempenho
    relatorio["comparacao_contextual_anterior"] = comparacao
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
