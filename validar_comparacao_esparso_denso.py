"""Revalida de forma independente a baseline densa e sua comparação.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from comparar_esparso_denso import (
    CHECKPOINT_ESPARSO_OFICIAL,
    construir_comparacao,
    gerar_markdown_comparacao,
)
from executar_gerador_esparso import carregar_gerador
from src.modelo_gerador_denso import (
    ConfiguracaoGeradorDenso,
    ModeloGeradorDenso,
)
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from treinar_gerador_esparso import (
    QUANTIDADES_PADRAO,
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    benchmark_autorregressivo,
    codificar_registros,
    gerar_divisoes_gerador,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS_PADRAO = (
    RAIZ / "resultados" / "comparacao_esparso_denso_50k"
)


def _carregar_checkpoints(
    resultados: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """Confere as cinco épocas com a carga segura exigida pelo projeto."""

    checkpoints: list[dict[str, object]] = []
    for epoca in range(1, 6):
        caminho = resultados / "denso" / f"epoca_{epoca:02d}.pt"
        checkpoint = torch.load(
            caminho,
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["modelo"] != "baseline-comparativa-densa":
            raise ValueError(f"modelo inesperado em {caminho.name}")
        if int(checkpoint["epoca"]) != epoca:
            raise ValueError(f"epoca inesperada em {caminho.name}")
        if int(checkpoint["passos_por_epoca"]) != 500:
            raise ValueError(f"protocolo divergente em {caminho.name}")
        checkpoints.append(checkpoint)
    return checkpoints[-1], checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTADOS_PADRAO,
    )
    args = parser.parse_args()
    relatorio_denso_caminho = (
        args.resultados / "denso" / "relatorio.json"
    )
    relatorio_comparativo_caminho = (
        args.resultados / "comparacao.json"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para a revalidacao")
    if (
        not relatorio_denso_caminho.exists()
        or not relatorio_comparativo_caminho.exists()
    ):
        raise FileNotFoundError("relatorios comparativos ausentes")

    checkpoint, checkpoints = _carregar_checkpoints(args.resultados)
    dispositivo = torch.device("cuda")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    configuracao = ConfiguracaoGeradorDenso(
        **checkpoint["configuracao"]
    )
    modelo = ModeloGeradorDenso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()

    _, _, teste = gerar_divisoes_gerador(QUANTIDADES_PADRAO)
    codificados_teste = codificar_registros(teste, tokenizador)
    linguagem = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        100,
        dispositivo,
    )
    geracao = avaliar_geracao_livre(
        modelo,
        tokenizador,
        teste,
        dispositivo,
        quantidade=24,
    )
    desempenho_denso = benchmark(
        modelo,
        tokenizador.tamanho,
        dispositivo,
    )
    autorregressivo_denso = benchmark_autorregressivo(
        modelo,
        tokenizador,
        str(teste[0]["pedido"]),
        dispositivo,
    )
    relatorio_denso = json.loads(
        relatorio_denso_caminho.read_text(encoding="utf-8")
    )
    ppl_registrada = float(relatorio_denso["avaliacao_final"]["ppl"])
    criterios = {
        "cinco_checkpoints_weights_only": len(checkpoints) == 5,
        "carga_estrita": True,
        "ppl_reproduzida": math.isclose(
            linguagem["ppl"],
            ppl_registrada,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ),
        "vinte_quatro_geracoes": (
            int(geracao["amostras_avaliadas"]) == 24
        ),
        "aprovacao_100": geracao["taxa_aprovacao"] == 1.0,
        "recuperacao_100": (
            geracao["recuperacao_campos_pedido"] == 1.0
        ),
        "consistencia_objeto_local_100": (
            geracao["consistencia_objeto_local"] == 1.0
        ),
        "consistencia_acoes_100": (
            geracao["consistencia_acoes"] == 1.0
        ),
        "todas_as_saidas_salvas": len(geracao["exemplos"]) == 24,
    }
    resultado = {
        "weights_only": True,
        "checkpoint_epoca": int(checkpoint["epoca"]),
        "ppl_recalculada": linguagem["ppl"],
        "ppl_registrada": ppl_registrada,
        "diferenca_absoluta_ppl": abs(
            linguagem["ppl"] - ppl_registrada
        ),
        "geracao_recalculada": {
            chave: valor
            for chave, valor in geracao.items()
            if chave not in {"exemplos", "falhas"}
        },
        "criterios": criterios,
        "aprovado": all(criterios.values()),
    }
    relatorio_denso["versao"] = VERSAO_PROJETO
    relatorio_denso["avaliacao_final"] = linguagem
    relatorio_denso["geracao_livre"] = geracao
    relatorio_denso["desempenho_forward"] = desempenho_denso
    relatorio_denso["desempenho_autorregressivo"] = (
        autorregressivo_denso
    )
    relatorio_denso_caminho.write_text(
        json.dumps(relatorio_denso, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    modelo_esparso, tokenizador_esparso, _ = carregar_gerador(
        CHECKPOINT_ESPARSO_OFICIAL,
        dispositivo,
        classe_modelo=ModeloGeradorEsparsoV62,
    )
    if tokenizador_esparso.id_para_token != tokenizador.id_para_token:
        raise RuntimeError("tokenizador esparso diverge da baseline densa")
    linguagem_esparsa = avaliar_linguagem(
        modelo_esparso,
        codificados_teste,
        tokenizador.pad_id,
        100,
        dispositivo,
    )
    geracao_esparsa = avaliar_geracao_livre(
        modelo_esparso,
        tokenizador,
        teste,
        dispositivo,
        quantidade=24,
    )
    desempenho_esparso = benchmark(
        modelo_esparso,
        tokenizador.tamanho,
        dispositivo,
    )
    autorregressivo_esparso = benchmark_autorregressivo(
        modelo_esparso,
        tokenizador,
        str(teste[0]["pedido"]),
        dispositivo,
    )
    relatorio_esparso = json.loads(
        (args.resultados / "esparso" / "relatorio.json").read_text(
            encoding="utf-8"
        )
    )
    relatorio_esparso["versao"] = VERSAO_PROJETO
    relatorio_esparso["arquitetura"] = modelo_esparso.auditoria()
    relatorio_esparso["avaliacao_final"] = linguagem_esparsa
    relatorio_esparso["geracao_livre"] = geracao_esparsa
    relatorio_esparso["desempenho_forward"] = desempenho_esparso
    relatorio_esparso["desempenho_autorregressivo"] = (
        autorregressivo_esparso
    )
    (
        args.resultados / "esparso" / "relatorio.json"
    ).write_text(
        json.dumps(relatorio_esparso, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparacao = construir_comparacao(
        relatorio_esparso,
        relatorio_denso,
    )
    comparacao["revalidacao_independente"] = resultado
    relatorio_comparativo_caminho.write_text(
        json.dumps(comparacao, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.resultados / "COMPARACAO.md").write_text(
        gerar_markdown_comparacao(comparacao),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **resultado,
                "primeiro_texto_denso": geracao["exemplos"][0],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not resultado["aprovado"]:
        raise RuntimeError("revalidacao comparativa reprovada")


if __name__ == "__main__":
    main()
