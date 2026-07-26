"""Carrega e avalia o checkpoint canonico do modelo V6.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from src.atencao_causal_texto import (
    CodigoEsparsoCausal,
    ConfiguracaoTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    preparar_candidatos,
)
from src.modelo_v6 import ConfiguracaoFFNTexto, ModeloV6
from treinar_v6 import avaliar, avaliar_geracao


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_PADRAO = RAIZ / "modelos" / "v6_base.pt"


def _filtrar_configuracao(
    valores: dict[str, object],
    classe: type[ConfiguracaoTextoCausal] | type[ConfiguracaoFFNTexto],
) -> dict[str, object]:
    """Ignora metadados extras de checkpoints sem alterar a arquitetura."""

    permitidos = {campo.name for campo in fields(classe)}
    return {
        chave: valor
        for chave, valor in valores.items()
        if chave in permitidos
    }


def carregar_v6(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[ModeloV6, CodigoEsparsoCausal, dict[str, object]]:
    checkpoint = torch.load(
        caminho, map_location=dispositivo, weights_only=False
    )
    if checkpoint.get("modelo") != "V6":
        raise ValueError("o checkpoint informado nao pertence ao modelo V6")
    configuracao_texto = ConfiguracaoTextoCausal(
        **_filtrar_configuracao(
            checkpoint["configuracao_texto"],
            ConfiguracaoTextoCausal,
        )
    )
    configuracao_ffn = ConfiguracaoFFNTexto(
        **_filtrar_configuracao(
            checkpoint["configuracao_ffn"],
            ConfiguracaoFFNTexto,
        )
    )
    vocabulario = criar_vocabulario()
    modelo = ModeloV6(
        vocabulario,
        configuracao_texto,
        semente_topologia=int(checkpoint["semente"]),
        configuracao_ffn=configuracao_ffn,
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()
    codigo = CodigoEsparsoCausal(
        vocabulario, configuracao_texto
    ).to(dispositivo)
    return modelo, codigo, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia o checkpoint canonico V6."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PADRAO,
    )
    parser.add_argument("--amostras", type=int, default=2000)
    argumentos = parser.parse_args()
    if argumentos.amostras < 72:
        raise ValueError("--amostras deve ser pelo menos 72")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para a avaliacao oficial")

    dispositivo = torch.device("cuda")
    modelo, codigo, checkpoint = carregar_v6(
        argumentos.checkpoint.resolve(), dispositivo
    )
    vocabulario = criar_vocabulario()
    dados = gerar_corpus_texto_causal(
        argumentos.amostras,
        303,
        vocabulario,
        modelo.configuracao_texto,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    dados, descritores = preparar_candidatos(dados, codigo)
    resultado = {
        "modelo": "V6",
        "checkpoint": str(argumentos.checkpoint.resolve()),
        "semente": int(checkpoint["semente"]),
        "epoca": int(checkpoint["epoca"]),
        "arquitetura": modelo.auditoria(),
        "teste": avaliar(modelo, dados, descritores),
        "geracao": avaliar_geracao(modelo, codigo, dados),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
