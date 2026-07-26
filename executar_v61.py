"""Carrega e avalia estritamente o checkpoint-base V6.1.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.atencao_causal_texto import (
    ConfiguracaoTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    selecionar_candidatos_causais,
)
from src.modelo_v61 import (
    CodigoTemporalEsparso,
    ModeloV61Posicional,
)
from testar_v6_isolado import avaliar_geracao_livre
from treinar_v6 import avaliar


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_V61 = RAIZ / "modelos" / "v61_base.pt"


def carregar_v61(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[
    ModeloV61Posicional,
    CodigoTemporalEsparso,
    dict,
]:
    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=False,
    )
    if checkpoint.get("modelo") != "V6.1-base":
        raise ValueError("checkpoint nao pertence a V6.1 Base")
    configuracao = ConfiguracaoTextoCausal(
        **checkpoint["configuracao_texto"]
    )
    vocabulario = criar_vocabulario()
    modelo = ModeloV61Posicional(
        vocabulario,
        configuracao,
        semente_topologia=int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )
    modelo.eval()
    codigo = CodigoTemporalEsparso(
        vocabulario,
        configuracao,
        semente=int(checkpoint["semente"]),
    ).to(dispositivo)
    return modelo, codigo, checkpoint


def main() -> None:
    if not CHECKPOINT_V61.exists():
        raise FileNotFoundError(
            f"checkpoint-base ausente: {CHECKPOINT_V61}"
        )
    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    modelo, codigo, checkpoint = carregar_v61(
        CHECKPOINT_V61,
        dispositivo,
    )
    vocabulario = criar_vocabulario()
    dados = gerar_corpus_texto_causal(
        2000,
        303,
        vocabulario,
        modelo.configuracao_texto,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    descritores = codigo.descritores(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores,
        modelo.configuracao_texto.top_k,
    )
    dados = dados.com_candidatos(candidatos, validos)
    resultado = {
        "modelo": checkpoint["modelo"],
        "checkpoint": str(CHECKPOINT_V61),
        "semente": checkpoint["semente"],
        "epoca": checkpoint["epoca"],
        "arquitetura": modelo.auditoria(),
        "codigo_temporal": codigo.auditoria(),
        "teste": avaliar(modelo, dados, descritores),
        "geracao_livre": avaliar_geracao_livre(
            modelo,
            dados,
            codigo.descritores,
            modelo.configuracao_texto,
            vocabulario,
        ),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
