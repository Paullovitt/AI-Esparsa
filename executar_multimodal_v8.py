"""Executa um checkpoint treinado da V8 com uma imagem e um prompt.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dados_multimodais_v8 import carregar_imagem_rgb
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_multimodal_v8 import (
    ConfiguracaoMultimodalV8,
    ModeloMultimodalEsparsoV8,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.tokenizador_palavras import TokenizadorPalavras


def carregar_v8(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[
    ModeloMultimodalEsparsoV8,
    TokenizadorPalavras,
    dict,
]:
    """Reconstrói estritamente a V8 usando carregamento seguro."""

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    if checkpoint.get("modelo") != "gerador-esparso-v8-multimodal":
        raise ValueError("checkpoint nao pertence a V8 multimodal")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    modelo = ModeloMultimodalEsparsoV8(
        tokenizador.tamanho,
        tokenizador.pad_id,
        ConfiguracaoGeradorEsparso(**checkpoint["configuracao"]),
        int(checkpoint["semente"]),
        ConfiguracaoRoteamentoV73(
            **checkpoint["configuracao_roteamento"]
        ),
        ConfiguracaoMultimodalV8(
            **checkpoint["configuracao_multimodal"]
        ),
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()
    return modelo, tokenizador, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imagem", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    modelo, tokenizador, _ = carregar_v8(
        args.checkpoint,
        dispositivo,
    )
    tokenizador.validar_texto_no_vocabulario(
        f"pergunta: {args.prompt} resposta:"
    )
    prefixo = tokenizador.codificar(
        f"pergunta: {args.prompt} resposta:",
        bos=True,
        eos=False,
    )
    tokens = torch.tensor([prefixo], device=dispositivo)
    configuracao = modelo.configuracao_multimodal
    imagem = carregar_imagem_rgb(
        args.imagem,
        configuracao.altura_imagem,
        configuracao.largura_imagem,
    ).unsqueeze(0).to(dispositivo)
    gerados = modelo.gerar_greedy(
        tokens,
        imagem,
        args.max_tokens,
        tokenizador.eos_id,
    )
    resposta = tokenizador.decodificar(
        gerados[0, len(prefixo):].tolist()
    )
    print(resposta)


if __name__ == "__main__":
    main()
