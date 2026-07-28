"""Carrega o gerador esparso-base e produz um relato condicionado.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.decodificador_contextual import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_gerador_esparso import (
    extrair_campos_pedido,
    gerar_relato_validado,
)


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_PADRAO = (
    RAIZ
    / "modelos"
    / "gerador_esparso_base.pt"
)
PROMPT_PADRAO = (
    "Pedido: escreva um relato sobre bruno, com ajuda de tiago, para "
    "preparar uma feira de ciencias. Inclua a caixa na estante e o "
    "problema uma janela quebrada. Texto:"
)


def carregar_gerador(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[
    ModeloGeradorEsparso,
    TokenizadorPalavras,
    dict[str, object],
]:
    """Reconstroi estritamente arquitetura e vocabulario do checkpoint."""

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    modelos_aceitos = {
        "gerador-esparso-coerente-base",
        "gerador-esparso-coerente-experimental",
    }
    if checkpoint.get("modelo") not in modelos_aceitos:
        raise ValueError("checkpoint nao pertence ao gerador esparso")
    if int(checkpoint.get("epoca", 0)) != 5:
        raise ValueError("o checkpoint final deve corresponder a epoca 5")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    configuracao = ConfiguracaoGeradorEsparso(
        **checkpoint["configuracao"]
    )
    modelo = ModeloGeradorEsparso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )
    modelo.eval()
    return modelo, tokenizador, checkpoint


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=PROMPT_PADRAO)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PADRAO,
    )
    parser.add_argument(
        "--amostrar",
        action="store_true",
        help="usa amostragem Top-K; por padrao a saida e deterministica",
    )
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    modelo, tokenizador, _ = carregar_gerador(
        args.checkpoint,
        dispositivo,
    )
    campos = extrair_campos_pedido(args.prompt)
    if campos is not None and not args.amostrar:
        texto, _ = gerar_relato_validado(
            modelo,
            tokenizador,
            args.prompt,
            campos,
            dispositivo,
        )
    else:
        configuracao = ConfiguracaoDecodificacao(
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
        )
        texto = gerar_controlado(
            modelo,
            tokenizador,
            args.prompt,
            dispositivo,
            configuracao=configuracao,
            amostrar=args.amostrar,
        )
    print(texto)


if __name__ == "__main__":
    main()
