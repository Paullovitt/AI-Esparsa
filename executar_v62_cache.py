"""Alias de execução explícita da V6.2-base com cache causal incremental.

O checkpoint oficial é lido com ``weights_only=True`` e nunca modificado. A
V6.2 reutiliza exatamente os pesos e a topologia esparsa de referência.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from executar_gerador_esparso import (
    CHECKPOINT_PADRAO,
    PROMPT_PADRAO,
    carregar_gerador,
)
from src.decodificador_gerador import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from treinar_gerador_esparso import (
    FRASES_GERACAO,
    gerar_relato_validado,
    validar_prompt_publico,
)


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
        help="usa amostragem Top-K em vez da saida deterministica",
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
        classe_modelo=ModeloGeradorEsparsoV62,
    )
    campos = validar_prompt_publico(args.prompt, tokenizador)
    if not args.amostrar:
        texto, _ = gerar_relato_validado(
            modelo,
            tokenizador,
            args.prompt,
            campos,
            dispositivo,
        )
    else:
        texto = gerar_controlado(
            modelo,
            tokenizador,
            args.prompt,
            dispositivo,
            configuracao=ConfiguracaoDecodificacao(
                maximo_tokens=500,
                minimo_tokens_frase=300,
                ngrama_bloqueado=0,
                penalidade_repeticao=0.0,
                janela_penalidade=14,
                temperatura=0.58,
                top_k_amostragem=5,
                parar_apos_frase=True,
                frases_para_encerrar=FRASES_GERACAO,
                bloquear_trigrama_adicional=False,
            ),
            amostrar=True,
        )
    print(texto)


if __name__ == "__main__":
    main()
