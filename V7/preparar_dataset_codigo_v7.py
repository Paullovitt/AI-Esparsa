"""Prepara BPE-8192 e shards uint16 para a V7 textual.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.dados_binarios_v7 import preparar_dataset_shards


RAIZ = Path(__file__).resolve().parent
DATASET_PADRAO = Path(
    os.environ.get(
        "AI_ESPARSA_DADOS_BRUTOS",
        str(RAIZ / "dados" / "alpaca-cleaned-pt-50k"),
    )
)
DESTINO_PADRAO = Path(
    os.environ.get(
        "AI_ESPARSA_DATASET",
        str(DATASET_PADRAO / "bpe_codigo_8192"),
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PADRAO)
    parser.add_argument("--saida", type=Path, default=DESTINO_PADRAO)
    parser.add_argument("--vocabulario", type=int, default=8192)
    parser.add_argument("--tokens-por-shard", type=int, default=4_000_000)
    args = parser.parse_args()
    manifesto = preparar_dataset_shards(
        args.dataset,
        args.saida,
        tamanho_vocabulario=args.vocabulario,
        tokens_por_shard=args.tokens_por_shard,
    )
    print(manifesto)


if __name__ == "__main__":
    main()
