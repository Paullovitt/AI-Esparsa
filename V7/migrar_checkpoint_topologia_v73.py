"""Migra um checkpoint V7.3 legado sem sobrescrever o arquivo original."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73
from src.topologia_checkpoint_v73 import (
    metadados_topologia,
    validar_topologia_checkpoint,
)


RAIZ = Path(__file__).resolve().parent
ENTRADA_PADRAO = RAIZ / "modelos" / "gerador_esparso_v73_bpe8192_5x3000.pt"
SAIDA_PADRAO = (
    RAIZ / "modelos" / "gerador_esparso_v73_bpe8192_5x3000_topologia.pt"
)


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def migrar(entrada: Path, saida: Path) -> str:
    if saida.exists():
        raise FileExistsError(f"saida ja existe; arquivo preservado: {saida}")
    modelo, _, checkpoint = carregar_v73(
        entrada,
        torch.device("cpu"),
        exigir_kernel_cuda=False,
        permitir_checkpoint_legado=True,
    )
    migrado = dict(checkpoint)
    migrado["formato_checkpoint"] = "v73-topologia-v1"
    migrado["topologia"] = metadados_topologia(modelo)
    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = saida.with_name(saida.name + ".tmp")
    try:
        torch.save(migrado, temporario)
        verificado = torch.load(
            temporario,
            map_location="cpu",
            weights_only=True,
        )
        validar_topologia_checkpoint(verificado, modelo)
        os.replace(temporario, saida)
    finally:
        temporario.unlink(missing_ok=True)
    return sha256_arquivo(saida)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=ENTRADA_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    args = parser.parse_args()
    print(migrar(args.entrada, args.saida))


if __name__ == "__main__":
    main()
