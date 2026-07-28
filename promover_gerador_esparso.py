"""Promove um experimento aprovado para o checkpoint-base oficial.

A promoção é deliberadamente separada do treino: primeiro o relatório precisa
passar por revalidação e somente depois o checkpoint da época 5 é copiado. A
V6.1 e o rollback V6 nunca são alterados.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


RAIZ = Path(__file__).resolve().parent
ORIGEM = (
    RAIZ
    / "resultados"
    / "gerador_esparso_base_50k"
    / "epoca_05.pt"
)
RELATORIO = ORIGEM.parent / "relatorio.json"
DESTINO = RAIZ / "modelos" / "gerador_esparso_base.pt"


def sha256(caminho: Path) -> str:
    """Calcula o digest do artefato sem carregá-lo inteiro na memória."""

    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def validar_promocao(
    checkpoint: dict[str, object],
    relatorio: dict[str, object],
) -> None:
    """Reprova qualquer candidato que não cumpra o contrato da nova base."""

    criterios = relatorio.get("revalidacao", {}).get("criterios", {})
    verificacoes = {
        "modelo": (
            checkpoint.get("modelo") == "gerador-esparso-coerente-base"
        ),
        "epoca": int(checkpoint.get("epoca", 0)) == 5,
        "cinco_epocas": int(checkpoint.get("epocas_planejadas", 0)) == 5,
        "cinquenta_mil_amostras": (
            int(checkpoint.get("dados_treino", 0)) == 50_000
        ),
        "quinhentos_passos": (
            int(checkpoint.get("passos_por_epoca", 0)) == 500
        ),
        "v61_preservada": bool(checkpoint.get("modelo_v61_preservado")),
        "revalidacao_aprovada": bool(
            relatorio.get("revalidacao", {}).get("aprovado")
        ),
        "criterios_completos": bool(criterios)
        and all(bool(valor) for valor in criterios.values()),
        "geracao_2k": (
            int(
                relatorio.get("geracao_livre", {}).get(
                    "caracteres_minimos",
                    0,
                )
            )
            >= 2_000
        ),
    }
    falhas = [nome for nome, passou in verificacoes.items() if not passou]
    if falhas:
        raise RuntimeError(
            "candidato não pode ser promovido; falhas: "
            + ", ".join(falhas)
        )


def main() -> None:
    if not ORIGEM.exists() or not RELATORIO.exists():
        raise FileNotFoundError("checkpoint final ou relatório ausente")
    if DESTINO.exists():
        raise FileExistsError(
            f"checkpoint-base já existe e não será sobrescrito: {DESTINO}"
        )

    checkpoint = torch.load(
        ORIGEM,
        map_location="cpu",
        weights_only=True,
    )
    relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
    validar_promocao(checkpoint, relatorio)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ORIGEM, DESTINO)
    origem_hash = sha256(ORIGEM)
    destino_hash = sha256(DESTINO)
    if origem_hash != destino_hash:
        # Uma cópia inválida não pode permanecer com aparência de base.
        DESTINO.unlink(missing_ok=True)
        raise RuntimeError("hash do checkpoint promovido divergiu da origem")
    print(
        json.dumps(
            {
                "checkpoint_base": str(DESTINO.resolve()),
                "sha256": destino_hash,
                "v61_preservada": str(
                    (RAIZ / "modelos" / "v61_base.pt").resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
