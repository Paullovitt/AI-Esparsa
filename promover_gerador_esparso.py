"""Promove um experimento aprovado para o checkpoint-base oficial.

A promoção é deliberadamente separada do treino: primeiro o relatório precisa
passar por revalidação e somente depois o checkpoint da época 5 é copiado.
O checkpoint oficial nunca é sobrescrito automaticamente.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
ORIGEM_V73 = (
    RAIZ
    / "resultados"
    / "v73_base"
    / "treino_fp32_5epocas"
    / "epoca_05.pt"
)
RELATORIO_V73 = ORIGEM_V73.parent / "validacao" / "comparacao.json"
DESTINO_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"


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
        "revalidacao_aprovada": bool(
            relatorio.get("revalidacao", {}).get("aprovado")
        ),
        "criterios_completos": bool(criterios)
        and all(bool(valor) for valor in criterios.values()),
        "validador_v3": (
            relatorio.get("revalidacao", {}).get("versao_validador")
            == "3.0.0"
        ),
        "vinte_quatro_saidas_auditaveis": (
            len(
                relatorio.get("geracao_livre", {}).get(
                    "exemplos",
                    [],
                )
            )
            == 24
        ),
        "benchmark_autorregressivo": (
            float(
                relatorio.get(
                    "desempenho_autorregressivo",
                    {},
                ).get("tokens_por_segundo", 0.0)
            )
            > 0.0
        ),
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


def validar_promocao_v73(
    checkpoint: dict[str, object],
    relatorio: dict[str, object],
    *,
    hash_checkpoint: str | None = None,
) -> None:
    """Valida um candidato V7.3 de cinco épocas contra o controle denso."""

    modelos = relatorio.get("modelos", {})
    if not isinstance(modelos, dict):
        modelos = {}
    dados_v73 = modelos.get("v73", {})
    dados_denso = modelos.get("denso", {})
    if not isinstance(dados_v73, dict):
        dados_v73 = {}
    if not isinstance(dados_denso, dict):
        dados_denso = {}
    puro = dados_v73.get("geracao_modelo_puro", {})
    restrito = dados_v73.get("geracao_greedy_restrita", {})
    sistema = dados_v73.get("geracao_livre", {})
    if not isinstance(puro, dict):
        puro = {}
    if not isinstance(restrito, dict):
        restrito = {}
    if not isinstance(sistema, dict):
        sistema = {}
    avaliacao_v73 = dados_v73.get("avaliacao_teste", {})
    avaliacao_densa = dados_denso.get("avaliacao_teste", {})
    if not isinstance(avaliacao_v73, dict):
        avaliacao_v73 = {}
    if not isinstance(avaliacao_densa, dict):
        avaliacao_densa = {}
    desempenho = dados_v73.get("desempenho_autorregressivo", {})
    if not isinstance(desempenho, dict):
        desempenho = {}
    protocolo = desempenho.get("protocolo_estatistico", {})
    if not isinstance(protocolo, dict):
        protocolo = {}
    hash_relatorio = dados_v73.get("checkpoint_sha256")
    verificacoes = {
        "modelo_v73": checkpoint.get("modelo")
        in {
            "gerador-esparso-v73-base",
            "gerador-esparso-v73-treino-fp32",
        },
        "epoca_5": int(checkpoint.get("epoca", 0)) == 5,
        "cinco_epocas": int(checkpoint.get("epocas_planejadas", 0)) == 5,
        "cinquenta_mil_amostras": (
            int(checkpoint.get("dados_treino", 0)) == 50_000
        ),
        "quinhentos_passos": (
            int(checkpoint.get("passos_por_epoca", 0)) == 500
        ),
        "candidato_elegivel": bool(
            checkpoint.get("elegivel_promocao", False)
        ),
        "hash_auditavel": (
            hash_checkpoint is None
            or (
                isinstance(hash_relatorio, str)
                and hash_relatorio == hash_checkpoint
            )
        ),
        "vinte_quatro_saidas_modelo_puro": (
            len(puro.get("exemplos", [])) == 24
        ),
        "vinte_quatro_saidas_restritas": (
            len(restrito.get("exemplos", [])) == 24
        ),
        "vinte_quatro_saidas_sistema": (
            len(sistema.get("exemplos", [])) == 24
        ),
        "sistema_aprovado": (
            float(sistema.get("taxa_aprovacao", 0.0)) == 1.0
        ),
        "benchmark_estatistico": (
            int(protocolo.get("repeticoes", 0)) >= 5
            and int(protocolo.get("aquecimentos", 0)) >= 1
        ),
        "desempenho_medido": (
            float(desempenho.get("tokens_por_segundo", 0.0)) > 0.0
        ),
        "ppl_proxima_ao_denso": (
            float(avaliacao_v73.get("ppl", float("inf")))
            <= float(avaliacao_densa.get("ppl", 0.0)) * 1.05
        ),
    }
    falhas = [nome for nome, passou in verificacoes.items() if not passou]
    if falhas:
        raise RuntimeError(
            "candidato V7.3 nao pode ser promovido; falhas: "
            + ", ".join(falhas)
        )


def _copiar_atomicamente(
    origem: Path,
    destino: Path,
    *,
    substituir: bool,
) -> tuple[str, Path | None]:
    """Copia por arquivo temporário e cria backup antes de substituir."""

    backup: Path | None = None
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        if not substituir:
            raise FileExistsError(
                f"checkpoint-base ja existe: {destino}; use --substituir "
                "somente depois de revisar o relatorio"
            )
        backup = destino.with_suffix(destino.suffix + ".backup")
        if backup.exists():
            raise FileExistsError(f"backup anterior ja existe: {backup}")
        shutil.copy2(destino, backup)
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    try:
        shutil.copy2(origem, temporario)
        if sha256(origem) != sha256(temporario):
            raise RuntimeError("hash da copia temporaria divergiu da origem")
        os.replace(temporario, destino)
    finally:
        temporario.unlink(missing_ok=True)
    return sha256(destino), backup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida e promove explicitamente V6.2 ou V7.3."
    )
    parser.add_argument(
        "--arquitetura",
        choices=("v62", "v73"),
        default="v62",
    )
    parser.add_argument("--origem", type=Path)
    parser.add_argument("--relatorio", type=Path)
    parser.add_argument("--destino", type=Path)
    parser.add_argument(
        "--confirmar-promocao",
        action="store_true",
        help="confirma a copia depois de todos os criterios passarem",
    )
    parser.add_argument(
        "--substituir",
        action="store_true",
        help="preserva a base anterior em .backup antes da troca atomica",
    )
    args = parser.parse_args()
    padroes = {
        "v62": (ORIGEM, RELATORIO, DESTINO),
        "v73": (ORIGEM_V73, RELATORIO_V73, DESTINO_V73),
    }
    origem_padrao, relatorio_padrao, destino_padrao = padroes[
        args.arquitetura
    ]
    origem = args.origem or origem_padrao
    relatorio_caminho = args.relatorio or relatorio_padrao
    destino = args.destino or destino_padrao
    if not origem.exists() or not relatorio_caminho.exists():
        raise FileNotFoundError("checkpoint final ou relatório ausente")

    checkpoint = torch.load(
        origem,
        map_location="cpu",
        weights_only=True,
    )
    relatorio = json.loads(relatorio_caminho.read_text(encoding="utf-8"))
    origem_hash = sha256(origem)
    if args.arquitetura == "v73":
        validar_promocao_v73(
            checkpoint,
            relatorio,
            hash_checkpoint=origem_hash,
        )
    else:
        validar_promocao(checkpoint, relatorio)
    if not args.confirmar_promocao:
        print(
            json.dumps(
                {
                    "status": "validado_sem_promover",
                    "arquitetura": args.arquitetura,
                    "checkpoint": str(origem.resolve()),
                    "sha256": origem_hash,
                    "proximo_passo": "--confirmar-promocao",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    destino_hash, backup = _copiar_atomicamente(
        origem,
        destino,
        substituir=args.substituir,
    )
    if origem_hash != destino_hash:
        raise RuntimeError("hash do checkpoint promovido divergiu da origem")
    print(
        json.dumps(
            {
                "checkpoint_base": str(destino.resolve()),
                "sha256": destino_hash,
                "modelo": "Gerador Esparso Coerente",
                "arquitetura": args.arquitetura,
                "backup": str(backup.resolve()) if backup else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
