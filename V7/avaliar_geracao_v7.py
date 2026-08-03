"""Avalia diversidade e repeticao da geracao V7.3 em um conjunto fixo.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch

from executar_gerador_esparso_v73 import (
    CHECKPOINT_PADRAO,
    carregar_v73,
    formatar_prompt,
)
from src.geracao_textual_v73 import (
    ConfiguracaoDecodificacaoV73,
    gerar_v73,
    resumir_metricas_geracao,
)


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def carregar_prompts(caminho: Path, quantidade: int, semente: int) -> list[str]:
    if quantidade <= 0:
        raise ValueError("quantidade deve ser positiva")
    prompts: list[str] = []
    with caminho.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            item = json.loads(linha)
            prompt = item.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
    if len(prompts) < quantidade:
        raise ValueError("arquivo nao possui prompts suficientes")
    random.Random(semente).shuffle(prompts)
    return prompts[:quantidade]


def avaliar_modo(
    modelo,
    tokenizador,
    prompts: list[str],
    configuracao: ConfiguracaoDecodificacaoV73,
    maximo_tokens: int,
    dispositivo: torch.device,
) -> dict[str, Any]:
    amostras: list[dict[str, Any]] = []
    for indice, prompt in enumerate(prompts):
        prefixo = formatar_prompt(prompt)
        ids = tokenizador.codificar(prefixo, bos=True, eos=False)
        entrada = torch.tensor([ids], dtype=torch.long, device=dispositivo)
        config_amostra = ConfiguracaoDecodificacaoV73(
            **{
                **asdict(configuracao),
                "semente": configuracao.semente + indice,
            }
        )
        gerados, medidas = gerar_v73(
            modelo,
            entrada,
            maximo_tokens,
            tokenizador.eos_id,
            config_amostra,
        )
        amostras.append(
            {
                "indice": indice,
                "prompt": prompt,
                "texto": tokenizador.decodificar(
                    gerados[0, entrada.shape[1] :].tolist()
                ),
                **medidas,
            }
        )
    return {
        "configuracao": asdict(configuracao),
        "resumo": resumir_metricas_geracao(amostras),
        "amostras": amostras,
    }



def avaliar_criterios(
    resumo: dict[str, float],
    *,
    maximo_fracao_repetidos: float,
    maximo_repeticao_consecutiva: float,
    minimo_distinct_2: float,
    minimo_taxa_eos: float,
) -> dict[str, Any]:
    """Aplica limites explícitos sem confundir PPL com qualidade gerativa."""

    verificacoes = {
        "fracao_tokens_repetidos": (
            resumo.get("fracao_tokens_repetidos_media", 1.0)
            <= maximo_fracao_repetidos
        ),
        "repeticao_consecutiva": (
            resumo.get("maior_repeticao_consecutiva_media", float("inf"))
            <= maximo_repeticao_consecutiva
        ),
        "distinct_2": (
            resumo.get("distinct_2_media", 0.0) >= minimo_distinct_2
        ),
        "taxa_eos": resumo.get("taxa_eos", 0.0) >= minimo_taxa_eos,
    }
    return {
        "aprovado": all(verificacoes.values()),
        "verificacoes": verificacoes,
        "limites": {
            "maximo_fracao_repetidos": maximo_fracao_repetidos,
            "maximo_repeticao_consecutiva": maximo_repeticao_consecutiva,
            "minimo_distinct_2": minimo_distinct_2,
            "minimo_taxa_eos": minimo_taxa_eos,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PADRAO)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True)
    parser.add_argument("--quantidade", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--semente", type=int, default=20260802)
    parser.add_argument("--permitir-fallback", action="store_true")
    parser.add_argument("--max-fracao-repetidos", type=float, default=0.50)
    parser.add_argument("--max-repeticao-consecutiva", type=float, default=3.0)
    parser.add_argument("--min-distinct-2", type=float, default=0.60)
    parser.add_argument("--min-taxa-eos", type=float, default=0.20)
    args = parser.parse_args()

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, tokenizador, checkpoint = carregar_v73(
        args.checkpoint,
        dispositivo,
        exigir_kernel_cuda=(
            dispositivo.type == "cuda" and not args.permitir_fallback
        ),
    )
    prompts = carregar_prompts(args.prompts_jsonl, args.quantidade, args.semente)
    configuracoes = {
        "greedy": ConfiguracaoDecodificacaoV73.greedy_puro(),
        "amostragem": ConfiguracaoDecodificacaoV73(
            modo="amostragem",
            temperatura=0.8,
            top_k=40,
            top_p=0.90,
            penalidade_repeticao=1.15,
            ngrama_sem_repeticao=3,
            maximo_repeticao_consecutiva=3,
            semente=args.semente,
        ),
    }
    modos = {
        nome: avaliar_modo(
            modelo,
            tokenizador,
            prompts,
            configuracao,
            args.max_tokens,
            dispositivo,
        )
        for nome, configuracao in configuracoes.items()
    }
    criterios = avaliar_criterios(
        modos["amostragem"]["resumo"],
        maximo_fracao_repetidos=args.max_fracao_repetidos,
        maximo_repeticao_consecutiva=args.max_repeticao_consecutiva,
        minimo_distinct_2=args.min_distinct_2,
        minimo_taxa_eos=args.min_taxa_eos,
    )
    resultado = {
        "status": "aprovado" if criterios["aprovado"] else "reprovado",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_arquivo(args.checkpoint),
        "checkpoint_modelo": checkpoint.get("modelo"),
        "checkpoint_epoca": checkpoint.get("epoca"),
        "prompts_jsonl": args.prompts_jsonl.name,
        "prompts_sha256": sha256_arquivo(args.prompts_jsonl),
        "quantidade_prompts": len(prompts),
        "maximo_tokens": args.max_tokens,
        "criterios_qualidade_amostragem": criterios,
        "modos": modos,
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = args.saida.with_suffix(args.saida.suffix + ".tmp")
    temporario.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(args.saida)
    print(args.saida)


if __name__ == "__main__":
    main()