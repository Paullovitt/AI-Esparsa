"""Sincroniza métricas voláteis do relatório com a documentação.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import re
from pathlib import Path


MARCADOR_INICIO = "<!-- metricas-desempenho:inicio -->"
MARCADOR_FIM = "<!-- metricas-desempenho:fim -->"


def _decimal_pt(valor: float) -> str:
    """Formata duas casas usando ponto de milhar e vírgula decimal."""

    parte_inteira, parte_decimal = f"{valor:.2f}".split(".")
    milhares = f"{int(parte_inteira):,}".replace(",", ".")
    return f"{milhares},{parte_decimal}"


def _inteiro_pt(valor: int) -> str:
    """Formata um inteiro usando ponto como separador de milhar."""

    return f"{valor:,}".replace(",", ".")


def blocos_metricas_desempenho(
    relatorio: dict[str, object],
) -> dict[str, str]:
    """Monta os blocos documentais a partir da fonte oficial."""

    forward = relatorio.get("desempenho_forward")
    autorregressivo = relatorio.get("desempenho_autorregressivo")
    if not isinstance(forward, dict) or not isinstance(
        autorregressivo,
        dict,
    ):
        raise ValueError("relatorio sem metricas de desempenho")

    tokens_forward = _decimal_pt(
        float(forward["tokens_por_segundo"])
    )
    latencia_forward = _decimal_pt(float(forward["latencia_ms"]))
    vram_forward = _decimal_pt(float(forward["vram_pico_mib"]))
    tokens_autorregressivos = _decimal_pt(
        float(autorregressivo["tokens_por_segundo"])
    )
    primeiro_token = _decimal_pt(
        float(autorregressivo["latencia_primeiro_token_ms"])
    )
    tempo_total = _decimal_pt(
        float(autorregressivo["tempo_total_segundos"])
    )
    tokens_gerados = int(float(autorregressivo["tokens_gerados"]))
    caracteres_gerados = int(
        float(autorregressivo["caracteres_gerados"])
    )

    return {
        "README.md": "\n".join(
            [
                "| Medição | Resultado |",
                "|---|---:|",
                (
                    "| Forward paralelo, lote 16 × contexto 640 | "
                    f"{tokens_forward} tokens/s |"
                ),
                f"| Pico de VRAM no forward | {vram_forward} MiB |",
                (
                    "| Geração autorregressiva real | "
                    f"{tokens_autorregressivos} tokens/s |"
                ),
                (
                    "| Latência até o primeiro token | "
                    f"{primeiro_token} ms |"
                ),
                f"| Tempo do relato completo | {tempo_total} s |",
            ]
        ),
        "STATUS.md": "\n".join(
            [
                (
                    f"- Forward paralelo: {tokens_forward} tokens/s e "
                    f"{vram_forward} MiB de VRAM."
                ),
                (
                    "- Geração autorregressiva: "
                    f"{tokens_autorregressivos} tokens/s, primeiro token "
                    f"em {primeiro_token} ms e\n"
                    f"  relato completo em {tempo_total} s."
                ),
            ]
        ),
        "DOCUMENTO_GERADOR_ESPARSO.md": "\n".join(
            [
                (
                    "- forward paralelo, lote 16 e contexto 640: "
                    f"{tokens_forward} tokens/s,\n"
                    f"  {latencia_forward} ms e pico de "
                    f"{vram_forward} MiB;"
                ),
                (
                    "- geração autorregressiva de um relato: "
                    f"{tokens_autorregressivos} tokens/s;"
                ),
                (
                    "- latência até o primeiro token: "
                    f"{primeiro_token} ms;"
                ),
                (
                    f"- tempo total para {tokens_gerados} tokens e "
                    f"{_inteiro_pt(caracteres_gerados)} caracteres: "
                    f"{tempo_total} s."
                ),
            ]
        ),
    }


def sincronizar_metricas_documentacao(
    relatorio: dict[str, object],
    raiz: Path,
) -> None:
    """Substitui somente os blocos marcados nos documentos do projeto."""

    padrao = re.compile(
        rf"({re.escape(MARCADOR_INICIO)}\n).*?"
        rf"(\n{re.escape(MARCADOR_FIM)})",
        flags=re.DOTALL,
    )
    for nome, bloco in blocos_metricas_desempenho(relatorio).items():
        caminho = raiz / nome
        conteudo = caminho.read_text(encoding="utf-8")
        atualizado, substituicoes = padrao.subn(
            lambda partes: partes.group(1) + bloco + partes.group(2),
            conteudo,
        )
        if substituicoes != 1:
            raise RuntimeError(
                f"bloco de metricas ausente ou duplicado em {nome}"
            )
        if atualizado != conteudo:
            # Evita regravar arquivos já sincronizados e gerar diffs vazios.
            caminho.write_text(atualizado, encoding="utf-8")
