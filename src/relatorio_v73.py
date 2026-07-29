"""Gera relatórios e sincroniza a documentação da comparação V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import re
from pathlib import Path


MARCADOR_INICIO = "<!-- comparacao-v73:inicio -->"
MARCADOR_FIM = "<!-- comparacao-v73:fim -->"
NOMES = ("v62", "v73", "denso")
TITULOS = {"v62": "V6.2", "v73": "V7.3", "denso": "Denso"}


def _decimal_pt(valor: float, casas: int = 2) -> str:
    parte_inteira, parte_decimal = f"{valor:.{casas}f}".split(".")
    return (
        f"{int(parte_inteira):,}".replace(",", ".")
        + ","
        + parte_decimal
    )


def _inteiro_pt(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _aprovadas(dados: dict[str, object], chave: str) -> str:
    geracao = dados[chave]
    if not isinstance(geracao, dict):
        raise TypeError(f"{chave} invalida")
    total = int(geracao["amostras_avaliadas"])
    quantidade = round(float(geracao["taxa_aprovacao"]) * total)
    return f"{quantidade}/{total}"


def _estatistica(
    dados: dict[str, object],
    metrica: str,
) -> dict[str, float]:
    protocolo = dados.get("protocolo_estatistico", {})
    if not isinstance(protocolo, dict):
        return {}
    estatisticas = protocolo.get("estatisticas", {})
    if not isinstance(estatisticas, dict):
        return {}
    resumo = estatisticas.get(metrica, {})
    return resumo if isinstance(resumo, dict) else {}


def _velocidade_com_dispersao(dados: dict[str, object]) -> str:
    resumo = _estatistica(dados, "tokens_por_segundo")
    mediana = float(dados["tokens_por_segundo"])
    if not resumo:
        return f"{_decimal_pt(mediana)} tok/s"
    return (
        f"{_decimal_pt(mediana)} tok/s "
        f"(média {_decimal_pt(float(resumo['media']))} ± "
        f"{_decimal_pt(float(resumo['desvio_padrao']))})"
    )


def _tabela(relatorio: dict[str, object]) -> list[str]:
    modelos = relatorio["modelos"]
    if not isinstance(modelos, dict):
        raise TypeError("modelos ausentes")
    linhas = [
        "| Métrica | V6.2 | V7.3 | Denso |",
        "|---|---:|---:|---:|",
    ]
    metricas = [
        (
            "Época",
            lambda item: str(int(item["epoca"])),
        ),
        (
            "Parâmetros",
            lambda item: _inteiro_pt(int(item["parametros"])),
        ),
        (
            "PPL teste",
            lambda item: _decimal_pt(
                float(item["avaliacao_teste"]["ppl"]),
                6,
            ),
        ),
        (
            "Aprovação — modelo puro",
            lambda item: _aprovadas(item, "geracao_modelo_puro"),
        ),
        (
            "Modelo puro — objeto/local",
            lambda item: (
                f"{float(item['geracao_modelo_puro']['consistencia_objeto_local']):.0%}"
            ),
        ),
        (
            "Modelo puro — ações",
            lambda item: (
                f"{float(item['geracao_modelo_puro']['consistencia_acoes']):.0%}"
            ),
        ),
        (
            "Modelo puro — concordância",
            lambda item: (
                f"{float(item['geracao_modelo_puro']['concordancia_genero']):.0%}"
            ),
        ),
        (
            "Aprovação — greedy restrito",
            lambda item: _aprovadas(item, "geracao_greedy_restrita"),
        ),
        (
            "Aprovação — sistema completo",
            lambda item: _aprovadas(item, "geracao_livre"),
        ),
        (
            "Mudanças de argmax pelas regras",
            lambda item: str(
                int(
                    item["geracao_greedy_restrita"][
                        "intervencoes_decodificador"
                    ]["mudancas_argmax"]
                )
            ),
        ),
        (
            "Tokens bloqueados pelas regras",
            lambda item: _inteiro_pt(
                int(
                    item["geracao_greedy_restrita"][
                        "intervencoes_decodificador"
                    ]["tokens_bloqueados"]
                )
            ),
        ),
        (
            "Forward — mediana, média ± DP",
            lambda item: _velocidade_com_dispersao(
                item["desempenho_forward"]
            ),
        ),
        (
            "Geração — mediana, média ± DP",
            lambda item: _velocidade_com_dispersao(
                item["desempenho_autorregressivo"]
            ),
        ),
        (
            "Primeiro token — mediana",
            lambda item: (
                f"{_decimal_pt(float(item['desempenho_autorregressivo']['latencia_primeiro_token_ms']))} ms"
            ),
        ),
        (
            "VRAM forward — mediana",
            lambda item: (
                f"{_decimal_pt(float(item['desempenho_forward']['vram_pico_mib']))} MiB"
            ),
        ),
        (
            "Checkpoint",
            lambda item: f"{_inteiro_pt(int(item['checkpoint_bytes']))} bytes",
        ),
    ]
    for rotulo, formatar in metricas:
        linhas.append(
            f"| {rotulo} | "
            + " | ".join(
                formatar(modelos[nome]) for nome in NOMES
            )
            + " |"
        )
    return linhas


def gerar_markdown(relatorio: dict[str, object]) -> str:
    protocolo = relatorio["protocolo_benchmark"]
    conclusao = relatorio["conclusao"]
    linhas = [
        "# Comparação auditável — V6.2, V7.3 e denso",
        "",
        "Autor: Paulo Augusto",
        "Ano: 2026",
        "",
        (
            f"Benchmark com {protocolo['aquecimentos']} aquecimentos e "
            f"{protocolo['repeticoes']} repetições. A mediana é o valor "
            "principal; média e desvio-padrão permanecem no JSON."
        ),
        "",
        *_tabela(relatorio),
        "",
        "## Leitura correta da qualidade",
        "",
        (
            "A linha **modelo puro** usa argmax sem as regras de repetição, "
            "gênero, ações ou objeto de apoio. A linha **greedy restrito** "
            "mede uma única geração causal com as regras. A linha "
            "**sistema completo** inclui o validador e eventuais retentativas."
        ),
        "",
        "Assim, 24/24 do sistema não é apresentado como capacidade exclusiva "
        "dos pesos aprendidos.",
        "",
        "## Conclusão",
        "",
        (
            "A V7.3 foi "
            f"**{float(conclusao['vantagem_geracao_v73_sobre_denso']):.2f}×** "
            "mais rápida que o denso na mediana da geração."
        ),
        (
            "O denso foi "
            f"**{float(conclusao['vantagem_forward_denso_sobre_v73']):.2f}×** "
            "mais rápido na mediana do forward."
        ),
        "",
        "## Reprodução",
        "",
        "```powershell",
        "python comparar_v73.py --repeticoes 5 --aquecimentos 2",
        "python comparar_v73.py --somente-documentos",
        "```",
        "",
    ]
    return "\n".join(linhas)


def gerar_exemplos(relatorio: dict[str, object]) -> str:
    modelos = relatorio["modelos"]
    if not isinstance(modelos, dict):
        raise TypeError("modelos ausentes")
    linhas = [
        "# Textos comparativos auditáveis",
        "",
        "Cada modelo aparece sem restrições, com restrições e como sistema.",
        "",
    ]
    modos = (
        ("geracao_modelo_puro", "Modelo puro"),
        ("geracao_greedy_restrita", "Greedy restrito"),
        ("geracao_livre", "Sistema completo"),
    )
    for nome in NOMES:
        linhas.extend([f"## {TITULOS[nome]}", ""])
        for chave, titulo in modos:
            geracao = modelos[nome][chave]
            exemplo = geracao["exemplos"][0]
            linhas.extend(
                [
                    f"### {titulo}",
                    "",
                    f"**Prompt:** {exemplo['pedido']}",
                    "",
                    str(exemplo["texto"]),
                    "",
                ]
            )
    return "\n".join(linhas)


def blocos_documentacao(
    relatorio: dict[str, object],
) -> dict[str, str]:
    modelos = relatorio["modelos"]
    protocolo = relatorio["protocolo_benchmark"]
    conclusao = relatorio["conclusao"]
    tabela = "\n".join(_tabela(relatorio))
    resumo = "\n".join(
        [
            tabela,
            "",
            (
                f"Protocolo: {protocolo['aquecimentos']} aquecimentos, "
                f"{protocolo['repeticoes']} repetições; mediana como medida "
                "principal e média ± desvio-padrão para dispersão."
            ),
            "",
            (
                "Aprovação do modelo puro, do greedy restrito e do sistema "
                "completo são métricas separadas. As intervenções do "
                "decodificador também são contabilizadas."
            ),
            "",
            (
                "Na mediana, a V7.3 atingiu "
                f"{float(conclusao['vantagem_geracao_v73_sobre_denso']):.2f}× "
                "a geração densa; o denso atingiu "
                f"{float(conclusao['vantagem_forward_denso_sobre_v73']):.2f}× "
                "o forward da V7.3."
            ),
        ]
    )
    status = []
    for nome in NOMES:
        dados = modelos[nome]
        velocidade = dados["desempenho_autorregressivo"]
        status.append(
            f"- {TITULOS[nome]}: mediana de "
            f"{_decimal_pt(float(velocidade['tokens_por_segundo']))} tok/s; "
            f"modelo puro {_aprovadas(dados, 'geracao_modelo_puro')}, "
            f"greedy restrito {_aprovadas(dados, 'geracao_greedy_restrita')} "
            f"e sistema {_aprovadas(dados, 'geracao_livre')}."
        )
    status.extend(
        [
            (
                f"- benchmark: {protocolo['aquecimentos']} aquecimentos e "
                f"{protocolo['repeticoes']} repetições, com mediana, média e "
                "desvio-padrão registrados no JSON."
            ),
            "- as regras do decodificador não são contabilizadas como "
            "capacidade aprendida pelo modelo.",
        ]
    )
    tecnico = "\n".join(
        [
            "O relatório separa três níveis de qualidade:",
            "",
            "- argmax do modelo puro, sem restrições de domínio;",
            "- argmax com restrições causais;",
            "- sistema completo, incluindo retentativas.",
            "",
            "Também registra candidatos bloqueados, mudanças do argmax, "
            "amostras brutas, mediana, média, desvio-padrão, clocks e energia "
            "da GPU quando `nvidia-smi` disponibiliza esses dados.",
            "",
            tabela,
        ]
    )
    return {
        "README.md": resumo,
        "STATUS.md": "\n".join(status),
        "DOCUMENTO_GERADOR_ESPARSO.md": tecnico,
    }


def sincronizar_documentacao(
    relatorio: dict[str, object],
    raiz: Path,
) -> None:
    """Atualiza somente blocos marcados, tornando o JSON a fonte única."""

    padrao = re.compile(
        rf"({re.escape(MARCADOR_INICIO)}\n).*?"
        rf"(\n{re.escape(MARCADOR_FIM)})",
        flags=re.DOTALL,
    )
    for nome, bloco in blocos_documentacao(relatorio).items():
        caminho = raiz / nome
        conteudo = caminho.read_text(encoding="utf-8")
        atualizado, quantidade = padrao.subn(
            lambda grupos: grupos.group(1) + bloco + grupos.group(2),
            conteudo,
        )
        if quantidade != 1:
            raise RuntimeError(
                f"bloco de comparação ausente ou duplicado em {nome}"
            )
        if atualizado != conteudo:
            caminho.write_text(atualizado, encoding="utf-8")
