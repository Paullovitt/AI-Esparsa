"""Testes da sincronização entre relatório e documentação.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from src.documentacao_metricas import (
    MARCADOR_FIM,
    MARCADOR_INICIO,
    blocos_metricas_desempenho,
    sincronizar_metricas_documentacao,
)


RAIZ = Path(__file__).resolve().parents[1]
RELATORIO = (
    RAIZ
    / "resultados"
    / "gerador_esparso_base_50k"
    / "relatorio.json"
)


def _extrair_bloco(conteudo: str) -> str:
    """Extrai o texto mantido entre os marcadores documentais."""

    correspondencia = re.search(
        rf"{re.escape(MARCADOR_INICIO)}\n(.*?)\n"
        rf"{re.escape(MARCADOR_FIM)}",
        conteudo,
        flags=re.DOTALL,
    )
    if correspondencia is None:
        raise AssertionError("bloco de metricas nao encontrado")
    return correspondencia.group(1)


class TesteDocumentacaoMetricas(unittest.TestCase):
    def test_documentos_correspondem_ao_relatorio_atual(self) -> None:
        relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
        esperados = blocos_metricas_desempenho(relatorio)
        for nome, esperado in esperados.items():
            atual = _extrair_bloco(
                (RAIZ / nome).read_text(encoding="utf-8")
            )
            self.assertEqual(atual, esperado, nome)

    def test_sincronizacao_substitui_somente_o_bloco_marcado(self) -> None:
        relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            for nome in blocos_metricas_desempenho(relatorio):
                (raiz / nome).write_text(
                    "cabecalho\n"
                    f"{MARCADOR_INICIO}\nvalores antigos\n"
                    f"{MARCADOR_FIM}\nrodape\n",
                    encoding="utf-8",
                )
            sincronizar_metricas_documentacao(relatorio, raiz)
            for nome, esperado in blocos_metricas_desempenho(
                relatorio
            ).items():
                conteudo = (raiz / nome).read_text(encoding="utf-8")
                self.assertEqual(_extrair_bloco(conteudo), esperado)
                self.assertTrue(conteudo.startswith("cabecalho\n"))
                self.assertTrue(conteudo.endswith("\nrodape\n"))


if __name__ == "__main__":
    unittest.main()
