"""Impede divergencias entre o relatorio canonico e a documentacao V6."""

from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]


def formatar_decimal(valor: float, casas: int) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


class TesteDocumentacaoV6(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.relatorio = json.loads(
            (RAIZ / "resultados" / "v6_ultimo.json").read_text(
                encoding="utf-8"
            )
        )
        cls.readme = (RAIZ / "README.md").read_text(encoding="utf-8")
        cls.status = (RAIZ / "STATUS.md").read_text(encoding="utf-8")
        cls.documento = (RAIZ / "DOCUMENTO_MODELO_V6.md").read_text(
            encoding="utf-8"
        )

    def test_metricas_de_qualidade_estao_atualizadas(self) -> None:
        agregado = self.relatorio["agregado"]
        melhor_ppl = min(
            execucao["teste"]["ppl"]
            for execucao in self.relatorio["execucoes"]
        )
        esperados = {
            formatar_decimal(agregado["ppl_media"], 4),
            formatar_decimal(
                agregado["acuracia_token_media"] * 100, 2
            )
            + "%",
            formatar_decimal(agregado["ppl_sem_ffn_media"], 4),
            formatar_decimal(melhor_ppl, 4),
        }
        for valor in esperados:
            self.assertIn(valor, self.readme)
            self.assertIn(valor, self.status)
            self.assertIn(valor, self.documento)

    def test_benchmarks_do_readme_sao_medias_atuais(self) -> None:
        for comprimento, casas in ((73, 3), (512, 3)):
            valores = [
                item["metricas"]["tokens_por_segundo"]
                for item in self.relatorio["benchmark"]
                if item["comprimento"] == comprimento
            ]
            self.assertEqual(len(valores), 3)
            media_milhoes = statistics.fmean(valores) / 1_000_000
            self.assertIn(
                formatar_decimal(media_milhoes, casas)
                + " M tokens/s",
                self.readme,
            )


if __name__ == "__main__":
    unittest.main()
