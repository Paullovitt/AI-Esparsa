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
        cls.relatorio_isolado = json.loads(
            (
                RAIZ
                / "resultados"
                / "teste_isolado_v6_ultimo.json"
            ).read_text(encoding="utf-8")
        )
        cls.relatorio_v61 = json.loads(
            (
                RAIZ
                / "resultados"
                / "v61_candidata_validacao.json"
            ).read_text(encoding="utf-8")
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

    def test_auditoria_isolada_esta_documentada_sem_inflar_metricas(
        self,
    ) -> None:
        isolado = self.relatorio_isolado
        aprendido = isolado["aprendido_sem_leitor_qk"]["qualidade"]
        adaptado = isolado["aprendido_com_leitor_qk"]["qualidade"]
        geracao = isolado["manual"]["geracao_livre"]
        benchmark = isolado["benchmark"]["manual"]
        esperados = {
            formatar_decimal(aprendido["ppl"], 4),
            formatar_decimal(
                aprendido["acuracia_local_resposta"] * 100, 2
            )
            + "%",
            formatar_decimal(
                aprendido["recuperacao_fato"] * 100, 2
            )
            + "%",
            formatar_decimal(adaptado["ppl"], 4),
            formatar_decimal(geracao["sequencias_exatas"] * 100, 2)
            + "%",
            formatar_decimal(geracao["taxa_eos"] * 100, 2) + "%",
            formatar_decimal(
                geracao["acuracia_tokens_alinhados"] * 100, 2
            )
            + "%",
            formatar_decimal(
                geracao["acuracia_locais_alinhados"] * 100, 2
            )
            + "%",
            formatar_decimal(
                benchmark["73"]["pipeline_completo"][
                    "tokens_por_segundo"
                ]
                / 1_000_000,
                3,
            )
            + " M tokens/s",
            formatar_decimal(
                benchmark["512"]["pipeline_completo"][
                    "tokens_por_segundo"
                ]
                / 1_000_000,
                3,
            )
            + " M tokens/s",
        }
        for documento in (self.readme, self.status, self.documento):
            for valor in esperados:
                self.assertIn(valor, documento)

        self.assertEqual(
            isolado["decisao"],
            "nao_promover_roteador_aprendido_manter_v6_base",
        )
        self.assertFalse(
            isolado["criterios"][
                "ppl_sem_leitor_ate_5_porcento_da_base"
            ]
        )
        self.assertFalse(
            isolado["criterios"][
                "geracao_livre_base_exata_minimo_90"
            ]
        )

        v61 = self.relatorio_v61
        agregado = v61["agregado"]
        esperados_v61 = {
            formatar_decimal(agregado["ppl"]["media"], 4),
            formatar_decimal(
                agregado["acuracia_token"]["media"] * 100,
                2,
            )
            + "%",
            formatar_decimal(
                agregado["geracao_exata"]["media"] * 100,
                2,
            )
            + "%",
            formatar_decimal(
                agregado["tokens_livres"]["media"] * 100,
                2,
            )
            + "%",
            formatar_decimal(
                agregado["locais_livres"]["media"] * 100,
                2,
            )
            + "%",
        }
        for documento in (self.readme, self.status, self.documento):
            for valor in esperados_v61:
                self.assertIn(valor, documento)

        for comprimento in ("73", "512"):
            benchmark = v61["benchmark_agregado"][comprimento]
            esperados_benchmark = {
                formatar_decimal(
                    benchmark[
                        "v6_base_tokens_por_segundo"
                    ]["media"]
                    / 1_000_000,
                    3,
                )
                + " M tokens/s",
                formatar_decimal(
                    benchmark["v61_tokens_por_segundo"]["media"]
                    / 1_000_000,
                    3,
                )
                + " M tokens/s",
                formatar_decimal(
                    benchmark["razao_velocidade_media"] * 100,
                    2,
                )
                + "%",
            }
            for documento in (self.readme, self.documento):
                for valor in esperados_benchmark:
                    self.assertIn(valor, documento)
        self.assertEqual(
            v61["decisao"],
            "aprovada_como_candidata_v61",
        )
        self.assertTrue(all(v61["criterios"].values()))


if __name__ == "__main__":
    unittest.main()
