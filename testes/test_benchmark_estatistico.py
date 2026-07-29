"""Testes do protocolo repetido de desempenho.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.benchmark_estatistico import executar_benchmark_estatistico


class TesteBenchmarkEstatistico(unittest.TestCase):
    def test_calcula_mediana_media_desvio_e_preserva_amostras(self) -> None:
        valores = iter((999.0, 10.0, 30.0, 20.0))

        def medir() -> dict[str, float]:
            return {
                "tokens_por_segundo": next(valores),
                "vram_pico_mib": 12.0,
            }

        with patch(
            "src.benchmark_estatistico.estado_gpu",
            return_value={"pstate": "P0"},
        ):
            resultado = executar_benchmark_estatistico(
                medir,
                repeticoes=3,
                aquecimentos=1,
            )
        self.assertEqual(resultado["tokens_por_segundo"], 20.0)
        protocolo = resultado["protocolo_estatistico"]
        self.assertEqual(protocolo["repeticoes"], 3)
        self.assertEqual(len(protocolo["amostras"]), 3)
        estatisticas = protocolo["estatisticas"]["tokens_por_segundo"]
        self.assertEqual(estatisticas["media"], 20.0)
        self.assertEqual(estatisticas["mediana"], 20.0)
        self.assertEqual(estatisticas["desvio_padrao"], 10.0)

    def test_rejeita_protocolo_sem_repeticao_suficiente(self) -> None:
        with self.assertRaisesRegex(ValueError, "ao menos 2"):
            executar_benchmark_estatistico(
                lambda: {"tokens_por_segundo": 1.0},
                repeticoes=1,
                aquecimentos=1,
            )


if __name__ == "__main__":
    unittest.main()
