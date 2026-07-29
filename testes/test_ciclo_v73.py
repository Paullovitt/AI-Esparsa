"""Testes do ciclo de carregamento, validação e promoção V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73
from promover_gerador_esparso import validar_promocao_v73


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"


def _relatorio_promovivel() -> dict[str, object]:
    geracao = {
        "exemplos": [{"texto": "x"} for _ in range(24)],
        "taxa_aprovacao": 1.0,
    }
    return {
        "modelos": {
            "v73": {
                "geracao_modelo_puro": dict(geracao),
                "geracao_greedy_restrita": dict(geracao),
                "geracao_livre": dict(geracao),
                "avaliacao_teste": {"ppl": 1.05},
                "desempenho_autorregressivo": {
                    "tokens_por_segundo": 100.0,
                    "protocolo_estatistico": {
                        "repeticoes": 5,
                        "aquecimentos": 2,
                    },
                },
            },
            "denso": {
                "avaliacao_teste": {"ppl": 1.04},
            },
        }
    }


class TesteCicloV73(unittest.TestCase):
    def test_loader_aceita_candidato_fp32_de_cinco_epocas(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint.update(
            {
                "modelo": "gerador-esparso-v73-treino-fp32",
                "epoca": 5,
                "epocas_planejadas": 5,
                "elegivel_promocao": True,
            }
        )
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "epoca_05.pt"
            torch.save(checkpoint, caminho)
            modelo, _, carregado = carregar_v73(
                caminho,
                torch.device("cpu"),
                exigir_kernel_cuda=False,
            )
        self.assertEqual(carregado["epoca"], 5)
        self.assertEqual(
            sum(parametro.numel() for parametro in modelo.parameters()),
            165_443,
        )

    def test_promotor_aceita_candidato_com_relatorio_completo(self) -> None:
        checkpoint = {
            "modelo": "gerador-esparso-v73-treino-fp32",
            "epoca": 5,
            "epocas_planejadas": 5,
            "dados_treino": 50_000,
            "passos_por_epoca": 500,
            "elegivel_promocao": True,
        }
        validar_promocao_v73(checkpoint, _relatorio_promovivel())

    def test_promotor_rejeita_base_historica_de_duas_epocas(self) -> None:
        checkpoint = {
            "modelo": "gerador-esparso-v73-base",
            "epoca": 2,
            "epocas_planejadas": 2,
            "dados_treino": 50_000,
            "passos_por_epoca": 500,
            "elegivel_promocao": False,
        }
        with self.assertRaisesRegex(RuntimeError, "epoca_5"):
            validar_promocao_v73(checkpoint, _relatorio_promovivel())


if __name__ == "__main__":
    unittest.main()
