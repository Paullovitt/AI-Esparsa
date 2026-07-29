"""Testes da comparação consolidada V6.2, V7.3 e denso.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import unittest

import torch

from comparar_v73 import (
    CHECKPOINT_V73,
    EXEMPLO_MARKDOWN,
    RAIZ,
    RELATORIO_JSON,
    RELATORIO_MARKDOWN,
    _gerar_exemplos,
    _gerar_markdown,
)
from executar_gerador_esparso_v73 import MODELO_CHECKPOINT
from src.relatorio_v73 import blocos_documentacao


class TesteComparacaoV73(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.relatorio = json.loads(
            RELATORIO_JSON.read_text(encoding="utf-8")
        )

    def test_checkpoint_base_e_fp32(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(checkpoint["modelo"], MODELO_CHECKPOINT)
        self.assertEqual(checkpoint["epoca"], 2)
        self.assertFalse(checkpoint["treinado_em_bf16"])
        self.assertFalse(checkpoint["elegivel_promocao"])

    def test_relatorio_contem_somente_os_tres_modelos(self) -> None:
        self.assertEqual(
            set(self.relatorio["modelos"]),
            {"v62", "v73", "denso"},
        )
        self.assertEqual(
            self.relatorio["modelos_mantidos"],
            ["v62", "v73", "denso"],
        )
        conclusao = self.relatorio["conclusao"]
        self.assertEqual(conclusao["base_desenvolvimento"], "v73")
        self.assertFalse(conclusao["artefato_bf16_presente"])
        self.assertTrue(conclusao["qualidade_textual_integral"])
        self.assertTrue(conclusao["v73_greedy_24"])

    def test_as_24_geracoes_possuem_concordancia(self) -> None:
        for dados in self.relatorio["modelos"].values():
            geracao = dados["geracao_livre"]
            self.assertEqual(geracao["amostras_avaliadas"], 24)
            self.assertEqual(len(geracao["exemplos"]), 24)
            self.assertEqual(geracao["taxa_aprovacao"], 1.0)
            self.assertEqual(geracao["concordancia_genero"], 1.0)
        self.assertEqual(
            self.relatorio["modelos"]["v73"]["geracao_livre"][
                "taxa_uso_retentativa"
            ],
            0.0,
        )

    def test_separa_modelo_puro_do_sistema_controlado(self) -> None:
        for dados in self.relatorio["modelos"].values():
            for chave in (
                "geracao_modelo_puro",
                "geracao_greedy_restrita",
                "geracao_livre",
            ):
                self.assertEqual(len(dados[chave]["exemplos"]), 24)
        v73 = self.relatorio["modelos"]["v73"]
        self.assertLess(
            v73["geracao_modelo_puro"]["taxa_aprovacao"],
            v73["geracao_greedy_restrita"]["taxa_aprovacao"],
        )
        intervencoes = v73["geracao_greedy_restrita"][
            "intervencoes_decodificador"
        ]
        self.assertGreater(intervencoes["tokens_bloqueados"], 0)
        self.assertGreater(intervencoes["mudancas_argmax"], 0)

    def test_benchmark_registra_media_mediana_desvio_e_gpu(self) -> None:
        self.assertEqual(
            self.relatorio["protocolo_benchmark"],
            {
                "aquecimentos": 2,
                "repeticoes": 5,
                "agregacao_principal": "mediana",
            },
        )
        for dados in self.relatorio["modelos"].values():
            for chave in (
                "desempenho_forward",
                "desempenho_autorregressivo",
            ):
                protocolo = dados[chave]["protocolo_estatistico"]
                self.assertEqual(len(protocolo["amostras"]), 5)
                resumo = protocolo["estatisticas"]["tokens_por_segundo"]
                self.assertIn("media", resumo)
                self.assertIn("mediana", resumo)
                self.assertIn("desvio_padrao", resumo)
                self.assertIsNotNone(protocolo["gpu_antes"])

    def test_documentos_derivam_do_json(self) -> None:
        self.assertEqual(
            RELATORIO_MARKDOWN.read_text(encoding="utf-8"),
            _gerar_markdown(self.relatorio),
        )
        self.assertEqual(
            EXEMPLO_MARKDOWN.read_text(encoding="utf-8"),
            _gerar_exemplos(self.relatorio),
        )

    def test_documentacao_declara_a_base_atual(self) -> None:
        documentos = "\n".join(
            [
                (RAIZ / "README.md").read_text(encoding="utf-8"),
                (RAIZ / "STATUS.md").read_text(encoding="utf-8"),
            ]
        )
        self.assertIn("base de desenvolvimento", documentos.lower())
        self.assertIn("V7.3", documentos)
        self.assertNotIn("v73_cuda_bf16_2epocas", documentos)
        for nome in ("v62", "v73", "denso"):
            velocidade = self.relatorio["modelos"][nome][
                "desempenho_autorregressivo"
            ]["tokens_por_segundo"]
            formatada = f"{velocidade:.2f}".replace(".", ",")
            self.assertIn(f"{formatada} tok/s", documentos)
        for nome, bloco in blocos_documentacao(self.relatorio).items():
            self.assertIn(
                bloco,
                (RAIZ / nome).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
