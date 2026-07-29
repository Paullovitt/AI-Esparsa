"""Testes do checkpoint textual V7.3 preservado.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"


class TesteCicloV73(unittest.TestCase):
    def test_loader_carrega_base_v73_em_cpu(self) -> None:
        modelo, _, checkpoint = carregar_v73(
            CHECKPOINT_V73,
            torch.device("cpu"),
            exigir_kernel_cuda=False,
        )
        self.assertEqual(checkpoint["modelo"], "gerador-esparso-v73-base")
        self.assertEqual(checkpoint["epoca"], 2)
        self.assertEqual(
            sum(parametro.numel() for parametro in modelo.parameters()),
            165_443,
        )

    def test_checkpoint_nao_contem_metadado_legado(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        texto_chaves = " ".join(str(chave) for chave in checkpoint)
        self.assertNotRegex(texto_chaves.lower(), r"v6[._]?[12]")
        self.assertEqual(
            checkpoint["arquitetura_base"],
            "v7.3-roteamento-combinatorio",
        )


if __name__ == "__main__":
    unittest.main()
