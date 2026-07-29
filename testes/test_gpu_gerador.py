"""Contrato CUDA do único checkpoint oficial."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from executar_gerador_esparso import carregar_gerador
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62


RAIZ = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponível")
class TesteGPUGerador(unittest.TestCase):
    def test_checkpoint_executa_na_gpu(self) -> None:
        dispositivo = torch.device("cuda")
        modelo, tokenizador, _ = carregar_gerador(
            RAIZ / "modelos" / "gerador_esparso_base.pt",
            dispositivo,
            classe_modelo=ModeloGeradorEsparsoV62,
        )
        entrada = torch.tensor(
            [tokenizador.codificar("pedido: texto:", eos=False)],
            device=dispositivo,
        )
        with torch.inference_mode():
            logits, _ = modelo(entrada)
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(modelo.auditoria()["cache_linear_csr"])


if __name__ == "__main__":
    unittest.main()
