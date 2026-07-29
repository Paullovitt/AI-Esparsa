"""Contrato CUDA do checkpoint V7.3 preservado.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73


RAIZ = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponivel")
class TesteGPUGerador(unittest.TestCase):
    def test_checkpoint_v73_executa_na_gpu_sem_compilar_kernel(self) -> None:
        dispositivo = torch.device("cuda")
        modelo, tokenizador, _ = carregar_v73(
            RAIZ / "modelos" / "gerador_esparso_v73_base.pt",
            dispositivo,
            exigir_kernel_cuda=False,
        )
        entrada = torch.tensor(
            [[tokenizador.bos_id, tokenizador.eos_id]],
            device=dispositivo,
        )
        # Gradiente habilitado força o caminho PyTorch e mantém este teste
        # independente do cache compilado opcional.
        logits, _ = modelo(entrada)
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
