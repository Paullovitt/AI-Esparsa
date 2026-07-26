"""Valida o contrato persistido do checkpoint canonico V6."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from executar_v6 import carregar_v6


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT = RAIZ / "modelos" / "v6_rollback.pt"


class TesteCheckpointV6(unittest.TestCase):
    def test_checkpoint_canonico_possui_metadados_v6(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=True
        )
        self.assertEqual(checkpoint["modelo"], "V6")
        self.assertEqual(checkpoint["nome"], "V6 Base")
        self.assertTrue(checkpoint["checkpoint_canonico"])
        self.assertEqual(checkpoint["semente"], 20260727)
        self.assertEqual(checkpoint["epoca"], 5)

    def test_checkpoint_recarrega_estritamente(self) -> None:
        modelo, _, _ = carregar_v6(CHECKPOINT, torch.device("cpu"))
        auditoria = modelo.auditoria()
        self.assertEqual(auditoria["modelo"], "V6")
        self.assertEqual(auditoria["parametros_qk_densos"], 0)
        self.assertFalse(
            any(
                isinstance(modulo, nn.Linear)
                for modulo in modelo.ffn.modules()
            )
        )


if __name__ == "__main__":
    unittest.main()
