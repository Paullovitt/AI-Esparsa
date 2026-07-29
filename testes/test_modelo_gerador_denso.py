"""Testes da baseline densa experimental.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.modelo_gerador_denso import (
    ConfiguracaoGeradorDenso,
    ModeloGeradorDenso,
)
from src.modelo_gerador_esparso import ModeloGeradorEsparso


class TesteModeloGeradorDenso(unittest.TestCase):
    def test_orcamento_de_parametros_difere_menos_de_um_porcento(
        self,
    ) -> None:
        esparso = ModeloGeradorEsparso(332, 0)
        denso = ModeloGeradorDenso(332, 0)
        parametros_esparsos = sum(
            parametro.numel() for parametro in esparso.parameters()
        )
        parametros_densos = sum(
            parametro.numel() for parametro in denso.parameters()
        )
        self.assertEqual(parametros_esparsos, 163_667)
        self.assertEqual(parametros_densos, 163_003)
        self.assertLess(
            abs(parametros_esparsos / parametros_densos - 1.0),
            0.01,
        )

    def test_qk_atencao_e_ffn_sao_densos(self) -> None:
        modelo = ModeloGeradorDenso(332, 0)
        auditoria = modelo.auditoria()
        self.assertTrue(auditoria["atencao_causal_densa"])
        self.assertTrue(auditoria["qk_denso"])
        self.assertTrue(auditoria["ffn_densa"])
        self.assertEqual(auditoria["lineares_densas_internas"], 12)
        self.assertEqual(
            sum(
                isinstance(modulo, nn.Linear)
                for modulo in modelo.modules()
            ),
            12,
        )

    def test_prefixo_nao_depende_de_tokens_futuros(self) -> None:
        torch.manual_seed(123)
        modelo = ModeloGeradorDenso(
            41,
            0,
            ConfiguracaoGeradorDenso(
                dimensao=16,
                camadas=2,
                especialistas=24,
                maximo_contexto=32,
            ),
        ).eval()
        primeira = torch.randint(1, 41, (2, 12))
        segunda = primeira.clone()
        segunda[:, 7:] = torch.randint(1, 41, (2, 5))
        with torch.inference_mode():
            logits_primeira, _ = modelo(primeira)
            logits_segunda, _ = modelo(segunda)
        torch.testing.assert_close(
            logits_primeira[:, :7],
            logits_segunda[:, :7],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_gradientes_alcancam_todas_as_lineares(self) -> None:
        modelo = ModeloGeradorDenso(
            37,
            0,
            ConfiguracaoGeradorDenso(
                dimensao=16,
                camadas=1,
                especialistas=24,
                maximo_contexto=20,
            ),
        )
        tokens = torch.randint(1, 37, (3, 10))
        logits, _ = modelo(tokens)
        logits.square().mean().backward()
        lineares = [
            modulo
            for modulo in modelo.modules()
            if isinstance(modulo, nn.Linear)
        ]
        self.assertEqual(len(lineares), 4)
        for camada in lineares:
            self.assertIsNotNone(camada.weight.grad)
            self.assertTrue(torch.isfinite(camada.weight.grad).all())
            self.assertGreater(float(camada.weight.grad.abs().sum()), 0.0)

    def test_valida_entrada_e_limite_de_contexto(self) -> None:
        modelo = ModeloGeradorDenso(
            29,
            0,
            ConfiguracaoGeradorDenso(maximo_contexto=12),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            modelo(torch.ones(12, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "vazia"):
            modelo(torch.empty((1, 0), dtype=torch.long))
        logits, _ = modelo(torch.randint(1, 29, (2, 20)))
        self.assertEqual(logits.shape, (2, 12, 29))


if __name__ == "__main__":
    unittest.main()
