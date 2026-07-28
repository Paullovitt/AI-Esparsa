"""Testes estruturais e causais do gerador esparso experimental."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)


def configuracao_pequena() -> ConfiguracaoGeradorEsparso:
    """Reduz o custo dos testes sem trocar os contratos da arquitetura."""

    return ConfiguracaoGeradorEsparso(
        dimensao=32,
        camadas=2,
        top_k_atencao=8,
        fan_in_qk=8,
        especialistas=64,
        fan_in_ffn=8,
        fan_out_ffn=12,
        top_k_ffn=16,
        maximo_contexto=48,
    )


class TesteModeloGeradorEsparso(unittest.TestCase):
    def test_prefixo_nao_depende_de_tokens_futuros(self) -> None:
        modelo = ModeloGeradorEsparso(
            80,
            0,
            configuracao_pequena(),
            123,
        ).eval()
        prefixo = torch.randint(1, 80, (2, 20))
        completo = torch.cat(
            (prefixo, torch.randint(1, 80, (2, 7))),
            dim=1,
        )
        with torch.inference_mode():
            logits_prefixo, _ = modelo(prefixo)
            logits_completo, _ = modelo(completo)
        self.assertTrue(
            torch.allclose(
                logits_prefixo,
                logits_completo[:, : prefixo.shape[1]],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_qk_ffn_residuais_e_normalizacao_sao_preservados(self) -> None:
        modelo = ModeloGeradorEsparso(
            80,
            0,
            configuracao_pequena(),
            456,
        )
        auditoria = modelo.auditoria()
        self.assertTrue(auditoria["qk_coo"])
        self.assertTrue(auditoria["ffn_coo"])
        self.assertTrue(auditoria["residuais"])
        self.assertTrue(auditoria["normalizacao"])
        self.assertEqual(auditoria["lineares_densas_internas"], 0)
        self.assertFalse(
            any(isinstance(modulo, nn.Linear) for modulo in modelo.modules())
        )
        self.assertAlmostEqual(float(auditoria["densidade_qk"]), 0.25)

    def test_gradientes_alcancam_qk_e_as_duas_projecoes_ffn(self) -> None:
        modelo = ModeloGeradorEsparso(
            80,
            0,
            configuracao_pequena(),
            789,
        )
        tokens = torch.randint(1, 80, (2, 24))
        logits, _ = modelo(tokens[:, :-1])
        perda = F.cross_entropy(
            logits.reshape(-1, 80),
            tokens[:, 1:].reshape(-1),
        )
        perda.backward()
        parametros = (
            modelo.blocos[0].consulta.valores,
            modelo.blocos[0].chave.valores,
            modelo.blocos[0].ffn.camada_1.valores,
            modelo.blocos[0].ffn.camada_2.valores,
        )
        for parametro in parametros:
            self.assertIsNotNone(parametro.grad)
            self.assertTrue(torch.isfinite(parametro.grad).all())
            self.assertGreater(float(parametro.grad.abs().sum()), 0.0)

    def test_valida_entrada_vazia_e_limita_contexto(self) -> None:
        modelo = ModeloGeradorEsparso(
            80,
            0,
            configuracao_pequena(),
            321,
        ).eval()
        with self.assertRaisesRegex(ValueError, "nao pode ser vazia"):
            modelo(torch.empty(1, 0, dtype=torch.long))
        with torch.inference_mode():
            logits, _ = modelo(torch.randint(1, 80, (1, 60)))
        self.assertEqual(logits.shape, (1, 48, 80))


if __name__ == "__main__":
    unittest.main()
