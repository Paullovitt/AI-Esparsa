"""Testes estruturais e funcionais do modelo-base V6."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.atencao_causal_texto import (
    CodigoEsparsoCausal,
    ConfiguracaoTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    preparar_candidatos,
)
from src.modelo_v6 import (
    FFNTextoBlocoEsparsa,
    ModeloV6,
)


class TesteModeloV6(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulario = criar_vocabulario()
        self.configuracao = ConfiguracaoTextoCausal()
        self.codigo = CodigoEsparsoCausal(
            self.vocabulario, self.configuracao
        )
        dados = gerar_corpus_texto_causal(
            4,
            20260731,
            self.vocabulario,
            self.configuracao,
            usar_combinacoes_ineditas=True,
        )
        self.dados, self.descritores = preparar_candidatos(
            dados, self.codigo
        )

    def test_ffn_textual_nao_possui_linear_densa(self) -> None:
        ffn = FFNTextoBlocoEsparsa(semente=123)
        self.assertFalse(
            any(isinstance(modulo, nn.Linear) for modulo in ffn.modules())
        )
        auditoria = ffn.auditoria()
        self.assertEqual(auditoria["top_k"], 12)
        self.assertLessEqual(auditoria["densidade"], 0.25)

    def test_forward_integrado_preserva_forma_e_causalidade(self) -> None:
        modelo = ModeloV6(
            self.vocabulario,
            self.configuracao,
            semente_topologia=456,
        )
        logits, posicoes = modelo(
            self.dados.entradas,
            self.descritores,
            self.dados.candidatos,
            self.dados.candidatos_validos,
        )
        self.assertEqual(
            logits.shape,
            (*self.dados.entradas.shape, self.vocabulario.tamanho),
        )
        self.assertEqual(posicoes.shape, self.dados.entradas.shape)
        tempo = torch.arange(self.dados.entradas.shape[1]).view(1, -1)
        self.assertFalse(torch.any(posicoes[:, 1:] >= tempo[:, 1:]))

    def test_ablar_ffn_altera_logits_sem_alterar_posicao_atendida(self) -> None:
        modelo = ModeloV6(
            self.vocabulario,
            self.configuracao,
            semente_topologia=789,
        )
        logits_ffn, posicoes_ffn = modelo(
            self.dados.entradas,
            self.descritores,
            self.dados.candidatos,
            self.dados.candidatos_validos,
        )
        logits_sem, posicoes_sem = modelo(
            self.dados.entradas,
            self.descritores,
            self.dados.candidatos,
            self.dados.candidatos_validos,
            usar_ffn=False,
        )
        self.assertTrue(torch.equal(posicoes_ffn, posicoes_sem))
        self.assertFalse(torch.equal(logits_ffn, logits_sem))

    def test_inicializacao_v6_e_deterministica(self) -> None:
        estados = []
        for _ in range(2):
            torch.manual_seed(2026)
            modelo = ModeloV6(
                self.vocabulario,
                self.configuracao,
                semente_topologia=2026,
            )
            estados.append(modelo.atencao.state_dict())
        for chave in estados[0]:
            self.assertTrue(torch.equal(estados[0][chave], estados[1][chave]))

    def test_gradientes_chegam_as_duas_camadas_esparsas(self) -> None:
        modelo = ModeloV6(
            self.vocabulario,
            self.configuracao,
        )
        logits, _ = modelo(
            self.dados.entradas,
            self.descritores,
            self.dados.candidatos,
            self.dados.candidatos_validos,
        )
        logits.square().mean().backward()
        self.assertIsNotNone(modelo.ffn.camada_1.valores.grad)
        self.assertIsNotNone(modelo.ffn.camada_2.valores.grad)
        self.assertTrue(modelo.atencao.q_feature.weight.grad.is_sparse)


if __name__ == "__main__":
    unittest.main()
