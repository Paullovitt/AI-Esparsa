"""Testes do experimento isolado de roteamento e geracao V6."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.atencao_causal_texto import (
    ConfiguracaoTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
)
from testar_v6_isolado import (
    RoteadorEsparsoAprendido,
    perda_leitor_qk,
    perda_contrastiva,
    posicoes_consulta_e_origem,
    preparar_com_roteador,
)


class TesteV6Isolado(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulario = criar_vocabulario()
        self.configuracao = ConfiguracaoTextoCausal()
        self.dados = gerar_corpus_texto_causal(
            8,
            909,
            self.vocabulario,
            self.configuracao,
            usar_combinacoes_ineditas=True,
        )
        self.roteador = RoteadorEsparsoAprendido(
            self.vocabulario.tamanho,
            semente=909,
        )

    def test_roteador_nao_possui_rede_densa(self) -> None:
        self.assertFalse(
            any(
                isinstance(modulo, nn.Linear)
                for modulo in self.roteador.modules()
            )
        )
        self.assertFalse(
            hasattr(self.roteador, "mapa_entidade")
            or hasattr(self.roteador, "mapa_objeto")
        )

    def test_descritores_sao_topk_validos(self) -> None:
        descritores = self.roteador.descritores(
            self.dados.entradas
        )
        self.assertEqual(descritores.shape[-1], 12)
        self.assertTrue(torch.all(descritores > 0))
        self.assertTrue(torch.all(descritores <= 128))

    def test_perda_contrastiva_tem_gradiente_esparso(self) -> None:
        perda, acuracia = perda_contrastiva(
            self.roteador, self.dados
        )
        perda.backward()
        self.assertTrue(torch.isfinite(perda))
        self.assertGreaterEqual(acuracia, 0.0)
        self.assertLessEqual(acuracia, 1.0)
        self.assertTrue(self.roteador.codigo_token.weight.grad.is_sparse)

    def test_posicoes_supervisionadas_sao_causais(self) -> None:
        consultas, origens = posicoes_consulta_e_origem(self.dados)
        self.assertEqual(consultas.shape, (8, 3))
        self.assertTrue(torch.all(origens < consultas))

    def test_perda_qk_alcanca_somente_pesos_esparsos(self) -> None:
        from src.modelo_v6 import ModeloV6

        modelo = ModeloV6(
            self.vocabulario,
            self.configuracao,
            semente_topologia=909,
        )
        dados, descritores = preparar_com_roteador(
            self.dados,
            self.roteador,
            self.configuracao.top_k,
        )
        perda, _ = perda_leitor_qk(modelo, dados, descritores)
        perda.backward()
        self.assertTrue(modelo.atencao.q_feature.weight.grad.is_sparse)
        self.assertTrue(modelo.atencao.k_feature.weight.grad.is_sparse)


if __name__ == "__main__":
    unittest.main()
