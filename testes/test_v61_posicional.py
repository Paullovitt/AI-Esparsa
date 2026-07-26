"""Testes estruturais da variante V6.1 posicional isolada."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import nn

from src.atencao_causal_texto import (
    ConfiguracaoTextoCausal,
    criar_vocabulario,
    gerar_corpus_texto_causal,
)
from src.modelo_v61 import (
    DIMENSAO_FEATURES_BASE,
    CodigoTemporalEsparso,
    ModeloV61Posicional,
)
from testar_v61_posicional import mascara_conteudo_temporal


class TesteV61Posicional(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulario = criar_vocabulario()
        self.configuracao = replace(
            ConfiguracaoTextoCausal(),
            dimensao_features=DIMENSAO_FEATURES_BASE + 5,
        )
        self.dados = gerar_corpus_texto_causal(
            4,
            808,
            self.vocabulario,
            self.configuracao,
            usar_combinacoes_ineditas=True,
        )
        self.codigo = CodigoTemporalEsparso(
            self.vocabulario,
            self.configuracao,
            semente=808,
        )

    def test_codigo_temporal_e_esparso_e_sem_parametros(self) -> None:
        auditoria = self.codigo.auditoria()
        self.assertEqual(auditoria["parametros_treinaveis"], 0)
        self.assertFalse(auditoria["usa_nn_linear"])
        self.assertEqual(auditoria["largura_descritor"], 6)

    def test_slots_ligam_cada_pergunta_ao_fato_na_mesma_ordem(
        self,
    ) -> None:
        descritores = self.codigo.descritores(self.dados.entradas)
        slots = descritores[0, :, -2]
        self.assertEqual(int(slots[0]), 0)
        for slot in range(3):
            posicao_fato = 1 + slot * 7
            posicao_pergunta = 22 + slot * 17
            esperado = DIMENSAO_FEATURES_BASE + 1 + slot
            self.assertEqual(int(slots[posicao_fato]), esperado)
            self.assertEqual(int(slots[posicao_pergunta]), esperado)

    def test_papeis_ligam_consultas_a_objeto_e_entidade(self) -> None:
        descritores = self.codigo.descritores(self.dados.entradas)
        papeis = descritores[0, :, -1]
        papel_objeto = DIMENSAO_FEATURES_BASE + 4
        papel_entidade = DIMENSAO_FEATURES_BASE + 5
        self.assertEqual(int(papeis[4]), papel_objeto)
        self.assertEqual(int(papeis[24]), papel_objeto)
        self.assertEqual(int(papeis[25]), papel_objeto)
        self.assertEqual(int(papeis[2]), papel_entidade)
        self.assertEqual(int(papeis[27]), papel_entidade)

    def test_descritor_de_prefixo_nao_depende_do_futuro(self) -> None:
        completo = self.codigo.descritores(self.dados.entradas)
        prefixo = self.dados.entradas[:, :40]
        parcial = self.codigo.descritores(prefixo)
        self.assertTrue(torch.equal(completo[:, :40], parcial))

    def test_modelo_preserva_qk_e_ffn_esparsos(self) -> None:
        modelo = ModeloV61Posicional(
            self.vocabulario,
            self.configuracao,
            semente_topologia=808,
        )
        auditoria = modelo.auditoria()
        self.assertEqual(auditoria["parametros_qk_densos"], 0)
        self.assertFalse(auditoria["ffn_esparsa"]["usa_nn_linear"])
        self.assertEqual(auditoria["parametros_posicionais"], 1)
        self.assertFalse(
            any(
                isinstance(modulo, nn.Linear)
                for modulo in modelo.ffn.modules()
            )
        )

    def test_mascara_temporal_reforca_conteudo_sem_ver_futuro(
        self,
    ) -> None:
        mascara = mascara_conteudo_temporal(
            self.dados,
            self.configuracao,
        )
        self.assertEqual(mascara.shape, (1, 73))
        self.assertEqual(int(mascara.sum()), 27)
        self.assertTrue(bool(mascara[0, 21]))
        self.assertFalse(bool(mascara[0, 20]))


if __name__ == "__main__":
    unittest.main()
