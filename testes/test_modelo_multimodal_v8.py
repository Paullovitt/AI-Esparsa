"""Testes estruturais e causais da arquitetura multimodal V8.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_multimodal_v8 import (
    ConfiguracaoMultimodalV8,
    ModeloMultimodalEsparsoV8,
    ModeloMultimodalEsparsoTreinoV8,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73


class TesteModeloMultimodalV8(unittest.TestCase):
    def setUp(self) -> None:
        self.base = ConfiguracaoGeradorEsparso(
            dimensao=16,
            camadas=2,
            top_k_atencao=8,
            fan_in_qk=4,
            especialistas=16,
            fan_in_ffn=4,
            fan_out_ffn=4,
            top_k_ffn=4,
            maximo_contexto=32,
            bloco_consultas_atencao=4,
        )
        self.roteamento = ConfiguracaoRoteamentoV73(
            grupos=4,
            grupos_ativos=2,
            dimensao_latente=2,
        )
        self.multimodal = ConfiguracaoMultimodalV8(
            canais_imagem=3,
            tamanho_patch=4,
            altura_imagem=8,
            largura_imagem=8,
            fan_in_projecao_visual=8,
            semente_visual=321,
        )
        self.modelo = ModeloMultimodalEsparsoV8(
            37,
            0,
            self.base,
            123,
            self.roteamento,
            self.multimodal,
        )

    def test_forward_preserva_shape_textual(self) -> None:
        tokens = torch.randint(1, 37, (2, 5))
        imagens = torch.randn(2, 3, 8, 8)
        logits, auxiliares = self.modelo(tokens, imagens)
        self.assertEqual(tuple(logits.shape), (2, 5, 37))
        self.assertIn("perda_roteamento", auxiliares)
        self.assertTrue(torch.isfinite(logits).all())

    def test_imagens_diferentes_alteram_logits(self) -> None:
        self.modelo.eval()
        tokens = torch.randint(1, 37, (1, 4))
        zeros = torch.zeros(1, 3, 8, 8)
        uns = torch.ones(1, 3, 8, 8)
        with torch.inference_mode():
            logits_zeros, _ = self.modelo(tokens, zeros)
            logits_uns, _ = self.modelo(tokens, uns)
        self.assertGreater(
            float((logits_zeros - logits_uns).abs().max()),
            1e-6,
        )

    def test_causalidade_textual_com_prefixo_visual(self) -> None:
        self.modelo.eval()
        imagem = torch.randn(1, 3, 8, 8)
        original = torch.tensor([[1, 2, 3, 4, 5]])
        alterado = original.clone()
        alterado[0, -1] = 6
        with torch.inference_mode():
            logits_original, _ = self.modelo(original, imagem)
            logits_alterado, _ = self.modelo(alterado, imagem)
        self.assertTrue(
            torch.allclose(
                logits_original[:, :-1],
                logits_alterado[:, :-1],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_gradiente_alcanca_pesos_visuais_e_roteador(self) -> None:
        tokens = torch.randint(1, 37, (2, 4))
        imagens = torch.randn(2, 3, 8, 8)
        logits, auxiliares = self.modelo(tokens, imagens)
        perda = logits.square().mean() + auxiliares["perda_roteamento"]
        perda.backward()
        gradiente_visual = (
            self.modelo.encoder_visual.projecao_patches.valores.grad
        )
        gradiente_roteador = (
            self.modelo.blocos[0].ffn.roteador.projecao_contexto.grad
        )
        self.assertIsNotNone(gradiente_visual)
        self.assertIsNotNone(gradiente_roteador)
        self.assertGreater(float(gradiente_visual.abs().sum()), 0.0)
        self.assertGreater(float(gradiente_roteador.abs().sum()), 0.0)

    def test_caminho_visual_nao_contem_linear_densa(self) -> None:
        self.assertFalse(
            any(
                isinstance(modulo, nn.Linear)
                for modulo in self.modelo.encoder_visual.modules()
            )
        )
        auditoria = self.modelo.auditoria()
        self.assertEqual(auditoria["lineares_densas_visuais"], 0)
        self.assertTrue(auditoria["projecao_visual_coo"])
        self.assertEqual(
            auditoria["modalidades"],
            ["imagem", "texto"],
        )
        self.assertFalse(auditoria["checkpoint_treinado"])

    def test_geracao_rejeita_modo_treino_e_funciona_em_eval(self) -> None:
        tokens = torch.tensor([[1, 2]])
        imagem = torch.randn(1, 3, 8, 8)
        with self.assertRaisesRegex(RuntimeError, "modo eval"):
            self.modelo.gerar_greedy(tokens, imagem, 2)
        self.modelo.eval()
        gerados = self.modelo.gerar_greedy(tokens, imagem, 2)
        self.assertEqual(tuple(gerados.shape), (1, 4))

    def test_shape_visual_invalido_e_rejeitado(self) -> None:
        tokens = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "dimensoes"):
            self.modelo(tokens, torch.randn(1, 3, 12, 8))
        with self.assertRaisesRegex(ValueError, "lotes"):
            self.modelo(tokens, torch.randn(2, 3, 8, 8))

    def test_contexto_excedente_e_cache_textual_sao_rejeitados(self) -> None:
        imagem = torch.randn(1, 3, 8, 8)
        tokens = torch.ones(1, 29, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "excedem"):
            self.modelo(tokens, imagem)
        with self.assertRaisesRegex(RuntimeError, "cache multimodal"):
            self.modelo.iniciar_cache_geracao(tokens)

    def test_backend_vetorizado_preserva_parametros_e_gradientes(self) -> None:
        treino = ModeloMultimodalEsparsoTreinoV8(
            37,
            0,
            self.base,
            123,
            self.roteamento,
            self.multimodal,
        )
        treino.load_state_dict(self.modelo.state_dict(), strict=True)
        self.assertEqual(
            set(treino.state_dict()),
            set(self.modelo.state_dict()),
        )
        tokens = torch.randint(1, 37, (2, 4))
        imagens = torch.randn(2, 3, 8, 8)
        logits, auxiliares = treino(tokens, imagens)
        (logits.square().mean() + auxiliares["perda_roteamento"]).backward()
        auditoria = treino.auditoria()
        self.assertTrue(auditoria["backend_treino_vetorizado"])
        self.assertTrue(auditoria["treino_somente_microgrupos_ativos"])
        self.assertEqual(
            auditoria["parametros"],
            self.modelo.auditoria()["parametros"],
        )
        self.assertIsNotNone(
            treino.encoder_visual.projecao_patches.valores.grad
        )


if __name__ == "__main__":
    unittest.main()
