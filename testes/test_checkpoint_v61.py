"""Valida o contrato do checkpoint-base V6.1."""

from __future__ import annotations

import unittest

import torch

from executar_v61 import CHECKPOINT_V61, carregar_v61
from src.atencao_causal_texto import (
    criar_vocabulario,
    gerar_corpus_texto_causal,
    selecionar_candidatos_causais,
)


class TesteCheckpointV61(unittest.TestCase):
    def test_checkpoint_possui_metadados_de_cinco_epocas(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V61,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            checkpoint["modelo"],
            "V6.1-base",
        )
        self.assertEqual(checkpoint["status"], "base_oficial")
        self.assertEqual(checkpoint["epoca"], 5)
        self.assertEqual(checkpoint["semente"], 20260727)

    def test_checkpoint_recarrega_estritamente_e_executa(self) -> None:
        dispositivo = torch.device("cpu")
        modelo, codigo, _ = carregar_v61(
            CHECKPOINT_V61,
            dispositivo,
        )
        vocabulario = criar_vocabulario()
        dados = gerar_corpus_texto_causal(
            2,
            707,
            vocabulario,
            modelo.configuracao_texto,
            usar_combinacoes_ineditas=True,
        )
        descritores = codigo.descritores(dados.entradas)
        candidatos, validos = selecionar_candidatos_causais(
            descritores,
            modelo.configuracao_texto.top_k,
        )
        with torch.inference_mode():
            logits, posicoes = modelo(
                dados.entradas,
                descritores,
                candidatos,
                validos,
            )
        self.assertEqual(
            logits.shape,
            (*dados.entradas.shape, vocabulario.tamanho),
        )
        self.assertEqual(posicoes.shape, dados.entradas.shape)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(
            modelo.auditoria()["parametros_qk_densos"],
            0,
        )
        self.assertFalse(
            modelo.auditoria()["ffn_esparsa"]["usa_nn_linear"]
        )


if __name__ == "__main__":
    unittest.main()
