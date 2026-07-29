"""Testes de equivalencia do runtime-base incremental V6.2.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from torch import nn

from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)
from src.modelo_gerador_esparso_v62 import (
    ModeloGeradorEsparsoV62,
)


RAIZ = Path(__file__).resolve().parents[1]
RELATORIO = (
    RAIZ / "resultados" / "v62_base_runtime" / "relatorio.json"
)


def configuracao_teste() -> ConfiguracaoGeradorEsparso:
    """Mantem as proporcoes esparsas com custo adequado a testes de CPU."""

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
        bloco_consultas_atencao=16,
    )


class TesteModeloGeradorEsparsoV62(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(620)
        configuracao = configuracao_teste()
        self.referencia = ModeloGeradorEsparso(
            80,
            0,
            configuracao,
            2026,
        ).eval()
        self.v62 = ModeloGeradorEsparsoV62(
            80,
            0,
            configuracao,
            2026,
        ).eval()
        self.v62.load_state_dict(
            self.referencia.state_dict(),
            strict=True,
        )

    def test_checkpoint_e_arquitetura_permanecem_identicos(self) -> None:
        self.assertEqual(
            tuple(self.referencia.state_dict()),
            tuple(self.v62.state_dict()),
        )
        self.assertEqual(
            sum(p.numel() for p in self.referencia.parameters()),
            sum(p.numel() for p in self.v62.parameters()),
        )
        self.assertFalse(
            any(isinstance(modulo, nn.Linear) for modulo in self.v62.modules())
        )
        auditoria = self.v62.auditoria()
        self.assertTrue(auditoria["qk_coo"])
        self.assertTrue(auditoria["ffn_coo"])
        self.assertTrue(auditoria["cache_causal_incremental"])
        self.assertTrue(auditoria["cache_linear_csr"])
        self.assertTrue(auditoria["cache_gates_escalares"])
        self.assertEqual(auditoria["parametros_adicionais_cache"], 0)
        self.assertTrue(auditoria["arquitetura_pesos_identica_oficial"])

    def test_csr_e_usado_so_no_cache_de_inferencia(self) -> None:
        camada = self.v62.blocos[0].consulta
        entrada = torch.randn(9, camada.entradas)
        with torch.inference_mode():
            saida_csr = camada(entrada)
            self.assertEqual(camada.matriz().layout, torch.sparse_csr)
            camada.configurar_cache_csr_inferencia(False)
            saida_coo = camada(entrada)
            self.assertEqual(camada.matriz().layout, torch.sparse_coo)
            camada.configurar_cache_csr_inferencia(True)
        self.assertTrue(
            torch.allclose(
                saida_csr,
                saida_coo,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        camada.train()
        self.assertEqual(camada.matriz().layout, torch.sparse_coo)

    def test_prefill_equivale_ao_forward_oficial(self) -> None:
        tokens = torch.randint(1, 80, (2, 31))
        with torch.inference_mode():
            esperado, _ = self.referencia(tokens)
            obtido, cache = self.v62.iniciar_cache_geracao(tokens)
        self.assertEqual(cache.comprimento, 31)
        self.assertEqual(len(cache.camadas), 2)
        self.assertTrue(
            torch.allclose(
                obtido,
                esperado[:, -1:],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_avanco_incremental_equivale_a_recalcular_prefixo(self) -> None:
        historico = torch.randint(1, 80, (2, 11))
        with torch.inference_mode():
            logits_cache, cache = self.v62.iniciar_cache_geracao(
                historico
            )
            for _ in range(20):
                esperado, _ = self.referencia(historico)
                self.assertTrue(
                    torch.allclose(
                        logits_cache,
                        esperado[:, -1:],
                        atol=2e-5,
                        rtol=2e-5,
                    )
                )
                novo_token = torch.randint(1, 80, (2, 1))
                historico = torch.cat((historico, novo_token), dim=1)
                logits_cache, cache = self.v62.avancar_cache_geracao(
                    novo_token,
                    cache,
                )

    def test_overflow_refaz_janela_e_preserva_posicoes(self) -> None:
        historico = torch.randint(1, 80, (1, 48))
        novo_token = torch.randint(1, 80, (1, 1))
        with torch.inference_mode():
            _, cache = self.v62.iniciar_cache_geracao(historico)
            obtido, cache = self.v62.avancar_cache_geracao(
                novo_token,
                cache,
            )
            janela = torch.cat((historico[:, 1:], novo_token), dim=1)
            esperado, _ = self.referencia(janela)
        self.assertEqual(cache.comprimento, 48)
        self.assertTrue(torch.equal(cache.tokens[:, :48], janela))
        self.assertTrue(
            torch.allclose(
                obtido,
                esperado[:, -1:],
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_cache_rejeita_modo_treino_e_shapes_invalidos(self) -> None:
        self.v62.train()
        with self.assertRaisesRegex(RuntimeError, "modo eval"):
            self.v62.iniciar_cache_geracao(
                torch.randint(1, 80, (1, 8))
            )
        self.v62.eval()
        with self.assertRaisesRegex(ValueError, r"\[lote, tempo\]"):
            self.v62.iniciar_cache_geracao(
                torch.randint(1, 80, (8,))
            )
        with torch.inference_mode():
            _, cache = self.v62.iniciar_cache_geracao(
                torch.randint(1, 80, (1, 8))
            )
            with self.assertRaisesRegex(ValueError, r"\[lote, 1\]"):
                self.v62.avancar_cache_geracao(
                    torch.randint(1, 80, (1, 2)),
                    cache,
                )

    def test_relatorio_real_comprova_equivalencia_e_aceleracao(self) -> None:
        relatorio = json.loads(RELATORIO.read_text(encoding="utf-8"))
        self.assertTrue(
            relatorio["checkpoint"]["oficial_preservado"]
        )
        self.assertFalse(relatorio["treino"]["executado"])
        self.assertTrue(
            relatorio["equivalencia_numerica"]["aprovado"]
        )
        self.assertEqual(
            relatorio["equivalencia_geracoes_oficiais"][
                "textos_identicos"
            ],
            24,
        )
        self.assertEqual(
            relatorio["geracao_livre"]["taxa_aprovacao"],
            1.0,
        )
        self.assertGreater(
            relatorio["benchmark_autorregressivo"]["aceleracao"],
            1.0,
        )
        self.assertTrue(
            relatorio["benchmark_autorregressivo"]["texto_identico"]
        )


if __name__ == "__main__":
    unittest.main()
