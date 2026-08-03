"""Testes do runtime V7.3 e de seu cache causal.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import unittest

import torch

from executar_gerador_esparso_v73 import gerar_greedy_v73
from src.geracao_textual_v73 import (
    ConfiguracaoDecodificacaoV73,
    gerar_v73,
    metricas_diversidade_tokens,
)
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v73 import (
    ConfiguracaoRuntimeV73,
    ModeloGeradorEsparsoV73,
)
from src.roteamento_combinatorio_v73 import (
    ConfiguracaoRoteamentoV73,
    ModeloGeradorEsparsoRoteadoV73,
)
from src.runtime_condicional_v73 import (
    ConfiguracaoRuntimeCondicionalV73,
)


class TesteModeloGeradorEsparsoV73(unittest.TestCase):
    def setUp(self) -> None:
        self.configuracao = ConfiguracaoGeradorEsparso(
            dimensao=32,
            camadas=2,
            top_k_atencao=4,
            fan_in_qk=8,
            especialistas=32,
            fan_in_ffn=8,
            fan_out_ffn=8,
            top_k_ffn=8,
            maximo_contexto=32,
            bloco_consultas_atencao=4,
        )
        self.roteamento = ConfiguracaoRoteamentoV73(
            grupos=4,
            grupos_ativos=2,
            dimensao_latente=3,
        )
        self.referencia = ModeloGeradorEsparsoRoteadoV73(
            41,
            0,
            self.configuracao,
            123,
            self.roteamento,
        ).eval()
        self.v73 = ModeloGeradorEsparsoV73(
            41,
            0,
            self.configuracao,
            123,
            self.roteamento,
            ConfiguracaoRuntimeCondicionalV73(),
            ConfiguracaoRuntimeV73(),
        ).eval()
        self.v73.load_state_dict(
            self.referencia.state_dict(),
            strict=True,
        )

    def test_fallback_cpu_preserva_logits_e_state_dict(self) -> None:
        tokens = torch.randint(1, 41, (2, 11))
        with torch.inference_mode():
            esperado, _ = self.referencia(tokens)
            obtido, _ = self.v73(tokens)
        self.assertTrue(
            torch.allclose(obtido, esperado, atol=2e-5, rtol=1e-5)
        )
        self.assertEqual(
            set(self.v73.state_dict()),
            set(self.referencia.state_dict()),
        )

    def test_cache_incremental_preserva_recalculo_completo(self) -> None:
        prefixo = torch.randint(1, 41, (1, 6))
        novo = torch.randint(1, 41, (1, 1))
        with torch.inference_mode():
            logits_cache, cache = self.v73.iniciar_cache_geracao(prefixo)
            logits_completos, _ = self.v73(prefixo)
            self.assertTrue(
                torch.allclose(
                    logits_cache,
                    logits_completos[:, -1:],
                    atol=2e-5,
                    rtol=1e-5,
                )
            )
            logits_cache, _ = self.v73.avancar_cache_geracao(novo, cache)
            logits_completos, _ = self.v73(torch.cat((prefixo, novo), dim=1))
        self.assertTrue(
            torch.allclose(
                logits_cache,
                logits_completos[:, -1:],
                atol=2e-5,
                rtol=1e-5,
            )
        )

    def test_geracao_greedy_retorna_metricas(self) -> None:
        prefixo = torch.randint(1, 41, (1, 3))
        gerados, medidas = gerar_greedy_v73(
            self.v73,
            prefixo,
            maximo_novos_tokens=2,
            eos_id=0,
        )
        self.assertEqual(tuple(gerados.shape), (1, 5))
        self.assertEqual(medidas["tokens_gerados"], 2.0)
        self.assertGreater(medidas["tokens_por_segundo"], 0.0)

    def test_decodificacao_limita_repeticao_sem_alterar_greedy_puro(self) -> None:
        class ModeloFalso:
            def __init__(self) -> None:
                self.logits = torch.tensor([[[0.0, -2.0, -2.0, 5.0, 4.0, 3.0]]])

            def iniciar_cache_geracao(self, tokens):
                return self.logits.to(tokens.device), object()

            def avancar_cache_geracao(self, token, cache):
                return self.logits.to(token.device), cache

        modelo = ModeloFalso()
        prefixo = torch.tensor([[1, 2]])
        puro, _ = gerar_greedy_v73(modelo, prefixo, 4, eos_id=0)
        controlado, medidas = gerar_v73(
            modelo,
            prefixo,
            4,
            eos_id=0,
            configuracao=ConfiguracaoDecodificacaoV73(
                modo="greedy",
                temperatura=1.0,
                top_k=0,
                top_p=1.0,
                penalidade_repeticao=1.0,
                ngrama_sem_repeticao=0,
                maximo_repeticao_consecutiva=2,
            ),
        )
        self.assertEqual(puro[0, -4:].tolist(), [3, 3, 3, 3])
        self.assertEqual(controlado[0, -4:].tolist(), [3, 3, 4, 3])
        self.assertEqual(medidas["maior_repeticao_consecutiva"], 2.0)

    def test_metricas_diversidade_detectam_colapso(self) -> None:
        metricas = metricas_diversidade_tokens([4, 4, 4, 4, 2], eos_id=2)
        self.assertTrue(metricas["eos_emitido"])
        self.assertEqual(metricas["distinct_1"], 0.25)
        self.assertEqual(metricas["maior_repeticao_consecutiva"], 4.0)
        self.assertGreater(metricas["fracao_tokens_repetidos"], 0.7)

    def test_auditoria_nao_inventa_parametros(self) -> None:
        auditoria = self.v73.auditoria()
        self.assertTrue(auditoria["kernel_ffn_fundido"])
        self.assertFalse(auditoria["caminho_incremental_fundido"])
        self.assertEqual(auditoria["lancamentos_kernel_por_token"], 0)
        self.assertTrue(auditoria["memoria_escores_tempo_quadratica"])
        self.assertTrue(auditoria["forward_lote_vetorizado"])
        self.assertEqual(auditoria["parametros_adicionais_kernel"], 0)
        self.assertFalse(auditoria["grupos_inativos_executados"])
        self.assertFalse(auditoria["elegivel_promocao"])

    def test_limiar_invalido_e_rejeitado(self) -> None:
        with self.assertRaisesRegex(ValueError, "deve ser positivo"):
            ModeloGeradorEsparsoV73(
                41,
                0,
                self.configuracao,
                123,
                self.roteamento,
                ConfiguracaoRuntimeCondicionalV73(),
                ConfiguracaoRuntimeV73(limiar_tokens_lote=0),
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponivel")
class TesteKernelCudaV73(unittest.TestCase):
    def test_kernel_preserva_argmax_do_forward(self) -> None:
        configuracao = ConfiguracaoGeradorEsparso()
        roteamento = ConfiguracaoRoteamentoV73()
        referencia = ModeloGeradorEsparsoRoteadoV73(
            41,
            0,
            configuracao,
            123,
            roteamento,
        ).cuda().eval()
        fundido = ModeloGeradorEsparsoV73(
            41,
            0,
            configuracao,
            123,
            roteamento,
            ConfiguracaoRuntimeCondicionalV73(),
            ConfiguracaoRuntimeV73(
                exigir_kernel_cuda=True,
                limiar_tokens_lote=1,
            ),
        ).cuda().eval()
        fundido.load_state_dict(referencia.state_dict(), strict=True)
        tokens = torch.randint(1, 41, (1, 37), device="cuda")
        with torch.inference_mode():
            esperado, _ = referencia(tokens)
            obtido, _ = fundido(tokens)
        self.assertEqual(
            int((esperado.argmax(-1) != obtido.argmax(-1)).sum()),
            0,
        )
        self.assertLess(float((esperado - obtido).abs().mean()), 2e-5)


if __name__ == "__main__":
    unittest.main()
