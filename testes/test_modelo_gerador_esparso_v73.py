"""Testes do runtime CUDA fundido V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

import executar_gerador_esparso_v73 as executor_v73
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

    def test_auditoria_nao_inventa_parametros(self) -> None:
        auditoria = self.v73.auditoria()
        self.assertTrue(auditoria["kernel_ffn_fundido"])
        self.assertTrue(auditoria["caminho_incremental_fundido"])
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

    def test_executor_separa_validacao_de_benchmark(self) -> None:
        """Evita repassar opções de benchmark ao contrato do validador."""

        medidas_validacao = {
            "cobertura_palavras_chave": 1.0,
            "retentativas": 0,
        }
        medidas_desempenho = {
            "tokens_por_segundo": 900.0,
            "latencia_primeiro_token_ms": 6.0,
        }
        with (
            patch.object(
                executor_v73,
                "carregar_v73",
                return_value=(object(), object(), {}),
            ) as carregar,
            patch.object(
                executor_v73,
                "validar_prompt_publico",
                return_value=["campos"],
            ),
            patch.object(
                executor_v73,
                "gerar_relato_validado",
                return_value=("texto aprovado", medidas_validacao),
            ) as gerar,
            patch.object(
                executor_v73,
                "benchmark_autorregressivo",
                return_value=medidas_desempenho,
            ) as benchmark,
            patch.object(
                executor_v73.torch.cuda,
                "is_available",
                return_value=False,
            ),
            patch.object(
                sys,
                "argv",
                ["executar_gerador_esparso_v73.py", "--permitir-fallback"],
            ),
            redirect_stdout(io.StringIO()) as saida,
        ):
            executor_v73.main()

        self.assertFalse(carregar.call_args.kwargs["exigir_kernel_cuda"])
        self.assertEqual(gerar.call_args.kwargs, {})
        benchmark.assert_called_once()
        self.assertIn("cobertura=100%", saida.getvalue())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponível")
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
