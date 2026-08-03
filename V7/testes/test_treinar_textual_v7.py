"""Testes do ciclo textual experimental da V7.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73
from src.tokenizador_palavras import (
    TokenizadorBPEBytes,
    TokenizadorBPECodigo,
    TokenizadorBytes,
    tokenizador_de_checkpoint,
)
from src.topologia_checkpoint_v73 import metadados_topologia
from treinar_textual_v7 import (
    SEMENTE_TOPOLOGIA_V73,
    ConfiguracaoExperimento,
    CorpusConversacionalV7,
    comprimento_balde,
    criar_modelos,
    estado_checkpoint,
    indices_epoca,
    iterar_lotes,
    perda_repeticao_incorreta,
    perda_resposta,
    selecionar_projecao_logits,
    transferir_base,
)


class TesteTreinoTextualV7(unittest.TestCase):
    def test_corpus_mascara_prompt_e_respeita_contexto(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "dados.jsonl"
            caminho.write_text(
                json.dumps(
                    {
                        "prompt": "Explique " + "algo " * 100,
                        "resposta": "Resposta objetiva.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            corpus = CorpusConversacionalV7(
                caminho, TokenizadorBytes(), maximo_contexto=96
            )
            entrada, labels = corpus.exemplos[0]
        self.assertEqual(len(entrada), len(labels))
        self.assertLessEqual(len(entrada) + 1, 96)
        self.assertIn(-100, labels)
        self.assertTrue(any(valor >= 0 for valor in labels))

    def test_indices_sao_deterministicos_e_tem_passos_exatos(self) -> None:
        configuracao = ConfiguracaoExperimento(
            passos_por_epoca=3, tamanho_lote=4
        )
        primeira = indices_epoca(20, configuracao, 1)
        segunda = indices_epoca(20, configuracao, 1)
        self.assertEqual(primeira, segunda)
        self.assertEqual(len(primeira), 3)
        self.assertTrue(all(len(lote) == 4 for lote in primeira))

    def test_epoca_integral_cobre_todos_os_segmentos_uma_vez(self) -> None:
        configuracao = ConfiguracaoExperimento(
            passos_por_epoca=None,
            tamanho_lote=4,
        )
        lotes = indices_epoca(10, configuracao, 1)
        indices = [indice for lote in lotes for indice in lote]
        self.assertEqual(len(lotes), 3)
        self.assertEqual(sorted(indices), list(range(10)))

    def test_resposta_longa_e_segmentada_sem_eos_artificial(self) -> None:
        resposta = "texto coerente e completo. " * 40
        tokenizador = TokenizadorBytes()
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "dados.jsonl"
            caminho.write_text(
                json.dumps(
                    {"prompt": "Continue o texto.", "resposta": resposta},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            corpus = CorpusConversacionalV7(
                caminho,
                tokenizador,
                maximo_contexto=96,
            )
        supervisionados = [
            token
            for _, labels in corpus.exemplos
            for token in labels
            if token >= 0
        ]
        esperado = tokenizador.codificar(
            resposta.strip(), bos=False, eos=True
        )
        self.assertEqual(supervisionados, esperado)
        self.assertEqual(supervisionados.count(tokenizador.eos_id), 1)
        self.assertGreater(len(corpus.exemplos), 1)
        self.assertEqual(corpus.estatisticas["tokens_descartados"], 0)
        self.assertEqual(corpus.estatisticas["eos_artificiais"], 0)

    def test_bpe_e_reconstruido_integralmente_do_checkpoint(self) -> None:
        tokenizador = TokenizadorBPEBytes(
            ["usuario: explique a rotina assistente: resposta completa"],
            tamanho_vocabulario=280,
        )
        checkpoint = {
            "tokenizador": {
                "tipo": tokenizador.TIPO,
                "mesclas": tokenizador.mesclas,
            },
            "vocabulario": tokenizador.id_para_token,
        }
        carregado = tokenizador_de_checkpoint(checkpoint)
        texto = "resposta em português sem token desconhecido"
        self.assertEqual(
            carregado.decodificar(
                carregado.codificar(texto, bos=True, eos=True)
            ),
            texto.capitalize(),
        )

    def test_checkpoint_bpe_codigo_preserva_serializacao(self) -> None:
        tokenizador = TokenizadorBPECodigo(
            ["usuario: Preserve Caixa, espa?os e s?mbolos!"],
            tamanho_vocabulario=280,
        )
        modelo, arquitetura, roteamento, treino = criar_modelos(tokenizador)
        otimizador = torch.optim.AdamW(modelo.parameters())
        agendador = torch.optim.lr_scheduler.LambdaLR(
            otimizador,
            lambda _: 1.0,
        )
        checkpoint = estado_checkpoint(
            modelo,
            otimizador,
            agendador,
            1,
            ConfiguracaoExperimento(tipo_tokenizador="bpe_codigo"),
            arquitetura,
            roteamento,
            treino,
            {"treino.jsonl": "a", "validacao.jsonl": "b"},
            tokenizador,
            False,
        )
        self.assertIn("serializado", checkpoint["tokenizador"])
        carregado = tokenizador_de_checkpoint(checkpoint)
        texto = "Caixa, espa?os e s?mbolos!"
        self.assertEqual(
            carregado.decodificar(carregado.codificar(texto)),
            texto,
        )

    def test_executor_carrega_checkpoint_bpe_com_weights_only(self) -> None:
        """Garante o ciclo salvar/carregar do novo formato experimental."""

        tokenizador = TokenizadorBPEBytes(
            ["usuario: pergunta assistente: resposta completa"],
            tamanho_vocabulario=280,
        )
        modelo, arquitetura, roteamento, _ = criar_modelos(tokenizador)
        checkpoint = {
            "modelo": "gerador-esparso-v7-textual-experimental",
            "semente": SEMENTE_TOPOLOGIA_V73,
            "configuracao": asdict(arquitetura),
            "configuracao_roteamento": asdict(roteamento),
            "tokenizador": {
                "tipo": tokenizador.TIPO,
                "mesclas": tokenizador.mesclas,
            },
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "topologia": metadados_topologia(modelo),
        }
        tokens = torch.tensor(
            [[tokenizador.bos_id, 4, 5, 6]],
            dtype=torch.long,
        )
        modelo.eval()
        with torch.inference_mode():
            logits_antes, _ = modelo(tokens)
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "checkpoint.pt"
            torch.save(checkpoint, caminho)
            carregado, tokenizador_carregado, metadados = carregar_v73(
                caminho,
                torch.device("cpu"),
                exigir_kernel_cuda=False,
            )
            with torch.inference_mode():
                logits_depois, _ = carregado(tokens)
        self.assertEqual(
            tokenizador_carregado.id_para_token,
            tokenizador.id_para_token,
        )
        self.assertEqual(
            metadados["modelo"],
            "gerador-esparso-v7-textual-experimental",
        )
        self.assertEqual(
            carregado.embedding.weight.shape[0],
            tokenizador.tamanho,
        )
        self.assertTrue(
            torch.allclose(logits_antes, logits_depois, atol=1e-5)
        )

    def test_checkpoint_separa_seed_topologica_da_seed_experimental(
        self,
    ) -> None:
        """Impede reconectar pesos COO ao restaurar um experimento."""

        tokenizador = TokenizadorBPEBytes(
            ["usuario: pergunta assistente: resposta"],
            tamanho_vocabulario=280,
        )
        modelo, arquitetura, roteamento, treino = criar_modelos(tokenizador)
        otimizador = torch.optim.AdamW(modelo.parameters())
        agendador = torch.optim.lr_scheduler.LambdaLR(
            otimizador,
            lambda _: 1.0,
        )
        configuracao = ConfiguracaoExperimento(semente=123456)
        checkpoint = estado_checkpoint(
            modelo,
            otimizador,
            agendador,
            1,
            configuracao,
            arquitetura,
            roteamento,
            treino,
            {"treino.jsonl": "a", "validacao.jsonl": "b"},
            tokenizador,
            False,
        )
        self.assertEqual(checkpoint["semente"], SEMENTE_TOPOLOGIA_V73)
        self.assertEqual(checkpoint["semente_experimento"], 123456)
        self.assertEqual(
            checkpoint["configuracao_experimento"]["semente"],
            123456,
        )

    def test_flags_arquiteturais_configuram_o_modelo(self) -> None:
        tokenizador = TokenizadorBytes()
        configuracao = ConfiguracaoExperimento(
            maximo_contexto=128,
            dimensao_modelo=64,
            camadas_modelo=2,
            top_k_atencao_modelo=8,
            fan_in_qk_modelo=8,
            especialistas_modelo=128,
            fan_in_ffn_modelo=16,
            fan_out_ffn_modelo=16,
            top_k_ffn_modelo=8,
        )
        modelo, arquitetura, _, _ = criar_modelos(
            tokenizador,
            configuracao_experimento=configuracao,
        )
        self.assertEqual(arquitetura.dimensao, 64)
        self.assertEqual(arquitetura.camadas, 2)
        self.assertEqual(arquitetura.top_k_atencao, 8)
        self.assertEqual(arquitetura.fan_in_qk, 8)
        self.assertEqual(arquitetura.especialistas, 128)
        self.assertEqual(arquitetura.maximo_contexto, 128)
        self.assertEqual(len(modelo.blocos), 2)
        self.assertEqual(modelo.embedding.embedding_dim, 64)

    def test_perda_ignora_prompt(self) -> None:
        logits = torch.zeros(1, 3, 5, requires_grad=True)
        labels = torch.tensor([[-100, 2, 3]])
        perda, quantidade = perda_resposta(logits, labels)
        perda.backward()
        self.assertEqual(quantidade, 2)
        self.assertTrue(torch.isfinite(perda))
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.0)

    def test_peso_eos_prioriza_fim_da_resposta(self) -> None:
        logits = torch.tensor(
            [[[5.0, -2.0, -2.0], [5.0, -2.0, -2.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 2]])
        comum, _ = perda_resposta(logits, labels)
        ponderada, _ = perda_resposta(
            logits,
            labels,
            eos_id=2,
            peso_eos=3.0,
        )
        self.assertGreater(float(ponderada.detach()), float(comum.detach()))
        ponderada.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unlikelihood_filtrado_equivale_ao_completo(self) -> None:
        logits = torch.full((1, 5, 7), -3.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 4] = 5.0
            logits[0, 1, 4] = 5.0
            logits[0, 2, 4] = 5.0
            logits[0, 3, 5] = 5.0
            logits[0, 4, 5] = 5.0
        labels = torch.tensor([[-100, 3, 2, 6, 1]])
        mascara = labels.ne(-100)
        perda_completa, quantidade_completa = perda_repeticao_incorreta(
            logits,
            labels,
            janela=3,
        )
        perda_filtrada, quantidade_filtrada = perda_repeticao_incorreta(
            logits[mascara],
            labels,
            janela=3,
        )
        self.assertGreater(quantidade_completa, 0)
        self.assertEqual(quantidade_filtrada, quantidade_completa)
        self.assertTrue(torch.allclose(perda_filtrada, perda_completa, atol=1e-6))
        self.assertGreater(float(perda_completa.detach()), 0.0)

    def test_logits_filtrados_equivalem_ao_forward_completo(self) -> None:
        tokenizador = TokenizadorBytes()
        modelo_completo, _, _, _ = criar_modelos(tokenizador)
        modelo_filtrado, _, _, _ = criar_modelos(tokenizador)
        modelo_filtrado.load_state_dict(modelo_completo.state_dict())
        tokens = torch.tensor(
            [
                [tokenizador.bos_id, 10, 11, 12, 13, 14],
                [tokenizador.bos_id, 20, 21, 22, 23, 24],
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [-100, -100, 11, 12, 13, 14],
                [-100, 20, 21, -100, 23, 24],
            ],
            dtype=torch.long,
        )
        mascara = labels.ne(-100)

        logits_completos, auxiliares_completos = modelo_completo(tokens)
        perda_completa, _ = perda_resposta(logits_completos, labels)
        perda_completa = (
            perda_completa + auxiliares_completos["perda_roteamento"]
        )
        perda_completa.backward()

        logits_filtrados, auxiliares_filtrados = (
            modelo_filtrado.forward_alvos(tokens, mascara)
        )
        perda_filtrada, _ = perda_resposta(logits_filtrados, labels)
        perda_filtrada = (
            perda_filtrada + auxiliares_filtrados["perda_roteamento"]
        )
        perda_filtrada.backward()

        self.assertTrue(
            torch.allclose(
                logits_filtrados,
                logits_completos[mascara],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(perda_filtrada, perda_completa, atol=1e-6)
        )
        for parametro_completo, parametro_filtrado in zip(
            modelo_completo.parameters(),
            modelo_filtrado.parameters(),
            strict=True,
        ):
            self.assertIsNotNone(parametro_completo.grad)
            self.assertIsNotNone(parametro_filtrado.grad)
            self.assertTrue(
                torch.allclose(
                    parametro_completo.grad,
                    parametro_filtrado.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponivel")
    def test_backend_coo_agregado_aceita_bf16(self) -> None:
        tokenizador = TokenizadorBytes()
        modelo, _, _, _ = criar_modelos(tokenizador, "coo_agregado")
        modelo = modelo.to(torch.device("cuda")).train()
        tokens = torch.randint(
            4,
            tokenizador.tamanho,
            (2, 32),
            device="cuda",
        )
        mascara = torch.zeros((2, 32), dtype=torch.bool, device="cuda")
        mascara[:, 16:] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, auxiliares = modelo.forward_alvos(tokens, mascara)
            perda = logits.float().square().mean()
            perda = perda + auxiliares["perda_roteamento"]
        perda.backward()
        self.assertEqual(logits.shape, (32, tokenizador.tamanho))
        self.assertTrue(torch.isfinite(perda))
        self.assertTrue(
            all(parametro.grad is not None for parametro in modelo.parameters())
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA indisponivel")
    def test_seletor_auto_de_projecao_executa_na_gpu(self) -> None:
        tokenizador = TokenizadorBytes()
        modelo, _, _, _ = criar_modelos(tokenizador)
        modelo = modelo.to(torch.device("cuda"))
        tokens = torch.randint(
            4,
            tokenizador.tamanho,
            (2, 32),
            device="cuda",
        )
        labels = torch.randint(
            4,
            tokenizador.tamanho,
            (2, 32),
            device="cuda",
        )
        labels[:, :16] = -100
        selecionada, auditoria = selecionar_projecao_logits(
            modelo,
            tokens,
            labels,
            "auto",
        )
        self.assertIn(selecionada, {"completa", "somente_alvos"})
        self.assertEqual(auditoria["selecionada"], selecionada)
        self.assertGreater(
            auditoria["benchmark"]["somente_alvos_segundos"],
            0.0,
        )

    def test_prefetch_e_baldes_preservam_o_lote(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "dados.jsonl"
            caminho.write_text(
                "".join(
                    json.dumps(
                        {
                            "prompt": f"Pergunta {indice}",
                            "resposta": "Resposta curta.",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for indice in range(4)
                ),
                encoding="utf-8",
            )
            corpus = CorpusConversacionalV7(
                caminho,
                TokenizadorBytes(),
                maximo_contexto=96,
            )
            indices = [0, 1]
            self.assertEqual(comprimento_balde(corpus, indices, True), 96)
            lotes = list(
                iterar_lotes(
                    corpus,
                    [indices, [2, 3]],
                    torch.device("cpu"),
                    usar_prefetch=True,
                    usar_baldes=True,
                )
            )
        self.assertEqual(len(lotes), 2)
        self.assertEqual(lotes[0][0].shape, (2, 96))
        self.assertEqual(lotes[0][1].shape, (2, 96))
        self.assertTrue(lotes[0][1].ne(-100).any())

    def test_transferencia_reinicializa_vocabulario_incompativel(self) -> None:
        raiz = Path(__file__).resolve().parents[1]
        modelo, _, _, _ = criar_modelos(TokenizadorBytes())
        resultado = transferir_base(
            modelo, raiz / "modelos" / "gerador_esparso_v73_bpe8192_5x3000.pt"
        )
        self.assertEqual(resultado["tensores_transferidos"], 54)
        self.assertEqual(
            resultado["tensores_reinicializados"],
            ["embedding.weight", "saida_bias"],
        )
        self.assertEqual(
            sum(parametro.numel() for parametro in modelo.parameters()),
            156_155,
        )


if __name__ == "__main__":
    unittest.main()
