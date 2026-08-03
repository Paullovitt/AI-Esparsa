"""Testes do checkpoint V7.3 BPE-8192 principal."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer

from avaliar_geracao_v7 import avaliar_criterios
from executar_gerador_esparso_v73 import carregar_v73, formatar_prompt


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT_V73 = (
    RAIZ / "modelos" / "gerador_esparso_v73_bpe8192_refino_eos3_ul005_topologia.pt"
)


class TesteCicloV73(unittest.TestCase):
    def test_loader_carrega_modelo_bpe_em_cpu(self) -> None:
        modelo, tokenizador, checkpoint = carregar_v73(
            CHECKPOINT_V73,
            torch.device("cpu"),
            exigir_kernel_cuda=False,
        )
        self.assertEqual(
            checkpoint["modelo"],
            "gerador-esparso-v7-textual-experimental",
        )
        self.assertEqual(checkpoint["epoca"], 3)
        self.assertEqual(checkpoint["status"], "aprovado_avaliacao_v280")
        self.assertTrue(checkpoint["elegivel_promocao"])
        self.assertEqual(checkpoint["avaliacao_qualidade"]["status"], "aprovado")
        self.assertEqual(checkpoint["tokenizador"]["tipo"], "bpe_codigo_bytes_v1")
        self.assertEqual(tokenizador.tamanho, 8192)
        self.assertEqual(checkpoint["topologia"]["formato"], "coo-v1")
        self.assertEqual(len(checkpoint["topologia"]["sha256"]), 64)
        auditoria = modelo.auditoria()
        self.assertFalse(auditoria["caminho_incremental_fundido"])
        self.assertEqual(auditoria["lancamentos_kernel_por_token"], 0)
        self.assertTrue(auditoria["memoria_escores_tempo_quadratica"])
        self.assertEqual(
            sum(parametro.numel() for parametro in modelo.parameters()),
            1_179_383,
        )

    def test_template_conversacional_evitar_eos_imediato(self) -> None:
        modelo, tokenizador, _ = carregar_v73(
            CHECKPOINT_V73,
            torch.device("cpu"),
            exigir_kernel_cuda=False,
        )
        prompt = "Explique inteligência artificial de forma simples."
        cru = torch.tensor(
            [tokenizador.codificar(prompt, bos=True, eos=False)]
        )
        formatado = torch.tensor(
            [
                tokenizador.codificar(
                    formatar_prompt(prompt),
                    bos=True,
                    eos=False,
                )
            ]
        )
        with torch.inference_mode():
            logits_cru, _ = modelo(cru)
            logits_formatados, _ = modelo(formatado)
        self.assertEqual(int(logits_cru[0, -1].argmax()), tokenizador.eos_id)
        self.assertNotEqual(
            int(logits_formatados[0, -1].argmax()),
            tokenizador.eos_id,
        )
        self.assertEqual(
            tokenizador.decodificar(formatado[0].tolist()),
            formatar_prompt(prompt),
        )

    def test_criterios_gerativos_reprovam_repeticao(self) -> None:
        resultado = avaliar_criterios(
            {
                "fracao_tokens_repetidos_media": 0.75,
                "maior_repeticao_consecutiva_media": 5.0,
                "distinct_2_media": 0.20,
                "taxa_eos": 0.0,
            },
            maximo_fracao_repetidos=0.50,
            maximo_repeticao_consecutiva=3.0,
            minimo_distinct_2=0.60,
            minimo_taxa_eos=0.20,
        )
        self.assertFalse(resultado["aprovado"])
        self.assertFalse(all(resultado["verificacoes"].values()))

    def test_bpe_legado_reconstroi_todos_os_bytes_gpt2(self) -> None:
        _, tokenizador, checkpoint = carregar_v73(
            CHECKPOINT_V73,
            torch.device("cpu"),
            exigir_kernel_cuda=False,
        )
        externo = Tokenizer.from_str(
            checkpoint["tokenizador"]["serializado"]
        )
        vocabulario = externo.get_vocab()
        bytes_visiveis = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(0xA1, 0xAC + 1))
            + list(range(0xAE, 0xFF + 1))
        )
        codigos = list(bytes_visiveis)
        proximo = 0
        for valor in range(256):
            if valor not in bytes_visiveis:
                bytes_visiveis.append(valor)
                codigos.append(256 + proximo)
                proximo += 1
        reverso = dict(
            zip(
                (chr(codigo) for codigo in codigos),
                bytes_visiveis,
                strict=True,
            )
        )
        for texto, indice in vocabulario.items():
            if indice < 4:
                esperado = b""
            else:
                try:
                    esperado = bytes(reverso[caractere] for caractere in texto)
                except KeyError:
                    esperado = externo.decode(
                        [indice],
                        skip_special_tokens=True,
                    ).encode("utf-8", errors="replace")
            self.assertEqual(tokenizador.bytes_por_id[indice], esperado)

    def test_loader_rejeita_hash_topologico_adulterado(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        adulterado = dict(checkpoint)
        adulterado["topologia"] = dict(checkpoint["topologia"])
        adulterado["topologia"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "adulterado.pt"
            torch.save(adulterado, caminho)
            with self.assertRaisesRegex(ValueError, "topologia COO"):
                carregar_v73(
                    caminho,
                    torch.device("cpu"),
                    exigir_kernel_cuda=False,
                )

    def test_checkpoint_e_somente_inferencia(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        self.assertNotIn("estado_otimizador", checkpoint)
        self.assertNotIn("estado_agendador", checkpoint)
        self.assertAlmostEqual(
            checkpoint["metricas_validacao"]["ppl_token"],
            55.87224300774265,
            places=8,
        )
        self.assertAlmostEqual(
            checkpoint["metricas_validacao"]["bits_por_byte"],
            1.3480766482646738,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
