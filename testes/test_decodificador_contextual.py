from __future__ import annotations

import unittest

import torch

from src.decodificador_contextual import (
    ConfiguracaoDecodificacao,
    aplicar_controle_repeticao,
    bloquear_proximo_ngram_repetido,
    gerar_controlado,
)


class TokenizadorFalso:
    id_para_token = ("<pad>", "<bos>", "<eos>", "resposta", ":", "na", "cozinha", ".")
    token_para_id = {token: indice for indice, token in enumerate(id_para_token)}
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def codificar(self, texto: str, *, bos: bool = True, eos: bool = True) -> list[int]:
        ids = [self.token_para_id["resposta"], self.token_para_id[":"]]
        return ([self.bos_id] if bos else []) + ids + ([self.eos_id] if eos else [])

    def decodificar(self, ids: list[int]) -> str:
        tokens = [self.id_para_token[i] for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id}]
        return " ".join(tokens).replace(" :", ":").replace(" .", ".")


class ModeloFalso(torch.nn.Module):
    class Configuracao:
        maximo_contexto = 32

    configuracao = Configuracao()

    def forward(self, tokens: torch.Tensor):
        gerados = tokens.shape[1] - 3
        sequencia = [5, 6, 7, 5, 6, 7]
        proximo = sequencia[min(gerados, len(sequencia) - 1)]
        logits = torch.full((1, tokens.shape[1], 8), -20.0, device=tokens.device)
        logits[0, -1, proximo] = 20.0
        return logits, {}


class TesteDecodificadorContextual(unittest.TestCase):
    def test_bloqueia_bigrama_repetido(self) -> None:
        logits = torch.zeros(10)
        bloquear_proximo_ngram_repetido(logits, [5, 6, 5], 2)
        self.assertTrue(torch.isneginf(logits[6]))

    def test_controle_pode_ser_desligado_para_repeticao_semantica(self) -> None:
        logits = torch.zeros(10)
        aplicar_controle_repeticao(
            logits,
            [5, 6, 7, 5, 6],
            ConfiguracaoDecodificacao(
                ngrama_bloqueado=0,
                penalidade_repeticao=0.0,
                bloquear_trigrama_adicional=False,
            ),
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_para_na_primeira_frase_completa(self) -> None:
        texto = gerar_controlado(
            ModeloFalso(),
            TokenizadorFalso(),
            "Resposta:",
            torch.device("cpu"),
            configuracao=ConfiguracaoDecodificacao(maximo_tokens=8),
        )
        self.assertEqual(texto, "resposta: na cozinha.")
        self.assertNotIn("cozinha. na", texto)


if __name__ == "__main__":
    unittest.main()
