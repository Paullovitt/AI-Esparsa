"""Tokenizador determinístico de palavras e pontuação da base esparsa.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenizadorPalavras:
    """Mapeia palavras/pontuação para IDs com vocabulário serializável."""

    def __init__(self, textos: Iterable[str]) -> None:
        especiais = ["<pad>", "<bos>", "<eos>", "<unk>"]
        tokens = sorted(
            {
                token
                for texto in textos
                for token in self.tokenizar(texto)
            }
        )
        self.id_para_token = tuple(
            especiais
            + [token for token in tokens if token not in especiais]
        )
        self.token_para_id = {
            token: indice
            for indice, token in enumerate(self.id_para_token)
        }

    @classmethod
    def de_vocabulario(
        cls,
        id_para_token: Iterable[str],
    ) -> "TokenizadorPalavras":
        """Reconstrói estritamente o vocabulário salvo no checkpoint."""

        tokens = tuple(str(token) for token in id_para_token)
        especiais = ("<pad>", "<bos>", "<eos>", "<unk>")
        if tokens[: len(especiais)] != especiais:
            raise ValueError("vocabulario nao preserva os tokens especiais")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulario contem tokens duplicados")
        tokenizador = cls.__new__(cls)
        tokenizador.id_para_token = tokens
        tokenizador.token_para_id = {
            token: indice for indice, token in enumerate(tokens)
        }
        return tokenizador

    @staticmethod
    def tokenizar(texto: str) -> list[str]:
        return TOKEN_RE.findall(texto.lower())

    def codificar(
        self,
        texto: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        ids = [
            self.token_para_id.get(token, self.unk_id)
            for token in self.tokenizar(texto)
        ]
        return (
            ([self.bos_id] if bos else [])
            + ids
            + ([self.eos_id] if eos else [])
        )

    def decodificar(self, ids: Iterable[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_para_token[int(token_id)]
            if token in {"<pad>", "<bos>"}:
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        texto = " ".join(tokens)
        for pontuacao in ".,;:!?":
            texto = texto.replace(" " + pontuacao, pontuacao)
        return texto[:1].upper() + texto[1:] if texto else texto

    @property
    def tamanho(self) -> int:
        return len(self.id_para_token)

    @property
    def pad_id(self) -> int:
        return self.token_para_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_para_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_para_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_para_id["<unk>"]
