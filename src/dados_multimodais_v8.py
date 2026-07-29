"""Manifesto e carregamento de dados imagem-texto para a V8.

O projeto não baixa nem incorpora datasets. Cada linha JSONL referencia uma
imagem local e contém os campos ``id``, ``imagem``, ``prompt`` e ``resposta``.
Os caminhos são sempre relativos à raiz informada e não podem escapar dela.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizador_palavras import TokenizadorPalavras


EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class RegistroMultimodalV8:
    """Um par supervisionado imagem, pergunta e resposta."""

    identificador: str
    imagem: Path
    prompt: str
    resposta: str

    @property
    def texto_completo(self) -> str:
        return f"pergunta: {self.prompt} resposta: {self.resposta}"


@dataclass(frozen=True)
class ExemploCodificadoV8:
    """Tensores textuais de um registro e seu caminho de imagem."""

    identificador: str
    imagem: Path
    entradas: tuple[int, ...]
    alvos: tuple[int, ...]


def _resolver_imagem_segura(raiz: Path, caminho_relativo: str) -> Path:
    relativo = Path(caminho_relativo)
    if relativo.is_absolute():
        raise ValueError("o caminho da imagem deve ser relativo")
    caminho = (raiz / relativo).resolve()
    raiz_resolvida = raiz.resolve()
    if caminho != raiz_resolvida and raiz_resolvida not in caminho.parents:
        raise ValueError("o caminho da imagem escapou da raiz configurada")
    if caminho.suffix.lower() not in EXTENSOES_IMAGEM:
        raise ValueError(f"extensao de imagem nao suportada: {caminho.suffix}")
    if not caminho.is_file():
        raise FileNotFoundError(f"imagem nao encontrada: {caminho}")
    return caminho


def carregar_manifesto_multimodal(
    caminho_manifesto: Path,
    raiz_imagens: Path,
) -> list[RegistroMultimodalV8]:
    """Lê e valida integralmente um manifesto JSONL local."""

    if not caminho_manifesto.is_file():
        raise FileNotFoundError(
            f"manifesto nao encontrado: {caminho_manifesto}"
        )
    registros: list[RegistroMultimodalV8] = []
    identificadores: set[str] = set()
    with caminho_manifesto.open("r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue
            try:
                bruto = json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f"JSON invalido na linha {numero_linha}"
                ) from erro
            if not isinstance(bruto, dict):
                raise ValueError(
                    f"linha {numero_linha} deve conter um objeto JSON"
                )
            faltantes = {
                "id",
                "imagem",
                "prompt",
                "resposta",
            } - bruto.keys()
            if faltantes:
                raise ValueError(
                    f"linha {numero_linha} sem campos: "
                    + ", ".join(sorted(faltantes))
                )
            identificador = str(bruto["id"]).strip()
            prompt = str(bruto["prompt"]).strip()
            resposta = str(bruto["resposta"]).strip()
            if not identificador or not prompt or not resposta:
                raise ValueError(
                    f"linha {numero_linha} possui campo textual vazio"
                )
            if identificador in identificadores:
                raise ValueError(f"id duplicado no manifesto: {identificador}")
            identificadores.add(identificador)
            registros.append(
                RegistroMultimodalV8(
                    identificador=identificador,
                    imagem=_resolver_imagem_segura(
                        raiz_imagens,
                        str(bruto["imagem"]),
                    ),
                    prompt=prompt,
                    resposta=resposta,
                )
            )
    if not registros:
        raise ValueError("o manifesto multimodal esta vazio")
    return registros


def criar_tokenizador_multimodal(
    registros: Iterable[RegistroMultimodalV8],
) -> TokenizadorPalavras:
    """Cria o vocabulário somente a partir do split de treinamento."""

    textos = [registro.texto_completo for registro in registros]
    if not textos:
        raise ValueError("nenhum registro fornecido ao tokenizador")
    return TokenizadorPalavras(textos)


def codificar_registro_multimodal(
    registro: RegistroMultimodalV8,
    tokenizador: TokenizadorPalavras,
) -> ExemploCodificadoV8:
    """Monta LM causal e mascara o prompt, treinando apenas a resposta."""

    prefixo = tokenizador.codificar(
        f"pergunta: {registro.prompt} resposta:",
        bos=True,
        eos=False,
    )
    resposta = tokenizador.codificar(
        registro.resposta,
        bos=False,
        eos=True,
    )
    sequencia = prefixo + resposta
    entradas = sequencia[:-1]
    alvos = sequencia[1:]
    # O último token do prefixo deve prever o primeiro token da resposta.
    quantidade_mascarada = max(0, len(prefixo) - 1)
    alvos = [tokenizador.pad_id] * quantidade_mascarada + alvos[
        quantidade_mascarada:
    ]
    return ExemploCodificadoV8(
        identificador=registro.identificador,
        imagem=registro.imagem,
        entradas=tuple(entradas),
        alvos=tuple(alvos),
    )


def carregar_imagem_rgb(
    caminho: Path,
    altura: int,
    largura: int,
) -> Tensor:
    """Decodifica, corrige orientação e normaliza RGB para ``[-1, 1]``."""

    if altura <= 0 or largura <= 0:
        raise ValueError("altura e largura devem ser positivas")
    with Image.open(caminho) as original:
        imagem = ImageOps.exif_transpose(original).convert("RGB")
        imagem = imagem.resize((largura, altura), Image.Resampling.BICUBIC)
        buffer = bytearray(imagem.tobytes())
    pixels = torch.frombuffer(buffer, dtype=torch.uint8).clone()
    pixels = pixels.view(altura, largura, 3).permute(2, 0, 1)
    return pixels.to(torch.float32).div_(127.5).sub_(1.0)


class DatasetMultimodalV8(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset sem cache implícito: as imagens são lidas sob demanda."""

    def __init__(
        self,
        registros: list[RegistroMultimodalV8],
        tokenizador: TokenizadorPalavras,
        altura_imagem: int,
        largura_imagem: int,
    ) -> None:
        if not registros:
            raise ValueError("o dataset multimodal nao pode ser vazio")
        self.exemplos = [
            codificar_registro_multimodal(registro, tokenizador)
            for registro in registros
        ]
        self.altura_imagem = altura_imagem
        self.largura_imagem = largura_imagem

    def __len__(self) -> int:
        return len(self.exemplos)

    def __getitem__(self, indice: int) -> tuple[Tensor, Tensor, Tensor]:
        exemplo = self.exemplos[indice]
        imagem = carregar_imagem_rgb(
            exemplo.imagem,
            self.altura_imagem,
            self.largura_imagem,
        )
        return (
            torch.tensor(exemplo.entradas, dtype=torch.long),
            torch.tensor(exemplo.alvos, dtype=torch.long),
            imagem,
        )


def agrupar_lote_multimodal(
    exemplos: list[tuple[Tensor, Tensor, Tensor]],
    pad_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Aplica padding textual e empilha imagens já redimensionadas."""

    if not exemplos:
        raise ValueError("nao e possivel agrupar um lote vazio")
    maximo = max(entradas.numel() for entradas, _, _ in exemplos)
    entradas_lote = torch.full(
        (len(exemplos), maximo),
        pad_id,
        dtype=torch.long,
    )
    alvos_lote = torch.full_like(entradas_lote, pad_id)
    imagens = []
    for indice, (entradas, alvos, imagem) in enumerate(exemplos):
        comprimento = entradas.numel()
        entradas_lote[indice, :comprimento] = entradas
        alvos_lote[indice, :comprimento] = alvos
        imagens.append(imagem)
    return entradas_lote, alvos_lote, torch.stack(imagens)
