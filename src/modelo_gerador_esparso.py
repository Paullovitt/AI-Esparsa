"""Gerador causal com Q/K e FFN estruturalmente esparsas.

A atencao seleciona os estados Top-K depois de projetar consultas e chaves por
matrizes COO treinaveis. O modulo e a arquitetura-base para geracao
condicionada longa; a V6.1 permanece preservada como controle historico.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .camada_linear_esparsa import LinearBlocoEsparsa


@dataclass(frozen=True)
class ConfiguracaoGeradorEsparso:
    """Hiperparametros do gerador condicionado por prompt."""

    dimensao: int = 128
    camadas: int = 3
    top_k_atencao: int = 32
    fan_in_qk: int = 32
    especialistas: int = 384
    fan_in_ffn: int = 32
    fan_out_ffn: int = 48
    top_k_ffn: int = 64
    maximo_contexto: int = 640

    def validar(self) -> None:
        if self.dimensao <= 0 or self.camadas <= 0:
            raise ValueError("dimensao e camadas devem ser positivas")
        if not 0 < self.top_k_atencao <= self.maximo_contexto:
            raise ValueError("top_k_atencao invalido")
        if not 0 < self.fan_in_qk <= self.dimensao:
            raise ValueError("fan_in_qk invalido")
        if not 0 < self.fan_in_ffn <= self.dimensao:
            raise ValueError("fan_in_ffn invalido")
        if not 0 < self.fan_out_ffn <= self.dimensao:
            raise ValueError("fan_out_ffn invalido")
        if not 0 < self.top_k_ffn <= self.especialistas:
            raise ValueError("top_k_ffn invalido")


def codificacao_senoidal(comprimento: int, dimensao: int) -> Tensor:
    """Cria posicoes fixas sem acrescentar uma matriz densa treinavel."""

    posicoes = torch.arange(comprimento, dtype=torch.float32).unsqueeze(1)
    frequencias = torch.exp(
        torch.arange(0, dimensao, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimensao)
    )
    codigos = torch.zeros(comprimento, dimensao)
    codigos[:, 0::2] = torch.sin(posicoes * frequencias)
    codigos[:, 1::2] = torch.cos(
        posicoes * frequencias[: codigos[:, 1::2].shape[1]]
    )
    return codigos


def _topologia_por_saida(
    entradas: int,
    saidas: int,
    fan_in: int,
    semente: int,
) -> tuple[Tensor, Tensor]:
    """Gera uma topologia COO deterministica e balanceada por saida."""

    gerador = torch.Generator().manual_seed(semente)
    linhas: list[int] = []
    colunas: list[int] = []
    for saida in range(saidas):
        escolhidas = torch.randperm(
            entradas,
            generator=gerador,
        )[:fan_in]
        linhas.extend([saida] * fan_in)
        colunas.extend(escolhidas.tolist())
    limite = math.sqrt(6.0 / max(1, fan_in))
    valores = (
        torch.rand(len(linhas), generator=gerador) * 2.0 - 1.0
    ) * limite
    return torch.tensor((linhas, colunas)), valores


def _topologia_por_entrada(
    entradas: int,
    saidas: int,
    fan_out: int,
    semente: int,
) -> tuple[Tensor, Tensor]:
    """Gera uma topologia COO deterministica e balanceada por entrada."""

    gerador = torch.Generator().manual_seed(semente)
    linhas: list[int] = []
    colunas: list[int] = []
    for entrada in range(entradas):
        escolhidas = torch.randperm(
            saidas,
            generator=gerador,
        )[:fan_out]
        linhas.extend(escolhidas.tolist())
        colunas.extend([entrada] * fan_out)
    limite = math.sqrt(6.0 / max(1, entradas))
    valores = (
        torch.rand(len(linhas), generator=gerador) * 2.0 - 1.0
    ) * limite
    return torch.tensor((linhas, colunas)), valores


class FFNGeradorEsparsa(nn.Module):
    """Transformacao nao linear COO com ativacoes Top-K."""

    def __init__(
        self,
        configuracao: ConfiguracaoGeradorEsparso,
        semente: int,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        indices_1, valores_1 = _topologia_por_saida(
            configuracao.dimensao,
            configuracao.especialistas,
            configuracao.fan_in_ffn,
            semente,
        )
        indices_2, valores_2 = _topologia_por_entrada(
            configuracao.especialistas,
            configuracao.dimensao,
            configuracao.fan_out_ffn,
            semente + 1,
        )
        self.camada_1 = LinearBlocoEsparsa(
            configuracao.dimensao,
            configuracao.especialistas,
            indices_1,
            valores_1,
        )
        self.camada_2 = LinearBlocoEsparsa(
            configuracao.especialistas,
            configuracao.dimensao,
            indices_2,
            valores_2,
        )

    def forward(self, estados: Tensor) -> Tensor:
        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        ativacoes = F.gelu(self.camada_1(planos))
        valores, indices = ativacoes.topk(
            self.configuracao.top_k_ffn,
            dim=-1,
        )
        podadas = torch.zeros_like(ativacoes)
        podadas.scatter_(1, indices, valores)
        return self.camada_2(podadas).view(*forma)


class BlocoGeradorEsparso(nn.Module):
    """Atencao causal Top-K com projecoes Q/K COO e residual FFN."""

    def __init__(
        self,
        configuracao: ConfiguracaoGeradorEsparso,
        semente: int,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        indices_q, valores_q = _topologia_por_saida(
            configuracao.dimensao,
            configuracao.dimensao,
            configuracao.fan_in_qk,
            semente,
        )
        indices_k, valores_k = _topologia_por_saida(
            configuracao.dimensao,
            configuracao.dimensao,
            configuracao.fan_in_qk,
            semente + 1,
        )
        self.consulta = LinearBlocoEsparsa(
            configuracao.dimensao,
            configuracao.dimensao,
            indices_q,
            valores_q,
        )
        self.chave = LinearBlocoEsparsa(
            configuracao.dimensao,
            configuracao.dimensao,
            indices_k,
            valores_k,
        )
        self.normalizacao_atencao = nn.LayerNorm(configuracao.dimensao)
        self.normalizacao_ffn = nn.LayerNorm(configuracao.dimensao)
        self.ffn = FFNGeradorEsparsa(configuracao, semente + 2)
        self.gate_atencao = nn.Parameter(torch.tensor(-0.25))
        self.gate_ffn = nn.Parameter(torch.tensor(-0.75))

    @staticmethod
    def _projetar(camada: LinearBlocoEsparsa, estados: Tensor) -> Tensor:
        forma = estados.shape
        return camada(estados.reshape(-1, forma[-1])).view(*forma)

    def _atencao_top_k(self, estados: Tensor) -> Tensor:
        normalizados = self.normalizacao_atencao(estados)
        consultas = self._projetar(self.consulta, normalizados)
        chaves = self._projetar(self.chave, normalizados)
        pontuacoes = torch.bmm(
            consultas,
            chaves.transpose(1, 2),
        ) / math.sqrt(self.configuracao.dimensao)

        comprimento = estados.shape[1]
        mascara_causal = torch.ones(
            comprimento,
            comprimento,
            dtype=torch.bool,
            device=estados.device,
        ).tril()
        pontuacoes = pontuacoes.masked_fill(
            ~mascara_causal.unsqueeze(0),
            -torch.inf,
        )
        quantidade = min(self.configuracao.top_k_atencao, comprimento)
        valores, indices = pontuacoes.topk(
            quantidade,
            dim=-1,
            sorted=False,
        )
        pesos_top = torch.softmax(valores, dim=-1)
        # A matriz de pesos continua Top-K; a forma densa evita materializar
        # [lote, tempo, Top-K, dimensao] durante o produto com os valores.
        pesos = torch.zeros_like(pontuacoes)
        pesos.scatter_(-1, indices, pesos_top)
        return torch.bmm(pesos, normalizados)

    def forward(self, estados: Tensor) -> Tensor:
        contexto = self._atencao_top_k(estados)
        estados = estados + torch.sigmoid(self.gate_atencao) * contexto
        transformados = self.ffn(self.normalizacao_ffn(estados))
        return estados + torch.sigmoid(self.gate_ffn) * transformados

    def conexoes_qk(self) -> int:
        return self.consulta.conexoes + self.chave.conexoes


class ModeloGeradorEsparso(nn.Module):
    """LM causal condicionado pelo prompt com classificador amarrado."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorEsparso = ConfiguracaoGeradorEsparso(),
        semente: int = 20260810,
    ) -> None:
        super().__init__()
        configuracao.validar()
        self.configuracao = configuracao
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            vocabulario,
            configuracao.dimensao,
            padding_idx=pad_id,
        )
        gerador = torch.Generator().manual_seed(semente)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(
                    vocabulario,
                    configuracao.dimensao,
                    generator=gerador,
                )
                * 0.025
            )
            self.embedding.weight[pad_id].zero_()
        self.register_buffer(
            "posicoes",
            codificacao_senoidal(
                configuracao.maximo_contexto,
                configuracao.dimensao,
            ),
            persistent=False,
        )
        self.gate_posicao = nn.Parameter(torch.tensor(-0.5))
        self.blocos = nn.ModuleList(
            BlocoGeradorEsparso(
                configuracao,
                semente + 100 * indice,
            )
            for indice in range(configuracao.camadas)
        )
        self.normalizacao_saida = nn.LayerNorm(configuracao.dimensao)
        self.saida_bias = nn.Parameter(torch.zeros(vocabulario))

    def forward(self, tokens: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if tokens.ndim != 2:
            raise ValueError("tokens deve possuir shape [lote, tempo]")
        if tokens.shape[1] == 0:
            raise ValueError("a sequencia nao pode ser vazia")
        if tokens.shape[1] > self.configuracao.maximo_contexto:
            tokens = tokens[:, -self.configuracao.maximo_contexto :]

        estados = self.embedding(tokens) * math.sqrt(
            self.configuracao.dimensao
        )
        estados = estados + torch.sigmoid(self.gate_posicao) * self.posicoes[
            : tokens.shape[1]
        ].unsqueeze(0)
        for bloco in self.blocos:
            estados = bloco(estados)
        normalizados = self.normalizacao_saida(estados)
        logits = F.linear(
            normalizados,
            self.embedding.weight,
            self.saida_bias,
        )
        return logits, {
            "gate_posicao": torch.sigmoid(self.gate_posicao),
        }

    def auditoria(self) -> dict[str, int | float | bool | str]:
        conexoes_qk = sum(
            bloco.conexoes_qk() for bloco in self.blocos
        )
        conexoes_qk_densas = (
            self.configuracao.camadas
            * 2
            * self.configuracao.dimensao**2
        )
        conexoes_ffn = sum(
            bloco.ffn.camada_1.conexoes
            + bloco.ffn.camada_2.conexoes
            for bloco in self.blocos
        )
        lineares = sum(
            isinstance(modulo, nn.Linear)
            for modulo in self.modules()
        )
        return {
            "modelo": "gerador-esparso-coerente-base",
            "parametros": sum(
                parametro.numel() for parametro in self.parameters()
            ),
            "camadas": self.configuracao.camadas,
            "atencao_causal_top_k": self.configuracao.top_k_atencao,
            "qk_coo": True,
            "conexoes_qk": conexoes_qk,
            "densidade_qk": conexoes_qk / conexoes_qk_densas,
            "ffn_coo": True,
            "conexoes_ffn": conexoes_ffn,
            "residuais": True,
            "normalizacao": True,
            "lineares_densas_internas": lineares,
            "classificador_denso": True,
            "pesos_embedding_saida_amarrados": True,
        }
