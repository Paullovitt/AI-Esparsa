"""Baseline densa experimental, equivalente por orçamento de parâmetros.

A baseline mantém o contrato causal, os três blocos, os residuais, as
normalizações, a posição senoidal e o classificador amarrado. Q/K, atenção e
FFN são densos para isolar a diferença estrutural em relação ao modelo oficial.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modelo_gerador_esparso import codificacao_senoidal


@dataclass(frozen=True)
class ConfiguracaoGeradorDenso:
    """Dimensões escolhidas para aproximar os 163.667 parâmetros oficiais."""

    dimensao: int = 88
    camadas: int = 3
    especialistas: int = 160
    maximo_contexto: int = 640

    def validar(self) -> None:
        if self.dimensao <= 0 or self.camadas <= 0:
            raise ValueError("dimensao e camadas devem ser positivas")
        if self.especialistas <= 0:
            raise ValueError("especialistas deve ser positivo")
        if self.maximo_contexto <= 0:
            raise ValueError("maximo_contexto deve ser positivo")


class BlocoGeradorDenso(nn.Module):
    """Bloco causal convencional com Q/K e FFN totalmente conectados."""

    def __init__(self, configuracao: ConfiguracaoGeradorDenso) -> None:
        super().__init__()
        dimensao = configuracao.dimensao
        self.normalizacao_atencao = nn.LayerNorm(dimensao)
        self.consulta = nn.Linear(dimensao, dimensao)
        self.chave = nn.Linear(dimensao, dimensao)
        self.normalizacao_ffn = nn.LayerNorm(dimensao)
        self.ffn_entrada = nn.Linear(
            dimensao,
            configuracao.especialistas,
        )
        self.ffn_saida = nn.Linear(
            configuracao.especialistas,
            dimensao,
        )
        self.gate_atencao = nn.Parameter(torch.tensor(-0.25))
        self.gate_ffn = nn.Parameter(torch.tensor(-0.75))

    def forward(self, estados: Tensor) -> Tensor:
        normalizados = self.normalizacao_atencao(estados)
        consultas = self.consulta(normalizados).unsqueeze(1)
        chaves = self.chave(normalizados).unsqueeze(1)
        valores = normalizados.unsqueeze(1)
        # A primitiva oficial do PyTorch permite o melhor kernel denso
        # disponível, sem prejudicar artificialmente a baseline.
        contexto = F.scaled_dot_product_attention(
            consultas,
            chaves,
            valores,
            is_causal=True,
        ).squeeze(1)
        estados = estados + torch.sigmoid(self.gate_atencao) * contexto
        transformados = self.ffn_saida(
            F.gelu(self.ffn_entrada(self.normalizacao_ffn(estados)))
        )
        return estados + torch.sigmoid(self.gate_ffn) * transformados


class ModeloGeradorDenso(nn.Module):
    """LM denso experimental com tamanho próximo do gerador esparso."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorDenso = ConfiguracaoGeradorDenso(),
        semente: int = 20260728,
    ) -> None:
        super().__init__()
        configuracao.validar()
        self.configuracao = configuracao
        self.pad_id = pad_id

        # O fork preserva a semente global do protocolo externo e torna a
        # inicialização do modelo reproduzível quando usado isoladamente.
        with torch.random.fork_rng():
            torch.manual_seed(semente)
            self.embedding = nn.Embedding(
                vocabulario,
                configuracao.dimensao,
                padding_idx=pad_id,
            )
            self.blocos = nn.ModuleList(
                BlocoGeradorDenso(configuracao)
                for _ in range(configuracao.camadas)
            )
            self.normalizacao_saida = nn.LayerNorm(
                configuracao.dimensao
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
        """Expõe as propriedades usadas para provar a justiça da baseline."""

        return {
            "modelo": "baseline-densa-parametros-equivalentes",
            "parametros": sum(
                parametro.numel() for parametro in self.parameters()
            ),
            "camadas": self.configuracao.camadas,
            "dimensao": self.configuracao.dimensao,
            "especialistas": self.configuracao.especialistas,
            "atencao_causal_densa": True,
            "qk_denso": True,
            "ffn_densa": True,
            "residuais": True,
            "normalizacao": True,
            "lineares_densas_internas": sum(
                isinstance(modulo, nn.Linear)
                for modulo in self.modules()
            ),
            "classificador_denso": True,
            "pesos_embedding_saida_amarrados": True,
            "checkpoint_oficial": False,
        }
