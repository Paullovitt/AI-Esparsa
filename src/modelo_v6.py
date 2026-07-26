"""Modelo V6: atencao causal esparsa com FFN bloco-esparsa.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .atencao_causal_texto import (
    ConfiguracaoTextoCausal,
    MiniLMCausal,
    VocabularioTexto,
)
from .camada_linear_esparsa import LinearBlocoEsparsa


@dataclass(frozen=True)
class ConfiguracaoFFNTexto:
    dimensao: int = 48
    especialistas: int = 144
    fan_in: int = 8
    fan_out: int = 16
    top_k: int = 12


CONFIGURACAO_FFN_TEXTO = ConfiguracaoFFNTexto()


def _topologia_entrada(
    configuracao: ConfiguracaoFFNTexto,
    semente: int,
) -> tuple[Tensor, Tensor]:
    gerador = torch.Generator().manual_seed(semente)
    linhas = []
    colunas = []
    for especialista in range(configuracao.especialistas):
        dimensoes = torch.randperm(
            configuracao.dimensao, generator=gerador
        )[: configuracao.fan_in]
        linhas.extend([especialista] * configuracao.fan_in)
        colunas.extend(dimensoes.tolist())
    limite = (6.0 / configuracao.fan_in) ** 0.5
    valores = (
        torch.rand(len(linhas), generator=gerador) * 2 - 1
    ) * limite
    return torch.tensor((linhas, colunas)), valores


def _topologia_saida(
    configuracao: ConfiguracaoFFNTexto,
    semente: int,
) -> tuple[Tensor, Tensor]:
    gerador = torch.Generator().manual_seed(semente + 1)
    linhas = []
    colunas = []
    for especialista in range(configuracao.especialistas):
        dimensoes = torch.randperm(
            configuracao.dimensao, generator=gerador
        )[: configuracao.fan_out]
        linhas.extend(dimensoes.tolist())
        colunas.extend([especialista] * configuracao.fan_out)
    limite = (6.0 / configuracao.especialistas) ** 0.5
    valores = (
        torch.rand(len(linhas), generator=gerador) * 2 - 1
    ) * limite
    return torch.tensor((linhas, colunas)), valores


class FFNTextoBlocoEsparsa(nn.Module):
    """FFN generica com duas projecoes COO e ativacao Top-K."""

    def __init__(
        self,
        configuracao: ConfiguracaoFFNTexto = CONFIGURACAO_FFN_TEXTO,
        semente: int = 20260726,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        indices_1, valores_1 = _topologia_entrada(
            configuracao, semente
        )
        indices_2, valores_2 = _topologia_saida(
            configuracao, semente
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
        ativacoes = F.relu(self.camada_1(planos))
        valores, indices = ativacoes.topk(
            self.configuracao.top_k, dim=-1
        )
        podadas = torch.zeros_like(ativacoes)
        podadas.scatter_(1, indices, valores)
        saida = self.camada_2(podadas)
        return saida.view(*forma)

    def auditoria(self) -> dict[str, int | float | bool | str]:
        conexoes = self.camada_1.conexoes + self.camada_2.conexoes
        densas = (
            self.configuracao.dimensao
            * self.configuracao.especialistas
            * 2
        )
        return {
            "tipo": "ffn_texto_bloco_esparsa",
            "parametros": sum(p.numel() for p in self.parameters()),
            "conexoes": conexoes,
            "densidade": conexoes / densas,
            "top_k": self.configuracao.top_k,
            "usa_nn_linear": any(
                isinstance(modulo, nn.Linear) for modulo in self.modules()
            ),
        }


class ModeloV6(nn.Module):
    """Base V6 com atencao e FFN esparsas sempre ativas."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao_texto: ConfiguracaoTextoCausal,
        semente_topologia: int = 20260726,
        configuracao_ffn: ConfiguracaoFFNTexto = CONFIGURACAO_FFN_TEXTO,
    ) -> None:
        super().__init__()
        if configuracao_texto.dimensao_modelo != configuracao_ffn.dimensao:
            raise ValueError("dimensoes da atencao e FFN divergem")
        self.configuracao_texto = configuracao_texto
        self.configuracao_ffn = configuracao_ffn
        self.atencao = MiniLMCausal(
            vocabulario, configuracao_texto, "esparsa"
        )
        self.ffn = FFNTextoBlocoEsparsa(
            configuracao_ffn, semente_topologia
        )
        self.normalizacao_ffn = nn.LayerNorm(
            configuracao_ffn.dimensao
        )
        self.gate_ffn = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        tokens: Tensor,
        descritores: Tensor,
        candidatos: Tensor,
        candidatos_validos: Tensor,
        *,
        usar_atencao: bool = True,
        usar_ffn: bool = True,
    ) -> tuple[Tensor, Tensor]:
        oculto, posicao = self.atencao.estado_contextual(
            tokens,
            descritores,
            candidatos,
            candidatos_validos,
            usar_atencao,
        )
        if usar_ffn:
            residual = self.ffn(oculto)
            oculto = self.normalizacao_ffn(
                oculto + torch.sigmoid(self.gate_ffn) * residual
            )
        return self.atencao.saida(oculto), posicao

    def parametros_qk_esparsos(self) -> list[nn.Parameter]:
        return self.atencao.parametros_esparsos()

    def parametros_otimizador_denso(self) -> list[nn.Parameter]:
        ids_qk = {id(parametro) for parametro in self.parametros_qk_esparsos()}
        return [
            parametro
            for parametro in self.parameters()
            if id(parametro) not in ids_qk
        ]

    def auditoria(self) -> dict[str, int | float | bool | str]:
        return {
            "modelo": "V6",
            "parametros_totais": sum(
                parametro.numel() for parametro in self.parameters()
            ),
            "parametros_qk_densos": 0,
            "ffn_esparsa": self.ffn.auditoria(),
            "gate_ffn": float(
                torch.sigmoid(self.gate_ffn).detach()
            ),
        }
