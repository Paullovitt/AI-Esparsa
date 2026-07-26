"""Variante V6.1 com ordem temporal explicita e esparsa.

O modulo existe apenas para experimento. Ele nao altera a V6 canonica.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from .atencao_causal_texto import (
    CodigoEsparsoCausal,
    ConfiguracaoTextoCausal,
    MiniLMCausal,
    VocabularioTexto,
)
from .modelo_v6 import (
    CONFIGURACAO_FFN_TEXTO,
    ConfiguracaoFFNTexto,
    ModeloV6,
)


DIMENSAO_FEATURES_BASE = 512
TOKENS_POR_FATO = 7
TOKENS_POR_PERGUNTA_RESPOSTA = 17


def periodo_historia(fatos_por_historia: int) -> int:
    """Calcula BOS + fatos + perguntas/respostas + EOS."""

    return (
        2
        + fatos_por_historia
        * (TOKENS_POR_FATO + TOKENS_POR_PERGUNTA_RESPOSTA)
    )


def codificacao_senoidal(
    comprimento: int,
    dimensao: int,
) -> Tensor:
    """Cria posicoes fixas, sem parametros de uma rede densa."""

    posicoes = torch.arange(comprimento, dtype=torch.float32).unsqueeze(1)
    frequencias = torch.exp(
        torch.arange(0, dimensao, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimensao)
    )
    codigos = torch.zeros(comprimento, dimensao)
    codigos[:, 0::2] = torch.sin(posicoes * frequencias)
    codigos[:, 1::2] = torch.cos(posicoes * frequencias)
    return codigos


class CodigoTemporalEsparso(nn.Module):
    """Acrescenta codigos de slot e papel aos descritores da V6."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao: ConfiguracaoTextoCausal,
        semente: int = 20260730,
    ) -> None:
        super().__init__()
        esperada = (
            DIMENSAO_FEATURES_BASE
            + configuracao.fatos_por_historia
            + 2
        )
        if configuracao.dimensao_features != esperada:
            raise ValueError(
                "dimensao_features deve reservar uma feature por fato"
            )
        configuracao_base = replace(
            configuracao,
            dimensao_features=DIMENSAO_FEATURES_BASE,
        )
        self.configuracao = configuracao
        self.codigo_base = CodigoEsparsoCausal(
            vocabulario,
            configuracao_base,
            semente=semente,
        )
        self.periodo = periodo_historia(
            configuracao.fatos_por_historia
        )
        self.register_buffer(
            "codigos_temporais_periodo",
            torch.stack(
                (
                    self._slots(
                        self.periodo,
                        torch.device("cpu"),
                    ),
                    self._papeis(
                        self.periodo,
                        torch.device("cpu"),
                    ),
                ),
                dim=-1,
            ),
            persistent=False,
        )

    @property
    def largura_descritor(self) -> int:
        return self.codigo_base.largura_descritor

    def _slots(self, comprimento: int, dispositivo: torch.device) -> Tensor:
        posicoes = (
            torch.arange(comprimento, device=dispositivo)
            % self.periodo
        )
        slots = torch.zeros_like(posicoes)
        validos = torch.zeros_like(posicoes, dtype=torch.bool)

        inicio_fatos = 1
        fim_fatos = (
            inicio_fatos
            + TOKENS_POR_FATO
            * self.configuracao.fatos_por_historia
        )
        nos_fatos = (posicoes >= inicio_fatos) & (
            posicoes < fim_fatos
        )
        slots[nos_fatos] = (
            posicoes[nos_fatos] - inicio_fatos
        ) // TOKENS_POR_FATO
        validos |= nos_fatos

        inicio_perguntas = fim_fatos
        fim_perguntas = (
            inicio_perguntas
            + TOKENS_POR_PERGUNTA_RESPOSTA
            * self.configuracao.fatos_por_historia
        )
        nas_perguntas = (posicoes >= inicio_perguntas) & (
            posicoes < fim_perguntas
        )
        slots[nas_perguntas] = (
            posicoes[nas_perguntas] - inicio_perguntas
        ) // TOKENS_POR_PERGUNTA_RESPOSTA
        validos |= nas_perguntas

        features = (
            DIMENSAO_FEATURES_BASE + 1 + slots
        )
        return torch.where(validos, features, torch.zeros_like(features))

    def _papeis(
        self,
        comprimento: int,
        dispositivo: torch.device,
    ) -> Tensor:
        """Marca posicoes que armazenam ou consultam objeto e entidade."""

        posicoes = (
            torch.arange(comprimento, device=dispositivo)
            % self.periodo
        )
        papeis = torch.zeros_like(posicoes)
        inicio_fatos = 1
        fim_fatos = (
            inicio_fatos
            + TOKENS_POR_FATO
            * self.configuracao.fatos_por_historia
        )
        nos_fatos = (posicoes >= inicio_fatos) & (
            posicoes < fim_fatos
        )
        offset_fato = (posicoes - inicio_fatos) % TOKENS_POR_FATO

        inicio_perguntas = fim_fatos
        fim_perguntas = (
            inicio_perguntas
            + TOKENS_POR_PERGUNTA_RESPOSTA
            * self.configuracao.fatos_por_historia
        )
        nas_perguntas = (posicoes >= inicio_perguntas) & (
            posicoes < fim_perguntas
        )
        offset_pergunta = (
            posicoes - inicio_perguntas
        ) % TOKENS_POR_PERGUNTA_RESPOSTA

        papel_objeto = (
            (nos_fatos & offset_fato.eq(3))
            | (
                nas_perguntas
                & (
                    offset_pergunta.eq(2)
                    | offset_pergunta.eq(3)
                    | offset_pergunta.eq(8)
                    | offset_pergunta.eq(9)
                )
            )
        )
        papel_entidade = (
            (nos_fatos & offset_fato.eq(1))
            | (
                nas_perguntas
                & (
                    offset_pergunta.eq(5)
                    | offset_pergunta.eq(11)
                )
            )
        )
        primeiro_papel = (
            DIMENSAO_FEATURES_BASE
            + self.configuracao.fatos_por_historia
            + 1
        )
        papeis = torch.where(
            papel_objeto,
            torch.full_like(papeis, primeiro_papel),
            papeis,
        )
        return torch.where(
            papel_entidade,
            torch.full_like(papeis, primeiro_papel + 1),
            papeis,
        )

    def descritores(self, tokens: Tensor) -> Tensor:
        base = self.codigo_base.descritores(tokens)
        if tokens.shape[1] <= self.periodo:
            temporais = self.codigos_temporais_periodo[
                : tokens.shape[1]
            ]
        else:
            indices = (
                torch.arange(tokens.shape[1], device=tokens.device)
                % self.periodo
            )
            temporais = self.codigos_temporais_periodo[indices]
        temporais = temporais.unsqueeze(0).expand(
            tokens.shape[0],
            -1,
            -1,
        )
        campos_base = base[..., -2:]
        livres = campos_base.eq(0).all(dim=-1, keepdim=True)
        campos_finais = torch.where(
            livres,
            temporais,
            campos_base,
        )
        return torch.cat((base[..., :-2], campos_finais), dim=-1)

    def auditoria(self) -> dict[str, int | bool | str]:
        return {
            "tipo": "codigo_v6_com_slot_temporal_esparso",
            "features_base": DIMENSAO_FEATURES_BASE,
            "features_temporais": (
                self.configuracao.fatos_por_historia + 2
            ),
            "features_ativas_temporais_por_token": 2,
            "papeis_temporais": "objeto_entidade",
            "cache_temporal": True,
            "largura_descritor": self.largura_descritor,
            "parametros_treinaveis": sum(
                parametro.numel() for parametro in self.parameters()
            ),
            "usa_nn_linear": any(
                isinstance(modulo, nn.Linear)
                for modulo in self.modules()
            ),
        }


class MiniLMCausalPosicional(MiniLMCausal):
    """Atencao V6 com posicao senoidal somada ao estado local."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao: ConfiguracaoTextoCausal,
        periodo: int,
    ) -> None:
        super().__init__(vocabulario, configuracao, "esparsa")
        self.periodo = periodo
        self.register_buffer(
            "posicoes_senoidais",
            codificacao_senoidal(
                periodo,
                configuracao.dimensao_modelo,
            ),
            persistent=False,
        )
        self.gate_posicao = nn.Parameter(torch.tensor(-0.5))

    def estados_locais(
        self,
        tokens: Tensor,
        embedding: Tensor | None = None,
    ) -> Tensor:
        if embedding is None:
            embedding = self.embedding(tokens)
        if tokens.shape[1] <= self.periodo:
            posicoes = self.posicoes_senoidais[: tokens.shape[1]]
        else:
            indices = (
                torch.arange(tokens.shape[1], device=tokens.device)
                % self.periodo
            )
            posicoes = self.posicoes_senoidais[indices]
        posicoes = posicoes.unsqueeze(0)
        entrada = embedding + torch.sigmoid(
            self.gate_posicao
        ) * posicoes
        return super().estados_locais(tokens, entrada)


class ModeloV61Posicional(ModeloV6):
    """V6 experimental com ordem temporal, sem trocar atencao ou FFN."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao_texto: ConfiguracaoTextoCausal,
        semente_topologia: int = 20260730,
        configuracao_ffn: ConfiguracaoFFNTexto = CONFIGURACAO_FFN_TEXTO,
    ) -> None:
        super().__init__(
            vocabulario,
            configuracao_texto,
            semente_topologia,
            configuracao_ffn,
        )
        self.atencao = MiniLMCausalPosicional(
            vocabulario,
            configuracao_texto,
            periodo_historia(
                configuracao_texto.fatos_por_historia
            ),
        )

    def auditoria(self) -> dict[str, int | float | bool | str]:
        resultado = super().auditoria()
        resultado.update(
            {
                "modelo": "V6.1-posicional-experimental",
                "posicao": "senoidal_fixa",
                "parametros_posicionais": 1,
                "gate_posicao": float(
                    torch.sigmoid(
                        self.atencao.gate_posicao
                    ).detach()
                ),
            }
        )
        return resultado
