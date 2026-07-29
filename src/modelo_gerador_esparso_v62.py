"""Runtime-base V6.2 com cache causal exato para inferencia.

Os pesos, a topologia COO, a atencao Top-K, a FFN, os residuais e as
normalizacoes continuam sendo os da V6.1. A diferenca desta versao e somente
o caminho incremental: chaves e valores causais ja calculados sao mantidos
por camada durante a geracao.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .camada_linear_esparsa import LinearBlocoEsparsa
from .modelo_gerador_esparso import (
    BlocoGeradorEsparso,
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)


def _gate_inferencia(modulo: object, nome: str) -> Tensor:
    """Reutiliza o gate imutavel enquanto a versao do parametro nao mudar."""

    parametro = getattr(modulo, nome)
    nome_cache = f"_cache_{nome}"
    nome_versao = f"_versao_{nome}"
    cache = getattr(modulo, nome_cache, None)
    versao = getattr(modulo, nome_versao, -1)
    if (
        cache is None
        or versao != parametro._version
        or cache.device != parametro.device
        or cache.dtype != parametro.dtype
    ):
        cache = torch.sigmoid(parametro.detach())
        setattr(modulo, nome_cache, cache)
        setattr(modulo, nome_versao, parametro._version)
    return cache


@dataclass
class CacheCamadaV62:
    """Buffers prealocados de uma camada para evitar concatenacoes por token."""

    valores: Tensor
    chaves: Tensor


@dataclass
class CacheGeracaoV62:
    """Estado causal completo de uma geracao incremental."""

    tokens: Tensor
    camadas: tuple[CacheCamadaV62, ...]
    comprimento: int
    capacidade: int


class ModeloGeradorEsparsoV62(ModeloGeradorEsparso):
    """Runtime-base com cache incremental, sem parametros adicionais."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorEsparso = ConfiguracaoGeradorEsparso(),
        semente: int = 20260810,
    ) -> None:
        super().__init__(
            vocabulario,
            pad_id,
            configuracao,
            semente,
        )
        # CSR evita a conversao interna COO->CSR repetida pelo cuSPARSE em
        # cada um dos doze produtos esparsos executados por token.
        for modulo in self.modules():
            if isinstance(modulo, LinearBlocoEsparsa):
                modulo.configurar_cache_csr_inferencia()

    @staticmethod
    def _aplicar_residuais(
        bloco: BlocoGeradorEsparso,
        estados: Tensor,
        contexto: Tensor,
    ) -> Tensor:
        """Reproduz exatamente o fechamento residual do bloco oficial."""

        estados = (
            estados
            + _gate_inferencia(bloco, "gate_atencao") * contexto
        )
        transformados = bloco.ffn(
            bloco.normalizacao_ffn(estados)
        )
        return estados + _gate_inferencia(
            bloco,
            "gate_ffn",
        ) * transformados

    @staticmethod
    def _contexto_prefill(
        bloco: BlocoGeradorEsparso,
        normalizados: Tensor,
        consultas: Tensor,
        chaves: Tensor,
    ) -> Tensor:
        """Executa a referencia Top-K em blocos e devolve o contexto completo."""

        comprimento = normalizados.shape[1]
        quantidade = min(
            bloco.configuracao.top_k_atencao,
            comprimento,
        )
        chaves_transpostas = chaves.transpose(1, 2)
        posicoes_chave = torch.arange(
            comprimento,
            device=normalizados.device,
        )
        contextos: list[Tensor] = []

        for inicio in range(
            0,
            comprimento,
            bloco.configuracao.bloco_consultas_atencao,
        ):
            fim = min(
                inicio
                + bloco.configuracao.bloco_consultas_atencao,
                comprimento,
            )
            pontuacoes = torch.bmm(
                consultas[:, inicio:fim],
                chaves_transpostas,
            ) / math.sqrt(bloco.configuracao.dimensao)
            posicoes_consulta = torch.arange(
                inicio,
                fim,
                device=normalizados.device,
            ).unsqueeze(1)
            mascara_causal = (
                posicoes_chave.unsqueeze(0) <= posicoes_consulta
            )
            pontuacoes.masked_fill_(
                ~mascara_causal.unsqueeze(0),
                -torch.inf,
            )
            valores, indices = pontuacoes.topk(
                quantidade,
                dim=-1,
                sorted=False,
            )
            pesos = torch.softmax(valores, dim=-1)
            estados_expandidos = normalizados.unsqueeze(1).expand(
                -1,
                fim - inicio,
                -1,
                -1,
            )
            selecionados = torch.gather(
                estados_expandidos,
                2,
                indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    normalizados.shape[-1],
                ),
            )
            contextos.append(
                (selecionados * pesos.unsqueeze(-1)).sum(dim=2)
            )
        return torch.cat(contextos, dim=1)

    @staticmethod
    def _contexto_incremental(
        bloco: BlocoGeradorEsparso,
        consulta: Tensor,
        cache: CacheCamadaV62,
        comprimento: int,
    ) -> Tensor:
        """Calcula somente a consulta do token novo contra o prefixo cacheado."""

        chaves = cache.chaves[:, :comprimento]
        valores_cache = cache.valores[:, :comprimento]
        pontuacoes = torch.bmm(
            consulta,
            chaves.transpose(1, 2),
        ) / math.sqrt(bloco.configuracao.dimensao)
        quantidade = min(
            bloco.configuracao.top_k_atencao,
            comprimento,
        )
        valores, indices = pontuacoes.topk(
            quantidade,
            dim=-1,
            sorted=False,
        )
        pesos = torch.softmax(valores, dim=-1)
        valores_expandidos = valores_cache.unsqueeze(1)
        selecionados = torch.gather(
            valores_expandidos,
            2,
            indices.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                valores_cache.shape[-1],
            ),
        )
        return (selecionados * pesos.unsqueeze(-1)).sum(dim=2)

    def _validar_modo_cache(self) -> None:
        if self.training:
            raise RuntimeError(
                "o cache V6.2 e exclusivo de modelo em modo eval"
            )
        if torch.is_grad_enabled():
            raise RuntimeError(
                "o cache V6.2 exige inference_mode ou no_grad"
            )

    @torch.inference_mode()
    def iniciar_cache_geracao(
        self,
        tokens: Tensor,
    ) -> tuple[Tensor, CacheGeracaoV62]:
        """Processa o prompt uma vez e prepara os buffers causais por camada."""

        self._validar_modo_cache()
        if tokens.ndim != 2:
            raise ValueError("tokens deve possuir shape [lote, tempo]")
        if tokens.shape[1] == 0:
            raise ValueError("a sequencia nao pode ser vazia")
        capacidade = self.configuracao.maximo_contexto
        if tokens.shape[1] > capacidade:
            tokens = tokens[:, -capacidade:]

        lote, comprimento = tokens.shape
        tokens_cache = torch.full(
            (lote, capacidade),
            self.pad_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens_cache[:, :comprimento].copy_(tokens)
        estados = self.embedding(tokens) * math.sqrt(
            self.configuracao.dimensao
        )
        estados = (
            estados
            + _gate_inferencia(self, "gate_posicao")
            * self.posicoes[:comprimento].unsqueeze(0)
        )
        caches: list[CacheCamadaV62] = []

        for bloco in self.blocos:
            normalizados = bloco.normalizacao_atencao(estados)
            consultas = bloco._projetar(
                bloco.consulta,
                normalizados,
            )
            chaves = bloco._projetar(
                bloco.chave,
                normalizados,
            )
            contexto = self._contexto_prefill(
                bloco,
                normalizados,
                consultas,
                chaves,
            )
            valores_cache = torch.empty(
                (
                    lote,
                    capacidade,
                    self.configuracao.dimensao,
                ),
                dtype=normalizados.dtype,
                device=normalizados.device,
            )
            chaves_cache = torch.empty_like(valores_cache)
            valores_cache[:, :comprimento].copy_(normalizados)
            chaves_cache[:, :comprimento].copy_(chaves)
            caches.append(
                CacheCamadaV62(
                    valores=valores_cache,
                    chaves=chaves_cache,
                )
            )
            estados = self._aplicar_residuais(
                bloco,
                estados,
                contexto,
            )

        normalizado_final = self.normalizacao_saida(estados[:, -1:])
        logits = F.linear(
            normalizado_final,
            self.embedding.weight,
            self.saida_bias,
        )
        return logits, CacheGeracaoV62(
            tokens=tokens_cache,
            camadas=tuple(caches),
            comprimento=comprimento,
            capacidade=capacidade,
        )

    @torch.inference_mode()
    def avancar_cache_geracao(
        self,
        novo_token: Tensor,
        cache: CacheGeracaoV62,
    ) -> tuple[Tensor, CacheGeracaoV62]:
        """Acrescenta um token e calcula somente seu caminho causal."""

        self._validar_modo_cache()
        if novo_token.ndim != 2 or novo_token.shape[1] != 1:
            raise ValueError("novo_token deve possuir shape [lote, 1]")
        if novo_token.shape[0] != cache.tokens.shape[0]:
            raise ValueError("lote do novo token diverge do cache")
        if novo_token.device != cache.tokens.device:
            raise ValueError("novo token e cache devem usar o mesmo dispositivo")
        if len(cache.camadas) != len(self.blocos):
            raise ValueError("quantidade de camadas do cache e invalida")

        if cache.comprimento >= cache.capacidade:
            # A V6.1 reinicia as posicoes ao deslizar a janela. Refazer o
            # prefill preserva exatamente esse contrato no raro overflow.
            janela = torch.cat(
                (
                    cache.tokens[:, 1 : cache.comprimento],
                    novo_token,
                ),
                dim=1,
            )
            return self.iniciar_cache_geracao(janela)

        posicao = cache.comprimento
        cache.tokens[:, posicao : posicao + 1].copy_(novo_token)
        estados = self.embedding(novo_token) * math.sqrt(
            self.configuracao.dimensao
        )
        estados = (
            estados
            + _gate_inferencia(self, "gate_posicao")
            * self.posicoes[posicao : posicao + 1].unsqueeze(0)
        )
        novo_comprimento = posicao + 1

        for bloco, cache_camada in zip(
            self.blocos,
            cache.camadas,
            strict=True,
        ):
            normalizados = bloco.normalizacao_atencao(estados)
            consulta = bloco._projetar(
                bloco.consulta,
                normalizados,
            )
            chave = bloco._projetar(
                bloco.chave,
                normalizados,
            )
            cache_camada.valores[
                :, posicao : novo_comprimento
            ].copy_(normalizados)
            cache_camada.chaves[
                :, posicao : novo_comprimento
            ].copy_(chave)
            contexto = self._contexto_incremental(
                bloco,
                consulta,
                cache_camada,
                novo_comprimento,
            )
            estados = self._aplicar_residuais(
                bloco,
                estados,
                contexto,
            )

        cache.comprimento = novo_comprimento
        normalizado_final = self.normalizacao_saida(estados)
        logits = F.linear(
            normalizado_final,
            self.embedding.weight,
            self.saida_bias,
        )
        return logits, cache

    def auditoria(self) -> dict[str, int | float | bool | str]:
        """Explicita que a V6.2 muda o runtime, nao a rede treinada."""

        auditoria = super().auditoria()
        auditoria.update(
            {
                "modelo": "gerador-esparso-v62-base-runtime",
                "cache_causal_incremental": True,
                "cache_pre_alocado": True,
                "cache_linear_csr": True,
                "cache_gates_escalares": True,
                "parametros_adicionais_cache": 0,
                "arquitetura_pesos_identica_oficial": True,
                "checkpoint_oficial": True,
            }
        )
        return auditoria
