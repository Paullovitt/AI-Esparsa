"""Primitiva linear COO usada pelo Gerador Esparso Coerente.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearBlocoEsparsa(nn.Module):
    """Camada COO fixa em que somente conexoes existentes sao treinaveis."""

    def __init__(
        self,
        entradas: int,
        saidas: int,
        indices: Tensor,
        valores_iniciais: Tensor,
        bias_inicial: Tensor | None = None,
    ) -> None:
        super().__init__()
        if indices.shape[0] != 2 or indices.shape[1] != len(valores_iniciais):
            raise ValueError("indices e valores possuem shapes incompativeis")
        self.entradas = entradas
        self.saidas = saidas
        chaves = indices[0].to(torch.long) * entradas + indices[1].to(
            torch.long
        )
        ordem = torch.argsort(chaves, stable=True)
        # A ordenacao torna o COO coalescido sem consolidacao por forward.
        self.register_buffer(
            "indices",
            indices[:, ordem].to(torch.long).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "ordem_valores",
            ordem.to(torch.long).contiguous(),
            persistent=False,
        )
        self.valores = nn.Parameter(valores_iniciais.to(torch.float32))
        if bias_inicial is None:
            bias_inicial = torch.zeros(saidas)
        self.bias = nn.Parameter(bias_inicial.to(torch.float32))
        self._matriz_inferencia: Tensor | None = None
        self._versao_cache = -1

    def _construir_matriz(self, valores: Tensor) -> Tensor:
        # A topologia é validada uma vez pelos testes e não muda em runtime.
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                self.indices,
                valores[self.ordem_valores],
                (self.saidas, self.entradas),
                device=self.valores.device,
                is_coalesced=True,
                check_invariants=False,
            )

    def _limpar_cache(self) -> None:
        self._matriz_inferencia = None
        self._versao_cache = -1

    def _apply(self, func, recurse: bool = True):
        self._limpar_cache()
        resultado = super()._apply(func, recurse)
        self._limpar_cache()
        return resultado

    def train(self, mode: bool = True):
        if mode:
            self._limpar_cache()
        return super().train(mode)

    def matriz(self) -> Tensor:
        if self.training or torch.is_grad_enabled():
            return self._construir_matriz(self.valores)
        versao = self.valores._version
        if (
            self._matriz_inferencia is None
            or self._versao_cache != versao
        ):
            self._matriz_inferencia = self._construir_matriz(
                self.valores.detach()
            )
            self._versao_cache = versao
        return self._matriz_inferencia

    def forward(self, entrada: Tensor) -> Tensor:
        if entrada.ndim != 2 or entrada.shape[1] != self.entradas:
            raise ValueError("shape da entrada incompativel")
        return torch.sparse.mm(self.matriz(), entrada.t()).t() + self.bias

    @property
    def conexoes(self) -> int:
        return self.valores.numel()
