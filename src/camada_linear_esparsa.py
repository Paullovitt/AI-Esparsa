"""Primitiva linear COO usada pela FFN do modelo V6.

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
        # A topologia deterministica nao e duplicada nos checkpoints.
        self.register_buffer(
            "indices",
            indices.to(torch.long).contiguous(),
            persistent=False,
        )
        self.valores = nn.Parameter(valores_iniciais.to(torch.float32))
        if bias_inicial is None:
            bias_inicial = torch.zeros(saidas)
        self.bias = nn.Parameter(bias_inicial.to(torch.float32))

    def matriz(self) -> Tensor:
        return torch.sparse_coo_tensor(
            self.indices,
            self.valores,
            (self.saidas, self.entradas),
            device=self.valores.device,
        ).coalesce()

    def forward(self, entrada: Tensor) -> Tensor:
        if entrada.ndim != 2 or entrada.shape[1] != self.entradas:
            raise ValueError("shape da entrada incompativel")
        return torch.sparse.mm(self.matriz(), entrada.t()).t() + self.bias

    @property
    def conexoes(self) -> int:
        return self.valores.numel()
