"""V7.3 com roteamento combinatório e FFN fundida em CUDA.

O modelo usa seu próprio checkpoint-base. Em inferência
CUDA FP32, um único kernel executa projeção de entrada, GELU, Top-64, gates e
projeção de saída para somente os quatro microgrupos selecionados. Ambientes
sem o toolchain ou chamadas com gradiente recuam para o runtime PyTorch
condicional da própria V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .kernel_cuda_v73 import (
    carregar_kernel_cuda_v73,
    erro_kernel_cuda_v73,
)
from .modelo_gerador_esparso import (
    BlocoGeradorEsparso,
    ConfiguracaoGeradorEsparso,
)
from .runtime_cache_esparso import (
    CacheGeracao,
    gate_inferencia_cacheado,
)
from .roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from .runtime_condicional_v73 import (
    ConfiguracaoRuntimeCondicionalV73,
    FFNCombinatoriaEmpacotadaV73,
    ModeloGeradorEsparsoCondicionalV73,
)


@dataclass(frozen=True)
class ConfiguracaoRuntimeV73:
    """Seleciona o kernel sem alterar parâmetros ou checkpoint."""

    exigir_kernel_cuda: bool = False
    usar_kernel_em_lotes: bool = True
    limiar_tokens_lote: int = 512


class BlocoInferenciaVetorizadoV73(BlocoGeradorEsparso):
    """Vetoriza Q/K e atenção para lotes, sem mudar o caminho incremental."""

    def __init__(
        self,
        origem: BlocoGeradorEsparso,
        configuracao_v73: ConfiguracaoRuntimeV73,
    ) -> None:
        nn.Module.__init__(self)
        self.configuracao = origem.configuracao
        self.consulta = origem.consulta
        self.chave = origem.chave
        self.normalizacao_atencao = origem.normalizacao_atencao
        self.normalizacao_ffn = origem.normalizacao_ffn
        self.ffn = origem.ffn
        self.gate_atencao = origem.gate_atencao
        self.gate_ffn = origem.gate_ffn
        self.configuracao_v73 = configuracao_v73
        self._cache_q: tuple[int, Tensor] | None = None
        self._cache_k: tuple[int, Tensor] | None = None
        self.register_buffer(
            "mascara_causal_v73",
            torch.ones(
                (
                    self.configuracao.maximo_contexto,
                    self.configuracao.maximo_contexto,
                ),
                dtype=torch.bool,
            ).tril(),
            persistent=False,
        )

    @staticmethod
    def _matriz_densa_cacheada(
        camada,
        cache: tuple[int, Tensor] | None,
    ) -> tuple[Tensor, tuple[int, Tensor]]:
        versao = camada.valores._version
        if cache is None or cache[0] != versao:
            matriz = camada.valores.new_zeros(
                (camada.saidas, camada.entradas)
            )
            matriz[camada.indices[0], camada.indices[1]] = (
                camada.valores.detach()[camada.ordem_valores]
            )
            cache = (versao, matriz)
        return cache[1], cache

    def _atencao_lote(self, estados: Tensor) -> Tensor:
        normalizados = self.normalizacao_atencao(estados)
        pesos_q, self._cache_q = self._matriz_densa_cacheada(
            self.consulta,
            self._cache_q,
        )
        pesos_k, self._cache_k = self._matriz_densa_cacheada(
            self.chave,
            self._cache_k,
        )
        consultas = F.linear(
            normalizados,
            pesos_q,
            self.consulta.bias,
        )
        chaves = F.linear(
            normalizados,
            pesos_k,
            self.chave.bias,
        )
        comprimento = estados.shape[1]
        pontuacoes = torch.bmm(
            consultas,
            chaves.transpose(1, 2),
        ) / math.sqrt(self.configuracao.dimensao)
        pontuacoes.masked_fill_(
            ~self.mascara_causal_v73[
                :comprimento,
                :comprimento,
            ].unsqueeze(0),
            -torch.inf,
        )
        quantidade = min(self.configuracao.top_k_atencao, comprimento)
        valores, indices = pontuacoes.topk(
            quantidade,
            dim=-1,
            sorted=False,
        )
        pesos_topk = torch.softmax(
            valores.float(),
            dim=-1,
        ).to(pontuacoes.dtype)
        pesos = torch.zeros_like(pontuacoes)
        pesos.scatter_(2, indices, pesos_topk)
        return torch.bmm(pesos, normalizados)

    def forward(self, estados: Tensor) -> Tensor:
        if (
            not self.training
            and not torch.is_grad_enabled()
            and estados.is_cuda
            and estados.numel() // estados.shape[-1]
            >= self.configuracao_v73.limiar_tokens_lote
        ):
            contexto = self._atencao_lote(estados)
            estados = (
                estados
                + gate_inferencia_cacheado(self, "gate_atencao") * contexto
            )
            transformados = self.ffn(
                self.normalizacao_ffn(estados)
            )
            return (
                estados
                + gate_inferencia_cacheado(self, "gate_ffn") * transformados
            )
        return super().forward(estados)


class FFNCombinatoriaFundidaV73(FFNCombinatoriaEmpacotadaV73):
    """Substitui vários operadores PyTorch por um lançamento CUDA."""

    def __init__(
        self,
        origem: FFNCombinatoriaEmpacotadaV73,
        configuracao_runtime: ConfiguracaoRuntimeCondicionalV73,
        configuracao_v73: ConfiguracaoRuntimeV73,
    ) -> None:
        super().__init__(origem, configuracao_runtime)
        self.configuracao_v73 = configuracao_v73
        self._kernel_usado_ultima_execucao = False

    def _pode_usar_kernel(self, planos: Tensor) -> bool:
        return (
            not self.training
            and not torch.is_grad_enabled()
            and planos.is_cuda
            and planos.dtype == torch.float32
            and (self.configuracao_v73.usar_kernel_em_lotes or planos.shape[0] == 1)
        )

    def forward(self, estados: Tensor) -> Tensor:
        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        if not self._pode_usar_kernel(planos):
            self._kernel_usado_ultima_execucao = False
            return super().forward(estados)

        kernel = carregar_kernel_cuda_v73(
            obrigatorio=self.configuracao_v73.exigir_kernel_cuda
        )
        if kernel is None:
            self._kernel_usado_ultima_execucao = False
            return super().forward(estados)

        grupos, gates = self.roteador(planos)
        pesos_1, pesos_2 = self._bancos_empacotados()
        bias_1 = self.camada_1.bias.view(
            self.configuracao_roteamento.grupos,
            self.camada_1.tamanho_grupo,
        )
        resultado = kernel.ffn_roteada(
            planos.contiguous(),
            grupos.contiguous(),
            gates.contiguous(),
            pesos_1.contiguous(),
            bias_1.contiguous(),
            pesos_2.contiguous(),
            self.camada_2.bias.contiguous(),
        )
        self._kernel_usado_ultima_execucao = True
        return resultado.view(*forma)


class ModeloGeradorEsparsoV73(ModeloGeradorEsparsoCondicionalV73):
    """Modelo-base V7.3 com execução CUDA e fallback PyTorch."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorEsparso = ConfiguracaoGeradorEsparso(),
        semente: int = 20260728,
        configuracao_roteamento: ConfiguracaoRoteamentoV73 = (
            ConfiguracaoRoteamentoV73()
        ),
        configuracao_runtime: ConfiguracaoRuntimeCondicionalV73 = (
            ConfiguracaoRuntimeCondicionalV73()
        ),
        configuracao_v73: ConfiguracaoRuntimeV73 = (
            ConfiguracaoRuntimeV73()
        ),
    ) -> None:
        super().__init__(
            vocabulario,
            pad_id,
            configuracao,
            semente,
            configuracao_roteamento,
            configuracao_runtime,
        )
        self.configuracao_v73 = configuracao_v73
        if configuracao_v73.limiar_tokens_lote <= 0:
            raise ValueError("limiar_tokens_lote deve ser positivo")
        self._cache_qk: dict[
            int,
            tuple[int, Tensor, Tensor],
        ] = {}
        for indice, bloco in enumerate(self.blocos):
            bloco.ffn = FFNCombinatoriaFundidaV73(
                bloco.ffn,
                configuracao_runtime,
                configuracao_v73,
            )
            self.blocos[indice] = BlocoInferenciaVetorizadoV73(
                bloco,
                configuracao_v73,
            )

    def _valores_qk_empacotados(self, camada) -> tuple[Tensor, Tensor]:
        """Ordena o fan-in uma vez sem criar parâmetros persistentes."""

        chave = id(camada)
        versao = camada.valores._version
        cache = self._cache_qk.get(chave)
        if cache is None or cache[0] != versao:
            fan_in = camada.indices.shape[1] // camada.saidas
            if (
                camada.entradas != 128
                or camada.saidas != 128
                or fan_in != 32
            ):
                raise RuntimeError(
                    "kernel V7.3 exige Q/K 128x128 com fan-in 32"
                )
            indices = camada.indices[1].view(128, 32).contiguous()
            valores = camada.valores.detach().index_select(
                0,
                camada.ordem_valores,
            ).view(128, 32).contiguous()
            cache = (versao, indices, valores)
            self._cache_qk[chave] = cache
        return cache[1], cache[2]

    def _pode_avancar_fundido(self, novo_token: Tensor) -> bool:
        return (
            novo_token.is_cuda
            and self.embedding.weight.dtype == torch.float32
            and not self.training
            and not torch.is_grad_enabled()
            and self.configuracao.dimensao == 128
            and self.configuracao.maximo_contexto <= 640
            and self.embedding.num_embeddings <= 512
        )

    @torch.inference_mode()
    def avancar_cache_geracao(
        self,
        novo_token: Tensor,
        cache: CacheGeracao,
    ) -> tuple[Tensor, CacheGeracao]:
        """Funde o caminho crítico incremental em onze kernels por token."""

        if not self._pode_avancar_fundido(novo_token):
            return super().avancar_cache_geracao(novo_token, cache)
        if novo_token.ndim != 2 or novo_token.shape[1] != 1:
            raise ValueError("novo_token deve possuir shape [lote, 1]")
        if novo_token.shape[0] != cache.tokens.shape[0]:
            raise ValueError("lote do novo token diverge do cache")
        if novo_token.device != cache.tokens.device:
            raise ValueError("novo token e cache devem usar o mesmo dispositivo")
        if len(cache.camadas) != len(self.blocos):
            raise ValueError("quantidade de camadas do cache e invalida")
        if cache.comprimento >= cache.capacidade:
            return super().avancar_cache_geracao(novo_token, cache)

        kernel = carregar_kernel_cuda_v73(
            obrigatorio=self.configuracao_v73.exigir_kernel_cuda
        )
        if kernel is None:
            return super().avancar_cache_geracao(novo_token, cache)

        posicao = cache.comprimento
        novo_comprimento = posicao + 1
        cache.tokens[:, posicao:novo_comprimento].copy_(novo_token)
        estados = kernel.embedding_posicao(
            novo_token.reshape(-1).contiguous(),
            self.embedding.weight.contiguous(),
            self.posicoes[posicao].contiguous(),
            self.gate_posicao.contiguous(),
        )

        for bloco, cache_camada in zip(
            self.blocos,
            cache.camadas,
            strict=True,
        ):
            indices_q, valores_q = self._valores_qk_empacotados(
                bloco.consulta
            )
            indices_k, valores_k = self._valores_qk_empacotados(
                bloco.chave
            )
            normalizados, consulta, chave = kernel.normalizar_qk(
                estados.contiguous(),
                bloco.normalizacao_atencao.weight.contiguous(),
                bloco.normalizacao_atencao.bias.contiguous(),
                float(bloco.normalizacao_atencao.eps),
                indices_q,
                valores_q,
                bloco.consulta.bias.contiguous(),
                indices_k,
                valores_k,
                bloco.chave.bias.contiguous(),
            )
            cache_camada.valores[:, posicao].copy_(normalizados)
            cache_camada.chaves[:, posicao].copy_(chave)
            contexto = kernel.contexto_incremental(
                consulta.contiguous(),
                cache_camada.chaves,
                cache_camada.valores,
                novo_comprimento,
            )

            ffn = bloco.ffn
            pesos_1, pesos_2 = ffn._bancos_empacotados()
            bias_1 = ffn.camada_1.bias.view(16, 24)
            estados = kernel.residual_ffn_roteada(
                estados.contiguous(),
                contexto.contiguous(),
                bloco.gate_atencao.contiguous(),
                bloco.normalizacao_ffn.weight.contiguous(),
                bloco.normalizacao_ffn.bias.contiguous(),
                float(bloco.normalizacao_ffn.eps),
                ffn.roteador.projecao_contexto.contiguous(),
                ffn.roteador.codigos_grupos.contiguous(),
                ffn.roteador.bias.contiguous(),
                pesos_1.contiguous(),
                bias_1.contiguous(),
                pesos_2.contiguous(),
                ffn.camada_2.bias.contiguous(),
                bloco.gate_ffn.contiguous(),
            )

        cache.comprimento = novo_comprimento
        logits = kernel.saida_vocabulario(
            estados.contiguous(),
            self.normalizacao_saida.weight.contiguous(),
            self.normalizacao_saida.bias.contiguous(),
            float(self.normalizacao_saida.eps),
            self.embedding.weight.contiguous(),
            self.saida_bias.contiguous(),
        )
        return logits.unsqueeze(1), cache

    def auditoria(self) -> dict[str, int | float | bool | str | dict]:
        auditoria = super().auditoria()
        auditoria.update(
            {
                "modelo": "gerador-esparso-v73-kernel-cuda",
                "kernel_ffn_fundido": True,
                "forward_lote_vetorizado": True,
                "caminho_incremental_fundido": True,
                "lancamentos_kernel_por_token": 11,
                "kernel_compilado": all(
                    bloco.ffn._kernel_usado_ultima_execucao
                    for bloco in self.blocos
                ),
                "erro_kernel": erro_kernel_cuda_v73(),
                "grupos_inativos_executados": False,
                "microblocos_ativos_multiplicam_zeros_coo": True,
                "parametros_adicionais_kernel": 0,
                "checkpoint_base_v73": True,
                "elegivel_promocao": False,
            }
        )
        return auditoria
