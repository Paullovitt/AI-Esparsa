"""Núcleo de roteamento combinatório da V7.3.

O banco de pesos das duas projecoes FFN e particionado em microgrupos
balanceados. Um roteador causal aprendido escolhe uma combinacao Top-K por
token; somente os grupos escolhidos participam da passagem.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .camada_linear_esparsa import LinearBlocoEsparsa
from .modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    FFNGeradorEsparsa,
)
from .modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62


@dataclass(frozen=True)
class ConfiguracaoRoteamentoV73:
    """Hiperparametros do roteamento sem ampliar materialmente o modelo."""

    grupos: int = 16
    grupos_ativos: int = 4
    dimensao_latente: int = 4
    coeficiente_balanceamento: float = 0.01
    coeficiente_z: float = 0.0001

    def validar(self, configuracao: ConfiguracaoGeradorEsparso) -> None:
        if self.grupos <= 1:
            raise ValueError("grupos deve ser maior que um")
        if not 0 < self.grupos_ativos < self.grupos:
            raise ValueError("grupos_ativos deve ficar entre um e grupos")
        if self.dimensao_latente <= 0:
            raise ValueError("dimensao_latente deve ser positiva")
        if self.coeficiente_balanceamento < 0 or self.coeficiente_z < 0:
            raise ValueError("coeficientes auxiliares nao podem ser negativos")
        # A divisibilidade garante a mesma quantidade de arestas de cada
        # saida/entrada nos grupos, evitando favorecer um circuito pela
        # topologia sorteada.
        if configuracao.fan_in_ffn % self.grupos != 0:
            raise ValueError("fan_in_ffn deve ser divisivel por grupos")
        if configuracao.fan_out_ffn % self.grupos != 0:
            raise ValueError("fan_out_ffn deve ser divisivel por grupos")


class RoteadorCombinatorio(nn.Module):
    """Roteador baixo-rank causal com Top-K duro e gates diferenciaveis."""

    def __init__(
        self,
        dimensao: int,
        configuracao: ConfiguracaoRoteamentoV73,
        semente: int,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        gerador = torch.Generator().manual_seed(semente)
        escala_entrada = 1.0 / math.sqrt(dimensao)
        escala_saida = 1.0 / math.sqrt(configuracao.dimensao_latente)
        self.projecao_contexto = nn.Parameter(
            torch.randn(
                dimensao,
                configuracao.dimensao_latente,
                generator=gerador,
            )
            * escala_entrada
        )
        self.codigos_grupos = nn.Parameter(
            torch.randn(
                configuracao.dimensao_latente,
                configuracao.grupos,
                generator=gerador,
            )
            * escala_saida
        )
        self.bias = nn.Parameter(torch.zeros(configuracao.grupos))
        self.ultimas_mascaras: Tensor | None = None
        self.ultima_entropia = torch.tensor(0.0)
        self.ultimo_balanceamento = torch.tensor(0.0)
        self.ultimo_z = torch.tensor(0.0)

    def forward(
        self,
        estados: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if estados.ndim != 2:
            raise ValueError("roteador espera estados [tokens, dimensao]")
        latentes = torch.tanh(estados @ self.projecao_contexto)
        logits = latentes @ self.codigos_grupos + self.bias
        probabilidades = torch.softmax(logits.float(), dim=-1)
        valores, indices = logits.topk(
            self.configuracao.grupos_ativos,
            dim=-1,
            sorted=True,
        )
        # Media um preserva a escala relativa dos grupos escolhidos.
        gates = torch.softmax(valores.float(), dim=-1)
        gates = gates * self.configuracao.grupos_ativos
        gates = gates.to(estados.dtype)

        mascara = torch.zeros_like(probabilidades)
        mascara.scatter_(1, indices, 1.0)
        importancia = probabilidades.mean(dim=0)
        carga = mascara.mean(dim=0)
        self.ultimo_balanceamento = (
            self.configuracao.grupos
            * (importancia * carga.detach()).sum()
        )
        self.ultimo_z = torch.logsumexp(logits.float(), dim=-1).square().mean()
        self.ultima_entropia = -(
            importancia * importancia.clamp_min(1e-9).log()
        ).sum() / math.log(self.configuracao.grupos)

        # Um inteiro de 16 bits identifica inequivocamente a combinacao.
        potencias = (
            1 << torch.arange(
                self.configuracao.grupos,
                device=indices.device,
                dtype=torch.int64,
            )
        )
        self.ultimas_mascaras = potencias[indices].sum(dim=-1).detach()
        return indices, gates

    def perda_auxiliar(self) -> Tensor:
        """Combina balanceamento e z-loss sem ocultar seus componentes."""

        return (
            self.configuracao.coeficiente_balanceamento
            * self.ultimo_balanceamento
            + self.configuracao.coeficiente_z * self.ultimo_z
        )


class LinearCOORoteada(nn.Module):
    """Aplica somente microgrupos COO selecionados para cada token."""

    def __init__(
        self,
        origem: LinearBlocoEsparsa,
        configuracao: ConfiguracaoRoteamentoV73,
    ) -> None:
        super().__init__()
        self.entradas = origem.entradas
        self.saidas = origem.saidas
        self.configuracao = configuracao
        # Os Parameters sao transferidos, nao copiados: o banco possui
        # exatamente os mesmos valores treinaveis da camada COO de origem.
        self.valores = origem.valores
        self.bias = origem.bias
        self._versao_cache = -1
        self._matrizes_inferencia: list[Tensor] = []

        grupos_ordenados = origem.ordem_valores.remainder(
            configuracao.grupos
        )
        for grupo in range(configuracao.grupos):
            posicoes = torch.nonzero(
                grupos_ordenados.eq(grupo),
                as_tuple=False,
            ).flatten()
            indices = origem.indices[:, posicoes].contiguous()
            parametros = origem.ordem_valores[posicoes].contiguous()
            contagens = torch.bincount(
                indices[0],
                minlength=self.saidas,
            )
            self.register_buffer(
                f"indices_grupo_{grupo:02d}",
                indices,
                persistent=False,
            )
            self.register_buffer(
                f"parametros_grupo_{grupo:02d}",
                parametros,
                persistent=False,
            )
            self.register_buffer(
                f"ponteiros_grupo_{grupo:02d}",
                torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long),
                        contagens.cumsum(0),
                    )
                ).contiguous(),
                persistent=False,
            )

    def _limpar_cache(self) -> None:
        self._matrizes_inferencia = []
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

    def _matriz_grupo(self, grupo: int) -> Tensor:
        indices = getattr(self, f"indices_grupo_{grupo:02d}")
        parametros = getattr(self, f"parametros_grupo_{grupo:02d}")
        valores = self.valores[parametros]
        if self.training or torch.is_grad_enabled():
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                return torch.sparse_coo_tensor(
                    indices,
                    valores,
                    (self.saidas, self.entradas),
                    device=valores.device,
                    is_coalesced=True,
                    check_invariants=False,
                )

        versao = self.valores._version
        if (
            len(self._matrizes_inferencia) != self.configuracao.grupos
            or self._versao_cache != versao
        ):
            matrizes: list[Tensor] = []
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                for indice_grupo in range(self.configuracao.grupos):
                    colunas = getattr(
                        self,
                        f"indices_grupo_{indice_grupo:02d}",
                    )[1]
                    posicoes = getattr(
                        self,
                        f"parametros_grupo_{indice_grupo:02d}",
                    )
                    ponteiros = getattr(
                        self,
                        f"ponteiros_grupo_{indice_grupo:02d}",
                    )
                    matrizes.append(
                        torch.sparse_csr_tensor(
                            ponteiros,
                            colunas,
                            self.valores[posicoes].detach(),
                            size=(self.saidas, self.entradas),
                            device=self.valores.device,
                            check_invariants=False,
                        )
                    )
            self._matrizes_inferencia = matrizes
            self._versao_cache = versao
        return self._matrizes_inferencia[grupo]

    def forward(
        self,
        entrada: Tensor,
        grupos: Tensor,
        gates: Tensor,
    ) -> Tensor:
        if entrada.ndim != 2 or entrada.shape[1] != self.entradas:
            raise ValueError("shape da entrada incompativel")
        if grupos.shape != gates.shape or grupos.shape[0] != entrada.shape[0]:
            raise ValueError("rotas e entradas possuem shapes incompativeis")
        if grupos.shape[1] != self.configuracao.grupos_ativos:
            raise ValueError("quantidade de grupos ativos divergente")

        saida = entrada.new_zeros((entrada.shape[0], self.saidas))
        escala = math.sqrt(
            self.configuracao.grupos / self.configuracao.grupos_ativos
        )
        for grupo in range(self.configuracao.grupos):
            ocorrencias = torch.nonzero(
                grupos.eq(grupo),
                as_tuple=False,
            )
            if ocorrencias.numel() == 0:
                continue
            tokens = ocorrencias[:, 0]
            posicoes_gate = ocorrencias[:, 1]
            projetados = torch.sparse.mm(
                self._matriz_grupo(grupo),
                entrada.index_select(0, tokens).t(),
            ).t()
            pesos = gates[tokens, posicoes_gate].unsqueeze(1)
            saida.index_add_(
                0,
                tokens,
                projetados * pesos * escala,
            )
        return saida + self.bias

    @property
    def conexoes(self) -> int:
        return self.valores.numel()

    @property
    def conexoes_ativas_por_token(self) -> int:
        return (
            self.conexoes
            * self.configuracao.grupos_ativos
            // self.configuracao.grupos
        )


class BancoMicroblocosCOO(nn.Module):
    """Recorta uma COO em blocos de saidas ou entradas sem copiar pesos."""

    def __init__(
        self,
        origem: LinearBlocoEsparsa,
        configuracao: ConfiguracaoRoteamentoV73,
        *,
        agrupar_saidas: bool,
    ) -> None:
        super().__init__()
        self.entradas = origem.entradas
        self.saidas = origem.saidas
        self.configuracao = configuracao
        self.agrupar_saidas = agrupar_saidas
        self.valores = origem.valores
        self.bias = origem.bias
        self._versao_cache = -1
        self._matrizes_inferencia: list[Tensor] = []
        tamanho_eixo = self.saidas if agrupar_saidas else self.entradas
        if tamanho_eixo % configuracao.grupos != 0:
            raise ValueError("eixo roteado deve ser divisivel por grupos")
        self.tamanho_grupo = tamanho_eixo // configuracao.grupos

        for grupo in range(configuracao.grupos):
            inicio = grupo * self.tamanho_grupo
            fim = inicio + self.tamanho_grupo
            eixo = origem.indices[0] if agrupar_saidas else origem.indices[1]
            posicoes = torch.nonzero(
                eixo.ge(inicio) & eixo.lt(fim),
                as_tuple=False,
            ).flatten()
            indices = origem.indices[:, posicoes].clone()
            if agrupar_saidas:
                indices[0].sub_(inicio)
                forma = (self.tamanho_grupo, self.entradas)
            else:
                indices[1].sub_(inicio)
                forma = (self.saidas, self.tamanho_grupo)
            parametros = origem.ordem_valores[posicoes].contiguous()
            contagens = torch.bincount(
                indices[0],
                minlength=forma[0],
            )
            self.register_buffer(
                f"indices_bloco_{grupo:02d}",
                indices.contiguous(),
                persistent=False,
            )
            self.register_buffer(
                f"parametros_bloco_{grupo:02d}",
                parametros,
                persistent=False,
            )
            self.register_buffer(
                f"ponteiros_bloco_{grupo:02d}",
                torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long),
                        contagens.cumsum(0),
                    )
                ).contiguous(),
                persistent=False,
            )

    def _forma_grupo(self) -> tuple[int, int]:
        if self.agrupar_saidas:
            return self.tamanho_grupo, self.entradas
        return self.saidas, self.tamanho_grupo

    def _limpar_cache(self) -> None:
        self._matrizes_inferencia = []
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

    def _matriz_grupo(self, grupo: int) -> Tensor:
        indices = getattr(self, f"indices_bloco_{grupo:02d}")
        parametros = getattr(self, f"parametros_bloco_{grupo:02d}")
        valores = self.valores[parametros]
        forma = self._forma_grupo()
        if self.training or torch.is_grad_enabled():
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                return torch.sparse_coo_tensor(
                    indices,
                    valores,
                    forma,
                    device=valores.device,
                    is_coalesced=True,
                    check_invariants=False,
                )
        versao = self.valores._version
        if (
            len(self._matrizes_inferencia) != self.configuracao.grupos
            or self._versao_cache != versao
        ):
            matrizes: list[Tensor] = []
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sparse CSR tensor support is in beta state.*",
                    category=UserWarning,
                )
                for indice_grupo in range(self.configuracao.grupos):
                    colunas = getattr(
                        self,
                        f"indices_bloco_{indice_grupo:02d}",
                    )[1]
                    posicoes = getattr(
                        self,
                        f"parametros_bloco_{indice_grupo:02d}",
                    )
                    ponteiros = getattr(
                        self,
                        f"ponteiros_bloco_{indice_grupo:02d}",
                    )
                    matrizes.append(
                        torch.sparse_csr_tensor(
                            ponteiros,
                            colunas,
                            self.valores[posicoes].detach(),
                            size=forma,
                            device=self.valores.device,
                            check_invariants=False,
                        )
                    )
            self._matrizes_inferencia = matrizes
            self._versao_cache = versao
        return self._matrizes_inferencia[grupo]

    def projetar(self, grupo: int, entrada: Tensor) -> Tensor:
        if entrada.ndim != 2 or entrada.shape[1] != self._forma_grupo()[1]:
            raise ValueError("entrada incompativel com o microbloco")
        saida = torch.sparse.mm(
            self._matriz_grupo(grupo),
            entrada.t(),
        ).t()
        if self.agrupar_saidas:
            inicio = grupo * self.tamanho_grupo
            saida = saida + self.bias[inicio : inicio + self.tamanho_grupo]
        return saida

    @property
    def conexoes(self) -> int:
        return self.valores.numel()

    @property
    def conexoes_ativas_por_token(self) -> int:
        return (
            self.conexoes
            * self.configuracao.grupos_ativos
            // self.configuracao.grupos
        )


class FFNCombinatoriaV73(nn.Module):
    """FFN em que a mesma rota escolhe arestas nas duas projecoes."""

    def __init__(
        self,
        origem: FFNGeradorEsparsa,
        configuracao: ConfiguracaoRoteamentoV73,
        semente: int,
    ) -> None:
        super().__init__()
        self.configuracao_base = origem.configuracao
        self.configuracao_roteamento = configuracao
        self.roteador = RoteadorCombinatorio(
            origem.configuracao.dimensao,
            configuracao,
            semente,
        )
        # Cada grupo contém 24 unidades internas e todas as arestas COO de
        # entrada/saída dessas unidades. Não há FFNs independentes nem cópias
        # do banco: circuitos diferentes são uniões Top-K dos mesmos blocos.
        self.camada_1 = BancoMicroblocosCOO(
            origem.camada_1,
            configuracao,
            agrupar_saidas=True,
        )
        self.camada_2 = BancoMicroblocosCOO(
            origem.camada_2,
            configuracao,
            agrupar_saidas=False,
        )

    def forward(self, estados: Tensor) -> Tensor:
        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        grupos, gates = self.roteador(planos)
        quantidade_tokens = planos.shape[0]
        tamanho_grupo = self.camada_1.tamanho_grupo
        ativacoes = planos.new_zeros(
            (
                quantidade_tokens,
                self.configuracao_roteamento.grupos_ativos,
                tamanho_grupo,
            )
        )
        for grupo in range(self.configuracao_roteamento.grupos):
            ocorrencias = torch.nonzero(
                grupos.eq(grupo),
                as_tuple=False,
            )
            if ocorrencias.numel() == 0:
                continue
            tokens = ocorrencias[:, 0]
            posicoes = ocorrencias[:, 1]
            projetados = F.gelu(
                self.camada_1.projetar(
                    grupo,
                    planos.index_select(0, tokens),
                )
            )
            ativacoes.index_put_(
                (tokens, posicoes),
                projetados,
            )
        ativacoes_planas = ativacoes.flatten(1)
        quantidade_topk = min(
            self.configuracao_base.top_k_ffn,
            ativacoes_planas.shape[1],
        )
        valores, indices = ativacoes_planas.topk(
            quantidade_topk,
            dim=-1,
        )
        podadas = torch.zeros_like(ativacoes_planas)
        podadas.scatter_(1, indices, valores)
        podadas = podadas.view_as(ativacoes)

        saida = planos.new_zeros(
            (quantidade_tokens, self.configuracao_base.dimensao)
        )
        for grupo in range(self.configuracao_roteamento.grupos):
            ocorrencias = torch.nonzero(
                grupos.eq(grupo),
                as_tuple=False,
            )
            if ocorrencias.numel() == 0:
                continue
            tokens = ocorrencias[:, 0]
            posicoes = ocorrencias[:, 1]
            projetados = self.camada_2.projetar(
                grupo,
                podadas[tokens, posicoes],
            )
            pesos = gates[tokens, posicoes].unsqueeze(1)
            saida.index_add_(0, tokens, projetados * pesos)
        return (saida + self.camada_2.bias).view(*forma)


class ModeloGeradorEsparsoRoteadoV73(ModeloGeradorEsparsoV62):
    """Núcleo treinável da V7.3, preservando a V6.2 separadamente."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorEsparso = ConfiguracaoGeradorEsparso(),
        semente: int = 20260728,
        configuracao_roteamento: ConfiguracaoRoteamentoV73 = (
            ConfiguracaoRoteamentoV73()
        ),
    ) -> None:
        configuracao_roteamento.validar(configuracao)
        super().__init__(vocabulario, pad_id, configuracao, semente)
        self.configuracao_roteamento = configuracao_roteamento
        for indice, bloco in enumerate(self.blocos):
            bloco.ffn = FFNCombinatoriaV73(
                bloco.ffn,
                configuracao_roteamento,
                semente + 10_000 + indice,
            )

    def forward(self, tokens: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits, auxiliares = super().forward(tokens)
        roteadores = [
            bloco.ffn.roteador
            for bloco in self.blocos
            if isinstance(bloco.ffn, FFNCombinatoriaV73)
        ]
        auxiliares.update(
            {
                "perda_roteamento": torch.stack(
                    [roteador.perda_auxiliar() for roteador in roteadores]
                ).mean(),
                "balanceamento_roteamento": torch.stack(
                    [
                        roteador.ultimo_balanceamento
                        for roteador in roteadores
                    ]
                ).mean(),
                "z_roteamento": torch.stack(
                    [roteador.ultimo_z for roteador in roteadores]
                ).mean(),
                "entropia_roteamento": torch.stack(
                    [
                        roteador.ultima_entropia
                        for roteador in roteadores
                    ]
                ).mean(),
            }
        )
        return logits, auxiliares

    def mascaras_ultima_execucao(self) -> tuple[Tensor, ...]:
        """Expõe bitmasks para auditoria sem persistir dados no checkpoint."""

        mascaras = []
        for bloco in self.blocos:
            roteador = bloco.ffn.roteador
            if roteador.ultimas_mascaras is None:
                raise RuntimeError("o modelo ainda nao executou um forward")
            mascaras.append(roteador.ultimas_mascaras)
        return tuple(mascaras)

    def auditoria(self) -> dict[str, int | float | bool | str | dict]:
        auditoria = super().auditoria()
        conexoes_ativas_ffn = sum(
            bloco.ffn.camada_1.conexoes_ativas_por_token
            + bloco.ffn.camada_2.conexoes_ativas_por_token
            for bloco in self.blocos
        )
        conexoes_qk = int(auditoria["conexoes_qk"])
        parametros_roteador = sum(
            parametro.numel()
            for bloco in self.blocos
            for parametro in bloco.ffn.roteador.parameters()
        )
        auditoria.update(
            {
                "modelo": "gerador-esparso-v73-roteamento-combinatorio",
                "checkpoint_oficial": False,
                "arquitetura_pesos_identica_oficial": False,
                "roteamento_combinatorio": True,
                "granularidade_roteamento": "token",
                "regiao_roteada": "ffn_coo",
                "tipo_microgrupo": (
                    "blocos_de_24_unidades_com_arestas_coo_compartilhadas"
                ),
                "ffns_independentes_por_grupo": False,
                "configuracao_roteamento": asdict(
                    self.configuracao_roteamento
                ),
                "parametros_roteador": parametros_roteador,
                "conexoes_ffn_ativas_por_token": conexoes_ativas_ffn,
                "conexoes_nucleo_ativas_por_token": (
                    conexoes_qk + conexoes_ativas_ffn
                ),
                "combinacoes_teoricas_por_camada": math.comb(
                    self.configuracao_roteamento.grupos,
                    self.configuracao_roteamento.grupos_ativos,
                ),
                "execucao_somente_grupos_selecionados": True,
            }
        )
        return auditoria
