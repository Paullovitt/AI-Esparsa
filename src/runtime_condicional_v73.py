"""Runtime condicional de referência da V7.3.

Este módulo contém o fallback PyTorch usado pela V7.3 quando a extensão CUDA
não está disponível. Ele executa somente as arestas COO selecionadas; chamadas
com gradiente continuam no caminho COO de referência.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from .roteamento_combinatorio_v73 import (
    ConfiguracaoRoteamentoV73,
    FFNCombinatoriaV73,
    ModeloGeradorEsparsoRoteadoV73,
)


@dataclass(frozen=True)
class ConfiguracaoRuntimeCondicionalV73:
    """Controla memoria temporaria sem alterar a matematica do modelo."""

    tokens_por_bloco: int = 512
    implementacao: str = "arestas_regulares"
    limiar_tokens_agrupados: int = 512

    def validar(self) -> None:
        if self.tokens_por_bloco <= 0:
            raise ValueError("tokens_por_bloco deve ser positivo")
        if self.limiar_tokens_agrupados <= 0:
            raise ValueError("limiar_tokens_agrupados deve ser positivo")
        if self.implementacao not in {
            "arestas_regulares",
            "microblocos_densos",
            "banco_completo_denso",
            "hibrido_agrupado",
        }:
            raise ValueError("implementacao condicional V7.3 desconhecida")


class FFNCombinatoriaEmpacotadaV73(FFNCombinatoriaV73):
    """Executa as rotas V7.3 por lotes, sem varrer grupos inativos."""

    def __init__(
        self,
        origem: FFNCombinatoriaV73,
        configuracao_runtime: ConfiguracaoRuntimeCondicionalV73,
    ) -> None:
        # Os submodulos sao transferidos para preservar nomes e state_dict.
        nn.Module.__init__(self)
        configuracao_runtime.validar()
        self.configuracao_base = origem.configuracao_base
        self.configuracao_roteamento = origem.configuracao_roteamento
        self.roteador = origem.roteador
        self.camada_1 = origem.camada_1
        self.camada_2 = origem.camada_2
        self.configuracao_runtime = configuracao_runtime
        self._pesos_1_empacotados: Tensor | None = None
        self._pesos_2_empacotados: Tensor | None = None
        self._versoes_cache: tuple[int, int] = (-1, -1)
        self._registrar_topologia_regular()

    def _registrar_topologia_regular(self) -> None:
        """Empacota fan-in/fan-out fixos sem persistir copias no checkpoint."""

        grupos = self.configuracao_roteamento.grupos
        tamanho_grupo = self.camada_1.tamanho_grupo
        entradas_1: list[Tensor] = []
        parametros_1: list[Tensor] = []
        saidas_2: list[Tensor] = []
        parametros_2: list[Tensor] = []
        for grupo in range(grupos):
            indices_1 = getattr(
                self.camada_1,
                f"indices_bloco_{grupo:02d}",
            )
            posicoes_1 = getattr(
                self.camada_1,
                f"parametros_bloco_{grupo:02d}",
            )
            contagens_1 = torch.bincount(
                indices_1[0],
                minlength=tamanho_grupo,
            )
            if not torch.equal(contagens_1, contagens_1[:1].expand_as(contagens_1)):
                raise RuntimeError("fan-in irregular no microgrupo V7.3")
            ordem_1 = torch.argsort(
                indices_1[0] * self.camada_1.entradas + indices_1[1],
                stable=True,
            )
            entradas_1.append(
                indices_1[1, ordem_1].view(tamanho_grupo, -1)
            )
            parametros_1.append(
                posicoes_1[ordem_1].view(tamanho_grupo, -1)
            )

            indices_2 = getattr(
                self.camada_2,
                f"indices_bloco_{grupo:02d}",
            )
            posicoes_2 = getattr(
                self.camada_2,
                f"parametros_bloco_{grupo:02d}",
            )
            contagens_2 = torch.bincount(
                indices_2[1],
                minlength=tamanho_grupo,
            )
            if not torch.equal(contagens_2, contagens_2[:1].expand_as(contagens_2)):
                raise RuntimeError("fan-out irregular no microgrupo V7.3")
            ordem_2 = torch.argsort(
                indices_2[1] * self.camada_2.saidas + indices_2[0],
                stable=True,
            )
            saidas_2.append(indices_2[0, ordem_2].view(tamanho_grupo, -1))
            parametros_2.append(
                posicoes_2[ordem_2].view(tamanho_grupo, -1)
            )

        self.register_buffer(
            "indices_entrada_1",
            torch.stack(entradas_1),
            persistent=False,
        )
        self.register_buffer(
            "parametros_1",
            torch.stack(parametros_1),
            persistent=False,
        )
        self.register_buffer(
            "indices_saida_2",
            torch.stack(saidas_2),
            persistent=False,
        )
        self.register_buffer(
            "parametros_2",
            torch.stack(parametros_2),
            persistent=False,
        )

    def _limpar_cache_empacotado(self) -> None:
        self._pesos_1_empacotados = None
        self._pesos_2_empacotados = None
        self._versoes_cache = (-1, -1)

    def _apply(self, func, recurse: bool = True):
        self._limpar_cache_empacotado()
        resultado = super()._apply(func, recurse)
        self._limpar_cache_empacotado()
        return resultado

    def train(self, mode: bool = True):
        if mode:
            self._limpar_cache_empacotado()
        return super().train(mode)

    def _materializar_banco(
        self,
        camada,
        *,
        primeira_projecao: bool,
    ) -> Tensor:
        """Reconstrui microblocos sem registrar pesos densos no checkpoint."""

        grupos = self.configuracao_roteamento.grupos
        forma = camada._forma_grupo()
        banco = camada.valores.new_zeros((grupos, *forma))
        for grupo in range(grupos):
            indices = getattr(camada, f"indices_bloco_{grupo:02d}")
            posicoes = getattr(camada, f"parametros_bloco_{grupo:02d}")
            banco[grupo, indices[0], indices[1]] = (
                camada.valores[posicoes].detach()
            )
        # A flag torna explicita a orientacao esperada por cada bmm.
        if primeira_projecao and forma[0] != camada.tamanho_grupo:
            raise RuntimeError("orientacao inesperada na primeira projecao")
        if not primeira_projecao and forma[1] != camada.tamanho_grupo:
            raise RuntimeError("orientacao inesperada na segunda projecao")
        return banco

    def _bancos_empacotados(self) -> tuple[Tensor, Tensor]:
        versoes = (
            self.camada_1.valores._version,
            self.camada_2.valores._version,
        )
        if (
            self._pesos_1_empacotados is None
            or self._pesos_2_empacotados is None
            or self._versoes_cache != versoes
        ):
            self._pesos_1_empacotados = self._materializar_banco(
                self.camada_1,
                primeira_projecao=True,
            )
            self._pesos_2_empacotados = self._materializar_banco(
                self.camada_2,
                primeira_projecao=False,
            )
            self._versoes_cache = versoes
        return self._pesos_1_empacotados, self._pesos_2_empacotados

    def _projetar_bloco(
        self,
        planos: Tensor,
        grupos: Tensor,
        gates: Tensor,
        pesos_1: Tensor,
        pesos_2: Tensor,
    ) -> Tensor:
        quantidade = planos.shape[0]
        ativos = self.configuracao_roteamento.grupos_ativos
        tamanho_grupo = self.camada_1.tamanho_grupo
        rotas_planas = grupos.reshape(-1)

        banco_1 = pesos_1.index_select(0, rotas_planas).view(
            quantidade,
            ativos,
            tamanho_grupo,
            self.configuracao_base.dimensao,
        )
        ativacoes = torch.matmul(
            banco_1,
            planos[:, None, :, None],
        ).squeeze(-1)
        vies_1 = self.camada_1.bias.view(
            self.configuracao_roteamento.grupos,
            tamanho_grupo,
        )
        ativacoes = F.gelu(
            ativacoes
            + vies_1.index_select(0, rotas_planas).view_as(ativacoes)
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

        banco_2 = pesos_2.index_select(0, rotas_planas).view(
            quantidade,
            ativos,
            self.configuracao_base.dimensao,
            tamanho_grupo,
        )
        projetadas = torch.matmul(
            banco_2,
            podadas.unsqueeze(-1),
        ).squeeze(-1)
        return (
            (projetadas * gates.unsqueeze(-1)).sum(dim=1)
            + self.camada_2.bias
        )

    def _projetar_bloco_arestas(
        self,
        planos: Tensor,
        grupos: Tensor,
        gates: Tensor,
    ) -> Tensor:
        """Executa apenas arestas existentes em tabelas fan-in/fan-out."""

        quantidade = planos.shape[0]
        ativos = self.configuracao_roteamento.grupos_ativos
        tamanho_grupo = self.camada_1.tamanho_grupo
        rotas_planas = grupos.reshape(-1)

        indices_1 = self.indices_entrada_1.index_select(
            0,
            rotas_planas,
        ).view(quantidade, ativos, tamanho_grupo, -1)
        pesos_1 = self.camada_1.valores[
            self.parametros_1.index_select(0, rotas_planas)
        ].view_as(indices_1)
        entradas = planos[:, None, None, :].expand(
            -1,
            ativos,
            tamanho_grupo,
            -1,
        )
        ativacoes = (
            torch.gather(entradas, -1, indices_1) * pesos_1
        ).sum(dim=-1)
        vies_1 = self.camada_1.bias.view(
            self.configuracao_roteamento.grupos,
            tamanho_grupo,
        )
        ativacoes = F.gelu(
            ativacoes
            + vies_1.index_select(0, rotas_planas).view_as(ativacoes)
        )

        ativacoes_planas = ativacoes.flatten(1)
        quantidade_topk = min(
            self.configuracao_base.top_k_ffn,
            ativacoes_planas.shape[1],
        )
        valores, indices_topk = ativacoes_planas.topk(
            quantidade_topk,
            dim=-1,
        )
        podadas = torch.zeros_like(ativacoes_planas)
        podadas.scatter_(1, indices_topk, valores)
        podadas = podadas.view_as(ativacoes)

        indices_2 = self.indices_saida_2.index_select(
            0,
            rotas_planas,
        ).view(quantidade, ativos, tamanho_grupo, -1)
        pesos_2 = self.camada_2.valores[
            self.parametros_2.index_select(0, rotas_planas)
        ].view_as(indices_2)
        contribuicoes = (
            podadas.unsqueeze(-1)
            * pesos_2
            * gates[:, :, None, None]
        ).flatten(2)
        saida_por_grupo = planos.new_zeros(
            (
                quantidade,
                ativos,
                self.configuracao_base.dimensao,
            )
        )
        saida_por_grupo.scatter_add_(
            2,
            indices_2.flatten(2),
            contribuicoes,
        )
        return saida_por_grupo.sum(dim=1) + self.camada_2.bias

    def _projetar_bloco_banco_completo(
        self,
        planos: Tensor,
        grupos: Tensor,
        gates: Tensor,
        pesos_1: Tensor,
        pesos_2: Tensor,
    ) -> Tensor:
        """Controle denso: preserva pesos/rotas, mas calcula os 16 grupos."""

        grupos_totais = self.configuracao_roteamento.grupos
        tamanho_grupo = self.camada_1.tamanho_grupo
        pesos_1_completos = pesos_1.reshape(
            grupos_totais * tamanho_grupo,
            self.configuracao_base.dimensao,
        )
        pesos_2_completos = pesos_2.permute(1, 0, 2).reshape(
            self.configuracao_base.dimensao,
            grupos_totais * tamanho_grupo,
        )
        ativacoes = F.gelu(
            F.linear(planos, pesos_1_completos, self.camada_1.bias)
        )
        gates_grupos = planos.new_zeros((planos.shape[0], grupos_totais))
        gates_grupos.scatter_(1, grupos, gates)
        gates_unidades = gates_grupos.repeat_interleave(
            tamanho_grupo,
            dim=1,
        )
        scores = ativacoes.masked_fill(gates_unidades.eq(0), -torch.inf)
        quantidade_topk = min(
            self.configuracao_base.top_k_ffn,
            self.configuracao_roteamento.grupos_ativos * tamanho_grupo,
        )
        _, indices_topk = scores.topk(quantidade_topk, dim=-1)
        podadas = torch.zeros_like(ativacoes)
        podadas.scatter_(
            1,
            indices_topk,
            torch.gather(ativacoes, 1, indices_topk),
        )
        return F.linear(
            podadas * gates_unidades,
            pesos_2_completos,
            self.camada_2.bias,
        )

    def _projetar_microblocos_agrupados(
        self,
        planos: Tensor,
        grupos: Tensor,
        gates: Tensor,
        pesos_1: Tensor,
        pesos_2: Tensor,
    ) -> Tensor:
        """Agrupa tokens por rota e executa somente quatro microblocos."""

        quantidade = planos.shape[0]
        ativos = self.configuracao_roteamento.grupos_ativos
        tamanho_grupo = self.camada_1.tamanho_grupo
        ativacoes = planos.new_zeros(
            (quantidade, ativos, tamanho_grupo)
        )
        vies_1 = self.camada_1.bias.view(
            self.configuracao_roteamento.grupos,
            tamanho_grupo,
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
                F.linear(
                    planos.index_select(0, tokens),
                    pesos_1[grupo],
                    vies_1[grupo],
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
        valores, indices_topk = ativacoes_planas.topk(
            quantidade_topk,
            dim=-1,
        )
        podadas = torch.zeros_like(ativacoes_planas)
        podadas.scatter_(1, indices_topk, valores)
        podadas = podadas.view_as(ativacoes)

        saida = planos.new_zeros(
            (quantidade, self.configuracao_base.dimensao)
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
            projetados = F.linear(
                podadas[tokens, posicoes],
                pesos_2[grupo],
            )
            saida.index_add_(
                0,
                tokens,
                projetados * gates[tokens, posicoes].unsqueeze(1),
            )
        return saida + self.camada_2.bias

    def forward(self, estados: Tensor) -> Tensor:
        # Todos os atalhos V7.3 são exclusivos de inferência. O backward
        # empacotado chegou a ser medido, mas sua expansao de indices elevou
        # muito a VRAM; manter a referencia COO evita um falso ganho de treino.
        if self.training or torch.is_grad_enabled():
            return super().forward(estados)

        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        grupos, gates = self.roteador(planos)
        pesos_1: Tensor | None = None
        pesos_2: Tensor | None = None
        if self.configuracao_runtime.implementacao in {
            "microblocos_densos",
            "banco_completo_denso",
            "hibrido_agrupado",
        }:
            pesos_1, pesos_2 = self._bancos_empacotados()
        if (
            self.configuracao_runtime.implementacao == "hibrido_agrupado"
            and planos.shape[0]
            >= self.configuracao_runtime.limiar_tokens_agrupados
        ):
            if pesos_1 is None or pesos_2 is None:
                raise RuntimeError("cache agrupado V7.3 ausente")
            return self._projetar_microblocos_agrupados(
                planos,
                grupos,
                gates,
                pesos_1,
                pesos_2,
            ).view(*forma)
        saidas = []
        tamanho = self.configuracao_runtime.tokens_por_bloco
        for inicio in range(0, planos.shape[0], tamanho):
            fim = min(inicio + tamanho, planos.shape[0])
            if self.configuracao_runtime.implementacao in {
                "arestas_regulares",
                "hibrido_agrupado",
            }:
                saida = self._projetar_bloco_arestas(
                    planos[inicio:fim],
                    grupos[inicio:fim],
                    gates[inicio:fim],
                )
            elif self.configuracao_runtime.implementacao == "microblocos_densos":
                if pesos_1 is None or pesos_2 is None:
                    raise RuntimeError("cache denso V7.3 ausente")
                saida = self._projetar_bloco(
                    planos[inicio:fim],
                    grupos[inicio:fim],
                    gates[inicio:fim],
                    pesos_1,
                    pesos_2,
                )
            else:
                if pesos_1 is None or pesos_2 is None:
                    raise RuntimeError("cache denso V7.3 ausente")
                saida = self._projetar_bloco_banco_completo(
                    planos[inicio:fim],
                    grupos[inicio:fim],
                    gates[inicio:fim],
                    pesos_1,
                    pesos_2,
                )
            saidas.append(saida)
        return torch.cat(saidas, dim=0).view(*forma)


class ModeloGeradorEsparsoCondicionalV73(ModeloGeradorEsparsoRoteadoV73):
    """Fallback condicional que preserva os pesos da V7.3."""

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
    ) -> None:
        super().__init__(
            vocabulario,
            pad_id,
            configuracao,
            semente,
            configuracao_roteamento,
        )
        configuracao_runtime.validar()
        self.configuracao_runtime = configuracao_runtime
        for bloco in self.blocos:
            bloco.ffn = FFNCombinatoriaEmpacotadaV73(
                bloco.ffn,
                configuracao_runtime,
            )

    def auditoria(self) -> dict[str, int | float | bool | str | dict]:
        auditoria = super().auditoria()
        auditoria.update(
            {
                "modelo": "gerador-esparso-v73-runtime-condicional",
                "runtime_empacotado": True,
                "checkpoint_v73_reutilizado": True,
                "parametros_adicionais_runtime": 0,
                "cache_denso_microgrupos_persistente": False,
                "implementacao_empacotada": (
                    self.configuracao_runtime.implementacao
                ),
                "multiplicacoes_internas_incluem_zeros_coo": (
                    self.configuracao_runtime.implementacao
                    in {
                        "microblocos_densos",
                        "banco_completo_denso",
                        "hibrido_agrupado",
                    }
                ),
                "execucao_real_somente_arestas_coo": (
                    self.configuracao_runtime.implementacao
                    == "arestas_regulares"
                ),
                "executa_grupos_inativos_fisicamente": (
                    self.configuracao_runtime.implementacao
                    == "banco_completo_denso"
                ),
                "runtime_hibrido_por_tamanho_lote": (
                    self.configuracao_runtime.implementacao
                    == "hibrido_agrupado"
                ),
                "limiar_tokens_agrupados": (
                    self.configuracao_runtime.limiar_tokens_agrupados
                ),
                "grupos_logicamente_executados_por_token": (
                    self.configuracao_roteamento.grupos_ativos
                ),
                "tokens_por_bloco_empacotado": (
                    self.configuracao_runtime.tokens_por_bloco
                ),
                "runtime_exclusivo_inferencia": True,
                "treino_mantem_coo_referencia": True,
                "elegivel_promocao": False,
            }
        )
        return auditoria
