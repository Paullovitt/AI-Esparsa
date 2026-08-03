"""Executor vetorizado de treinamento da V7.3.

A implementação de referência despacha microgrupos separadamente. Este módulo
fornece backends de treino que vetorizam essas operações.
As mascaras do roteador continuam anulando ativacoes e gradientes inativos.

O banco continua contendo somente os pesos COO treináveis da V7.3. Alguns
backends materializam zeros temporarios, o que e declarado pela auditoria.
Inferencia preserva os microgrupos condicionais e a atencao causal em blocos.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modelo_gerador_esparso import (
    BlocoGeradorEsparso,
    ConfiguracaoGeradorEsparso,
)
from .roteamento_combinatorio_v73 import (
    BancoMicroblocosCOO,
    ConfiguracaoRoteamentoV73,
    FFNCombinatoriaV73,
    ModeloGeradorEsparsoRoteadoV73,
)


@dataclass(frozen=True)
class ConfiguracaoTreinoV73:
    """Configura o executor de treino sem alterar pesos ou roteamento."""

    implementacao: str = "coo_agregado"
    atencao: str = "referencia_blocos"
    projecoes_qk: str = "coo"
    bloco_consultas_treino: int = 128

    def validar(self) -> None:
        if self.implementacao not in {
            "coo_agregado",
            "microblocos_densos",
            "banco_denso_completo",
        }:
            raise ValueError("implementacao de treino V7.3 desconhecida")
        if self.atencao not in {
            "referencia_blocos",
            "matriz_densa_topk",
            "matriz_blocos_topk",
        }:
            raise ValueError("implementacao de atencao V7.3 desconhecida")
        if self.projecoes_qk not in {
            "coo",
            "densas_temporarias",
        }:
            raise ValueError("implementacao Q/K de treino V7.3 desconhecida")
        if self.bloco_consultas_treino <= 0:
            raise ValueError("bloco_consultas_treino deve ser positivo")


class BlocoTreinoVetorizadoV73(BlocoGeradorEsparso):
    """Preserva Top-K causal, com controle denso exclusivo de treino."""

    def __init__(
        self,
        origem: BlocoGeradorEsparso,
        configuracao_treino: ConfiguracaoTreinoV73,
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
        self.configuracao_treino_v73 = configuracao_treino

    def _projetar_treino(self, camada, estados: Tensor) -> Tensor:
        """Permite AMP mantendo treináveis somente os valores COO existentes."""

        if self.configuracao_treino_v73.projecoes_qk == "coo":
            return self._projetar(camada, estados)
        pesos = camada.valores.new_zeros(
            (camada.saidas, camada.entradas)
        )
        pesos[camada.indices[0], camada.indices[1]] = camada.valores[
            camada.ordem_valores
        ]
        forma = estados.shape
        return F.linear(
            estados.reshape(-1, forma[-1]),
            pesos,
            camada.bias,
        ).view(*forma)

    def _atencao_matriz_densa_topk(self, estados: Tensor) -> Tensor:
        """Vetoriza scores e agregacao, mantendo mascara causal e Top-32."""

        normalizados = self.normalizacao_atencao(estados)
        consultas = self._projetar_treino(self.consulta, normalizados)
        chaves = self._projetar_treino(self.chave, normalizados)
        comprimento = estados.shape[1]
        quantidade = min(self.configuracao.top_k_atencao, comprimento)
        pontuacoes = torch.bmm(
            consultas,
            chaves.transpose(1, 2),
        ) / math.sqrt(self.configuracao.dimensao)
        mascara_causal = torch.ones(
            (comprimento, comprimento),
            dtype=torch.bool,
            device=estados.device,
        ).tril()
        pontuacoes.masked_fill_(
            ~mascara_causal.unsqueeze(0),
            -torch.inf,
        )
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

    def _atencao_matriz_blocos_topk(self, estados: Tensor) -> Tensor:
        """Evita a matriz global, preservando exatamente o Top-K causal."""

        normalizados = self.normalizacao_atencao(estados)
        consultas = self._projetar_treino(self.consulta, normalizados)
        chaves = self._projetar_treino(self.chave, normalizados)
        comprimento = estados.shape[1]
        quantidade = min(self.configuracao.top_k_atencao, comprimento)
        chaves_transpostas = chaves.transpose(1, 2)
        posicoes_chave = torch.arange(
            comprimento,
            device=estados.device,
        )
        contextos: list[Tensor] = []
        tamanho_bloco = self.configuracao_treino_v73.bloco_consultas_treino
        for inicio in range(0, comprimento, tamanho_bloco):
            fim = min(inicio + tamanho_bloco, comprimento)
            pontuacoes = torch.bmm(
                consultas[:, inicio:fim],
                chaves_transpostas,
            ) / math.sqrt(self.configuracao.dimensao)
            posicoes_consulta = torch.arange(
                inicio,
                fim,
                device=estados.device,
            ).unsqueeze(1)
            mascara = (
                posicoes_chave.unsqueeze(0) <= posicoes_consulta
            )
            pontuacoes.masked_fill_(
                ~mascara.unsqueeze(0),
                -torch.inf,
            )
            valores, indices = pontuacoes.topk(
                quantidade,
                dim=-1,
                sorted=False,
            )
            pesos = torch.softmax(
                valores.float(),
                dim=-1,
            ).to(pontuacoes.dtype)
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

    def _atencao_top_k(self, estados: Tensor) -> Tensor:
        if (
            self.configuracao_treino_v73.atencao
            == "matriz_densa_topk"
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._atencao_matriz_densa_topk(estados)
        if (
            self.configuracao_treino_v73.atencao
            == "matriz_blocos_topk"
            and self.training
            and torch.is_grad_enabled()
        ):
            return self._atencao_matriz_blocos_topk(estados)
        return super()._atencao_top_k(estados)


class FFNCombinatoriaTreinoAgregadoV73(FFNCombinatoriaV73):
    """Calcula o banco COO em duas chamadas e mascara grupos inativos."""

    def __init__(
        self,
        origem: FFNCombinatoriaV73,
        configuracao_treino: ConfiguracaoTreinoV73,
    ) -> None:
        # Transfere os mesmos Parameters para manter as chaves do state_dict.
        nn.Module.__init__(self)
        configuracao_treino.validar()
        self.configuracao_base = origem.configuracao_base
        self.configuracao_roteamento = origem.configuracao_roteamento
        self.roteador = origem.roteador
        self.camada_1 = origem.camada_1
        self.camada_2 = origem.camada_2
        self.configuracao_treino = configuracao_treino
        self._registrar_banco_completo(
            self.camada_1,
            prefixo="camada_1",
        )
        self._registrar_banco_completo(
            self.camada_2,
            prefixo="camada_2",
        )

    def _registrar_banco_completo(
        self,
        camada: BancoMicroblocosCOO,
        *,
        prefixo: str,
    ) -> None:
        """Reconstrui indices globais uma vez, sem persistir uma nova topologia."""

        indices_globais: list[Tensor] = []
        parametros_globais: list[Tensor] = []
        for grupo in range(self.configuracao_roteamento.grupos):
            indices = getattr(
                camada,
                f"indices_bloco_{grupo:02d}",
            ).clone()
            deslocamento = grupo * camada.tamanho_grupo
            if camada.agrupar_saidas:
                indices[0].add_(deslocamento)
            else:
                indices[1].add_(deslocamento)
            indices_globais.append(indices)
            parametros_globais.append(
                getattr(
                    camada,
                    f"parametros_bloco_{grupo:02d}",
                )
            )

        indices = torch.cat(indices_globais, dim=1)
        parametros = torch.cat(parametros_globais)
        chaves = indices[0] * camada.entradas + indices[1]
        ordem = torch.argsort(chaves, stable=True)
        indices = indices[:, ordem].contiguous()
        parametros = parametros[ordem].contiguous()
        if torch.unique(chaves).numel() != chaves.numel():
            raise RuntimeError("topologia V7.3 contem arestas duplicadas")
        self.register_buffer(
            f"indices_{prefixo}_completa",
            indices,
            persistent=False,
        )
        self.register_buffer(
            f"parametros_{prefixo}_completa",
            parametros,
            persistent=False,
        )

    def _matriz_completa(
        self,
        camada: BancoMicroblocosCOO,
        *,
        prefixo: str,
    ) -> Tensor:
        indices = getattr(self, f"indices_{prefixo}_completa")
        parametros = getattr(self, f"parametros_{prefixo}_completa")
        valores = camada.valores[parametros]
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                indices,
                valores,
                (camada.saidas, camada.entradas),
                device=valores.device,
                is_coalesced=True,
                check_invariants=False,
            )

    @staticmethod
    def _materializar_pesos(
        camada: BancoMicroblocosCOO,
        indices: Tensor,
        parametros: Tensor,
        forma: tuple[int, int],
    ) -> Tensor:
        """Espalha somente Parameters COO em uma matriz temporaria."""

        pesos = camada.valores.new_zeros(forma)
        pesos[indices[0], indices[1]] = camada.valores[parametros]
        return pesos

    def _banco_denso_completo(
        self,
        camada: BancoMicroblocosCOO,
        *,
        prefixo: str,
    ) -> Tensor:
        return self._materializar_pesos(
            camada,
            getattr(self, f"indices_{prefixo}_completa"),
            getattr(self, f"parametros_{prefixo}_completa"),
            (camada.saidas, camada.entradas),
        )

    def _selecionar_topk_ativo(
        self,
        ativacoes: Tensor,
        grupos: Tensor,
    ) -> Tensor:
        grupos_totais = self.configuracao_roteamento.grupos
        tamanho_grupo = self.camada_1.tamanho_grupo
        mascara_grupos = torch.zeros(
            (ativacoes.shape[0], grupos_totais),
            dtype=torch.bool,
            device=ativacoes.device,
        )
        mascara_grupos.scatter_(1, grupos, True)
        mascara_unidades = mascara_grupos.repeat_interleave(
            tamanho_grupo,
            dim=1,
        )
        scores = ativacoes.masked_fill(~mascara_unidades, -torch.inf)
        quantidade_topk = min(
            self.configuracao_base.top_k_ffn,
            self.configuracao_roteamento.grupos_ativos * tamanho_grupo,
        )
        _, indices_topk = scores.topk(
            quantidade_topk,
            dim=-1,
        )
        podadas = torch.zeros_like(ativacoes)
        podadas.scatter_(
            1,
            indices_topk,
            torch.gather(ativacoes, 1, indices_topk),
        )
        return podadas

    def _gates_por_unidade(
        self,
        planos: Tensor,
        grupos: Tensor,
        gates: Tensor,
    ) -> Tensor:
        gates_grupos = planos.new_zeros(
            (
                planos.shape[0],
                self.configuracao_roteamento.grupos,
            )
        )
        gates_grupos.scatter_(1, grupos, gates)
        return gates_grupos.repeat_interleave(
            self.camada_1.tamanho_grupo,
            dim=1,
        )

    def _forward_treino_agregado(self, estados: Tensor) -> Tensor:
        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        grupos, gates = self.roteador(planos)
        # O cuSPARSE COO nao oferece addmm BF16 em todas as GPUs. Desabilitar
        # autocast apenas nas duas multiplicacoes preserva o backend e evita
        # falha quando o restante do treino usa BF16.
        with torch.autocast(device_type=planos.device.type, enabled=False):
            planos_esparsos = planos.to(self.camada_1.valores.dtype)
            matriz_1 = self._matriz_completa(
                self.camada_1,
                prefixo="camada_1",
            )
            ativacoes = torch.sparse.mm(
                matriz_1,
                planos_esparsos.t(),
            ).t()
            ativacoes = F.gelu(ativacoes + self.camada_1.bias)
            podadas = self._selecionar_topk_ativo(ativacoes, grupos)

            # Mover o gate para a entrada da segunda projecao e
            # algebricamente identico a ponderar a saida de cada microgrupo.
            gates_unidades = self._gates_por_unidade(
                planos_esparsos,
                grupos,
                gates.to(planos_esparsos.dtype),
            )
            matriz_2 = self._matriz_completa(
                self.camada_2,
                prefixo="camada_2",
            )
            saida = torch.sparse.mm(
                matriz_2,
                (podadas * gates_unidades).t(),
            ).t()
        return (saida + self.camada_2.bias).to(planos.dtype).view(*forma)

    def _forward_treino_banco_denso(self, estados: Tensor) -> Tensor:
        """Controle rapido que mantem Parameters COO, mas multiplica zeros."""

        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        grupos, gates = self.roteador(planos)
        pesos_1 = self._banco_denso_completo(
            self.camada_1,
            prefixo="camada_1",
        )
        ativacoes = F.gelu(
            F.linear(planos, pesos_1, self.camada_1.bias)
        )
        podadas = self._selecionar_topk_ativo(ativacoes, grupos)
        gates_unidades = self._gates_por_unidade(planos, grupos, gates)
        pesos_2 = self._banco_denso_completo(
            self.camada_2,
            prefixo="camada_2",
        )
        return F.linear(
            podadas * gates_unidades,
            pesos_2,
            self.camada_2.bias,
        ).view(*forma)

    def _forward_treino_microblocos_densos(self, estados: Tensor) -> Tensor:
        """Executa so grupos ativos, densificando cada microbloco pequeno."""

        forma = estados.shape
        planos = estados.reshape(-1, forma[-1])
        grupos, gates = self.roteador(planos)
        quantidade_tokens = planos.shape[0]
        tamanho_grupo = self.camada_1.tamanho_grupo
        ativos = self.configuracao_roteamento.grupos_ativos
        dtype_calculo = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else planos.dtype
        )
        ativacoes = torch.zeros(
            (quantidade_tokens, ativos, tamanho_grupo),
            dtype=dtype_calculo,
            device=planos.device,
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
            indices = getattr(
                self.camada_1,
                f"indices_bloco_{grupo:02d}",
            )
            parametros = getattr(
                self.camada_1,
                f"parametros_bloco_{grupo:02d}",
            )
            pesos = self._materializar_pesos(
                self.camada_1,
                indices,
                parametros,
                self.camada_1._forma_grupo(),
            )
            inicio_bias = grupo * tamanho_grupo
            projetados = F.gelu(
                F.linear(
                    planos.index_select(0, tokens),
                    pesos,
                    self.camada_1.bias[
                        inicio_bias : inicio_bias + tamanho_grupo
                    ],
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

        saida = torch.zeros(
            (quantidade_tokens, self.configuracao_base.dimensao),
            dtype=dtype_calculo,
            device=planos.device,
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
            indices = getattr(
                self.camada_2,
                f"indices_bloco_{grupo:02d}",
            )
            parametros = getattr(
                self.camada_2,
                f"parametros_bloco_{grupo:02d}",
            )
            pesos = self._materializar_pesos(
                self.camada_2,
                indices,
                parametros,
                self.camada_2._forma_grupo(),
            )
            projetados = F.linear(
                podadas[tokens, posicoes],
                pesos,
            )
            saida.index_add_(
                0,
                tokens,
                projetados
                * gates[tokens, posicoes]
                .to(projetados.dtype)
                .unsqueeze(1),
            )
        return (saida + self.camada_2.bias).view(*forma)

    def forward(self, estados: Tensor) -> Tensor:
        if self.training and torch.is_grad_enabled():
            if self.configuracao_treino.implementacao == "coo_agregado":
                return self._forward_treino_agregado(estados)
            if (
                self.configuracao_treino.implementacao
                == "microblocos_densos"
            ):
                return self._forward_treino_microblocos_densos(estados)
            return self._forward_treino_banco_denso(estados)
        return super().forward(estados)


class ModeloGeradorEsparsoTreinoV73(ModeloGeradorEsparsoRoteadoV73):
    """V7.3 com executor agregado para treinamento."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao: ConfiguracaoGeradorEsparso = ConfiguracaoGeradorEsparso(),
        semente: int = 20260728,
        configuracao_roteamento: ConfiguracaoRoteamentoV73 = (
            ConfiguracaoRoteamentoV73()
        ),
        configuracao_treino: ConfiguracaoTreinoV73 = (
            ConfiguracaoTreinoV73()
        ),
    ) -> None:
        super().__init__(
            vocabulario,
            pad_id,
            configuracao,
            semente,
            configuracao_roteamento,
        )
        self.configuracao_treino = configuracao_treino
        configuracao_treino.validar()
        for indice, bloco in enumerate(self.blocos):
            bloco.ffn = FFNCombinatoriaTreinoAgregadoV73(
                bloco.ffn,
                configuracao_treino,
            )
            self.blocos[indice] = BlocoTreinoVetorizadoV73(
                bloco,
                configuracao_treino,
            )

    def auditoria(self) -> dict[str, int | float | bool | str | dict]:
        auditoria = super().auditoria()
        auditoria.update(
            {
                "modelo": "gerador-esparso-v73-treino-vetorizado",
                "runtime_treino_coo_agregado": (
                    self.configuracao_treino.implementacao
                    == "coo_agregado"
                ),
                "implementacao_treino": (
                    self.configuracao_treino.implementacao
                ),
                "implementacao_atencao_treino": (
                    self.configuracao_treino.atencao
                ),
                "implementacao_qk_treino": (
                    self.configuracao_treino.projecoes_qk
                ),
                "treino_qk_denso_temporario": (
                    self.configuracao_treino.projecoes_qk
                    == "densas_temporarias"
                ),
                "treino_materializa_matriz_atencao_completa": (
                    self.configuracao_treino.atencao
                    == "matriz_densa_topk"
                ),
                "inferencia_atencao_topk_em_blocos": True,
                "parametros_adicionais_runtime": 0,
                "roteamento_logico_somente_grupos_selecionados": True,
                "gradientes_grupos_inativos_zerados": True,
                "treino_executa_banco_coo_completo": (
                    self.configuracao_treino.implementacao
                    == "coo_agregado"
                ),
                "treino_executa_banco_denso_completo": (
                    self.configuracao_treino.implementacao
                    == "banco_denso_completo"
                ),
                "treino_executa_somente_grupos_ativos": (
                    self.configuracao_treino.implementacao
                    == "microblocos_densos"
                ),
                "treino_multiplica_arestas_de_grupos_inativos": (
                    self.configuracao_treino.implementacao
                    in {"coo_agregado", "banco_denso_completo"}
                ),
                "matriz_de_pesos_treino_esparsa_coo": (
                    self.configuracao_treino.implementacao
                    == "coo_agregado"
                ),
                "multiplicacoes_ffn_por_camada": (
                    2
                    if self.configuracao_treino.implementacao
                    in {"coo_agregado", "banco_denso_completo"}
                    else 2 * self.configuracao_roteamento.grupos
                ),
                "inferencia_somente_grupos_selecionados": True,
                "execucao_somente_grupos_selecionados": False,
                "elegivel_promocao": False,
            }
        )
        return auditoria
