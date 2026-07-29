"""V8 multimodal esparsa com fusão causal de imagem e texto.

A imagem é dividida em patches, projetada por uma matriz COO treinável e
inserida como prefixo causal antes dos tokens de texto. O núcleo de linguagem
reutiliza o roteamento combinatório da V7.3: contextos diferentes continuam
selecionando microgrupos diferentes do mesmo banco compartilhado de pesos.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .camada_linear_esparsa import LinearBlocoEsparsa
from .modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    _topologia_por_saida,
    codificacao_senoidal,
)
from .roteamento_combinatorio_v73 import (
    ConfiguracaoRoteamentoV73,
    ModeloGeradorEsparsoRoteadoV73,
)
from .treino_vetorizado_v73 import (
    BlocoTreinoVetorizadoV73,
    ConfiguracaoTreinoV73,
    FFNCombinatoriaTreinoAgregadoV73,
)


@dataclass(frozen=True)
class ConfiguracaoMultimodalV8:
    """Configuração do caminho visual sem criar uma projeção densa."""

    canais_imagem: int = 3
    tamanho_patch: int = 8
    altura_imagem: int = 64
    largura_imagem: int = 64
    fan_in_projecao_visual: int = 48
    semente_visual: int = 20260801

    @property
    def dimensao_patch(self) -> int:
        return self.canais_imagem * self.tamanho_patch**2

    @property
    def quantidade_patches(self) -> int:
        return (
            self.altura_imagem
            // self.tamanho_patch
            * (self.largura_imagem // self.tamanho_patch)
        )

    def validar(self, base: ConfiguracaoGeradorEsparso) -> None:
        if self.canais_imagem <= 0:
            raise ValueError("canais_imagem deve ser positivo")
        if self.tamanho_patch <= 0:
            raise ValueError("tamanho_patch deve ser positivo")
        if (
            self.altura_imagem <= 0
            or self.largura_imagem <= 0
            or self.altura_imagem % self.tamanho_patch
            or self.largura_imagem % self.tamanho_patch
        ):
            raise ValueError(
                "altura e largura devem ser positivas e divisiveis pelo patch"
            )
        if not 0 < self.fan_in_projecao_visual <= self.dimensao_patch:
            raise ValueError("fan_in_projecao_visual invalido")
        if self.quantidade_patches >= base.maximo_contexto:
            raise ValueError(
                "os patches devem deixar pelo menos uma posicao para texto"
            )


class EncoderVisualEsparsoV8(nn.Module):
    """Converte uma imagem normalizada em prefixo visual por projeção COO."""

    def __init__(
        self,
        dimensao_modelo: int,
        configuracao: ConfiguracaoMultimodalV8,
    ) -> None:
        super().__init__()
        self.dimensao_modelo = dimensao_modelo
        self.configuracao = configuracao
        indices, valores = _topologia_por_saida(
            configuracao.dimensao_patch,
            dimensao_modelo,
            configuracao.fan_in_projecao_visual,
            configuracao.semente_visual,
        )
        self.projecao_patches = LinearBlocoEsparsa(
            configuracao.dimensao_patch,
            dimensao_modelo,
            indices,
            valores,
        )
        # Em inferência, a topologia visual fixa é materializada em CSR uma
        # única vez; o treino continua usando COO com gradientes normais.
        self.projecao_patches.configurar_cache_csr_inferencia()
        self.normalizacao = nn.LayerNorm(dimensao_modelo)
        self.embedding_modalidade = nn.Parameter(
            torch.zeros(dimensao_modelo)
        )
        self.gate_visual = nn.Parameter(torch.tensor(-0.5))
        self.register_buffer(
            "posicoes_patches",
            codificacao_senoidal(
                configuracao.quantidade_patches,
                dimensao_modelo,
            ),
            persistent=False,
        )

    def extrair_patches(self, imagens: Tensor) -> Tensor:
        """Cria patches sem cópia intermediária densa de pesos."""

        if imagens.ndim != 4:
            raise ValueError("imagens deve possuir shape [lote, canais, H, W]")
        _, canais, altura, largura = imagens.shape
        esperado = self.configuracao
        if canais != esperado.canais_imagem:
            raise ValueError("quantidade de canais da imagem e invalida")
        if (
            altura != esperado.altura_imagem
            or largura != esperado.largura_imagem
        ):
            raise ValueError(
                "dimensoes da imagem divergem da configuracao multimodal"
            )
        patch = esperado.tamanho_patch
        patches = (
            imagens.unfold(2, patch, patch)
            .unfold(3, patch, patch)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        return patches.view(
            imagens.shape[0],
            esperado.quantidade_patches,
            esperado.dimensao_patch,
        )

    def forward(self, imagens: Tensor) -> Tensor:
        if not imagens.is_floating_point():
            raise TypeError("imagens devem ser tensores de ponto flutuante")
        patches = self.extrair_patches(imagens)
        forma = patches.shape
        projetados = self.projecao_patches(
            patches.reshape(-1, forma[-1])
        ).view(forma[0], forma[1], self.dimensao_modelo)
        projetados = self.normalizacao(projetados)
        contexto = (
            projetados
            + self.posicoes_patches[: forma[1]].unsqueeze(0).to(
                dtype=projetados.dtype
            )
            + self.embedding_modalidade.view(1, 1, -1)
        )
        return torch.sigmoid(self.gate_visual) * contexto

    def auditoria(self) -> dict[str, int | float | bool]:
        parametros = sum(p.numel() for p in self.parameters())
        conexoes = self.projecao_patches.conexoes
        densas = (
            self.projecao_patches.entradas
            * self.projecao_patches.saidas
        )
        return {
            "parametros_visuais": parametros,
            "projecao_visual_coo": True,
            "lineares_densas_visuais": sum(
                isinstance(modulo, nn.Linear) for modulo in self.modules()
            ),
            "conexoes_projecao_visual": conexoes,
            "conexoes_visuais_ativas_por_patch": conexoes,
            "densidade_projecao_visual": conexoes / densas,
            "quantidade_patches": self.configuracao.quantidade_patches,
        }


class ModeloMultimodalEsparsoV8(ModeloGeradorEsparsoRoteadoV73):
    """Modelo imagem-texto que preserva o núcleo combinatório da V7.3."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao_base: ConfiguracaoGeradorEsparso = (
            ConfiguracaoGeradorEsparso()
        ),
        semente: int = 20260801,
        configuracao_roteamento: ConfiguracaoRoteamentoV73 = (
            ConfiguracaoRoteamentoV73()
        ),
        configuracao_multimodal: ConfiguracaoMultimodalV8 = (
            ConfiguracaoMultimodalV8()
        ),
    ) -> None:
        configuracao_multimodal.validar(configuracao_base)
        super().__init__(
            vocabulario,
            pad_id,
            configuracao_base,
            semente,
            configuracao_roteamento,
        )
        self.configuracao_multimodal = configuracao_multimodal
        self.encoder_visual = EncoderVisualEsparsoV8(
            configuracao_base.dimensao,
            configuracao_multimodal,
        )
        self.embedding_modalidade_texto = nn.Parameter(
            torch.zeros(configuracao_base.dimensao)
        )

    def _estados_entrada(
        self,
        tokens: Tensor,
        imagens: Tensor | None,
    ) -> tuple[Tensor, int]:
        if tokens.ndim != 2:
            raise ValueError("tokens deve possuir shape [lote, tempo]")
        if tokens.shape[1] == 0:
            raise ValueError("a sequencia de texto nao pode ser vazia")

        prefixo = None
        quantidade_visual = 0
        if imagens is not None:
            if imagens.shape[0] != tokens.shape[0]:
                raise ValueError("lotes de imagens e tokens devem coincidir")
            prefixo = self.encoder_visual(imagens)
            quantidade_visual = prefixo.shape[1]

        limite_texto = self.configuracao.maximo_contexto - quantidade_visual
        if limite_texto <= 0:
            raise ValueError("prefixo visual ocupou todo o contexto")
        if tokens.shape[1] > limite_texto:
            raise ValueError(
                "texto e prefixo visual excedem o contexto configurado"
            )

        estados_texto = (
            self.embedding(tokens) * math.sqrt(self.configuracao.dimensao)
            + self.embedding_modalidade_texto.view(1, 1, -1)
        )
        estados = (
            estados_texto
            if prefixo is None
            else torch.cat((prefixo, estados_texto), dim=1)
        )
        estados = (
            estados
            + torch.sigmoid(self.gate_posicao)
            * self.posicoes[: estados.shape[1]]
            .unsqueeze(0)
            .to(dtype=estados.dtype)
        )
        return estados, tokens.shape[1]

    def forward(
        self,
        tokens: Tensor,
        imagens: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        estados, comprimento_texto = self._estados_entrada(tokens, imagens)
        for bloco in self.blocos:
            estados = bloco(estados)
        estados_texto = estados[:, -comprimento_texto:]
        normalizados = self.normalizacao_saida(estados_texto)
        logits = F.linear(
            normalizados,
            self.embedding.weight,
            self.saida_bias,
        )

        roteadores = [bloco.ffn.roteador for bloco in self.blocos]
        auxiliares = {
            "gate_posicao": torch.sigmoid(self.gate_posicao),
            "gate_visual": torch.sigmoid(self.encoder_visual.gate_visual),
            "perda_roteamento": torch.stack(
                [roteador.perda_auxiliar() for roteador in roteadores]
            ).mean(),
            "balanceamento_roteamento": torch.stack(
                [roteador.ultimo_balanceamento for roteador in roteadores]
            ).mean(),
            "z_roteamento": torch.stack(
                [roteador.ultimo_z for roteador in roteadores]
            ).mean(),
            "entropia_roteamento": torch.stack(
                [roteador.ultima_entropia for roteador in roteadores]
            ).mean(),
        }
        return logits, auxiliares

    @torch.inference_mode()
    def gerar_greedy(
        self,
        tokens: Tensor,
        imagens: Tensor,
        maximo_novos_tokens: int,
        eos_id: int | None = None,
    ) -> Tensor:
        """Gera texto multimodal por referência exata, sem cache experimental."""

        if maximo_novos_tokens <= 0:
            raise ValueError("maximo_novos_tokens deve ser positivo")
        if self.training:
            raise RuntimeError("a geracao exige o modelo em modo eval")
        gerados = tokens.clone()
        limite_texto = (
            self.configuracao.maximo_contexto
            - self.configuracao_multimodal.quantidade_patches
        )
        for _ in range(maximo_novos_tokens):
            contexto = gerados[:, -limite_texto:]
            logits, _ = self(contexto, imagens)
            proximo = logits[:, -1].argmax(dim=-1, keepdim=True)
            gerados = torch.cat((gerados, proximo), dim=1)
            if eos_id is not None and bool(proximo.eq(eos_id).all()):
                break
        return gerados

    def iniciar_cache_geracao(self, *args, **kwargs):
        """Impede uso acidental do cache textual que não recebe imagem."""

        raise RuntimeError(
            "a V8 ainda nao possui cache multimodal; use gerar_greedy"
        )

    def avancar_cache_geracao(self, *args, **kwargs):
        """Impede continuar um cache que não representa o prefixo visual."""

        raise RuntimeError(
            "a V8 ainda nao possui cache multimodal; use gerar_greedy"
        )

    def auditoria(self) -> dict[str, object]:
        auditoria = super().auditoria()
        visual = self.encoder_visual.auditoria()
        parametros = sum(p.numel() for p in self.parameters())
        parametros_visuais = int(visual["parametros_visuais"]) + (
            self.embedding_modalidade_texto.numel()
        )
        auditoria.update(
            {
                "modelo": "gerador-esparso-v8-multimodal",
                "versao_arquitetura": "8.0.0",
                "modalidades": ["imagem", "texto"],
                "fusao": "prefixo_visual_causal",
                "backbone": "v7.3-roteamento-combinatorio",
                "parametros": parametros,
                "parametros_visuais": parametros_visuais,
                "parametros_backbone_textual": parametros - parametros_visuais,
                "configuracao_multimodal": asdict(
                    self.configuracao_multimodal
                ),
                "projecao_visual_coo": visual["projecao_visual_coo"],
                "lineares_densas_visuais": visual[
                    "lineares_densas_visuais"
                ],
                "densidade_projecao_visual": visual[
                    "densidade_projecao_visual"
                ],
                "quantidade_patches": visual["quantidade_patches"],
                "causalidade_textual_preservada": True,
                "roteamento_dependente_de_imagem_e_texto": True,
                "cache_causal_incremental": False,
                "checkpoint_treinado": False,
                "cache_multimodal_incremental": False,
                "elegivel_promocao": False,
            }
        )
        return auditoria


class ModeloMultimodalEsparsoTreinoV8(ModeloMultimodalEsparsoV8):
    """V8 com backend vetorizado exclusivo do caminho de treinamento."""

    def __init__(
        self,
        vocabulario: int,
        pad_id: int,
        configuracao_base: ConfiguracaoGeradorEsparso = (
            ConfiguracaoGeradorEsparso()
        ),
        semente: int = 20260801,
        configuracao_roteamento: ConfiguracaoRoteamentoV73 = (
            ConfiguracaoRoteamentoV73()
        ),
        configuracao_multimodal: ConfiguracaoMultimodalV8 = (
            ConfiguracaoMultimodalV8()
        ),
        configuracao_treino: ConfiguracaoTreinoV73 = (
            ConfiguracaoTreinoV73(
                implementacao="microblocos_densos",
                atencao="matriz_densa_topk",
                projecoes_qk="densas_temporarias",
            )
        ),
    ) -> None:
        super().__init__(
            vocabulario,
            pad_id,
            configuracao_base,
            semente,
            configuracao_roteamento,
            configuracao_multimodal,
        )
        configuracao_treino.validar()
        self.configuracao_treino = configuracao_treino
        for indice, bloco in enumerate(self.blocos):
            bloco.ffn = FFNCombinatoriaTreinoAgregadoV73(
                bloco.ffn,
                configuracao_treino,
            )
            self.blocos[indice] = BlocoTreinoVetorizadoV73(
                bloco,
                configuracao_treino,
            )

    def auditoria(self) -> dict[str, object]:
        auditoria = super().auditoria()
        auditoria.update(
            {
                "backend_treino_vetorizado": True,
                "implementacao_treino": (
                    self.configuracao_treino.implementacao
                ),
                "atencao_treino": self.configuracao_treino.atencao,
                "projecoes_qk_treino": (
                    self.configuracao_treino.projecoes_qk
                ),
                "parametros_adicionais_backend_treino": 0,
                "treino_somente_microgrupos_ativos": (
                    self.configuracao_treino.implementacao
                    == "microblocos_densos"
                ),
            }
        )
        return auditoria
