"""Treina a V7 textual sem alterar o checkpoint-base preservado.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW

from src.geracao_textual_v73 import (
    ConfiguracaoDecodificacaoV73,
    gerar_v73,
    resumir_metricas_geracao,
)
from src.dados_binarios_v7 import CorpusShardsV7
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v73 import ConfiguracaoRuntimeV73, ModeloGeradorEsparsoV73
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.runtime_condicional_v73 import ConfiguracaoRuntimeCondicionalV73
from src.tokenizador_palavras import (
    TokenizadorBPEBytes,
    TokenizadorBPECodigo,
    TokenizadorBytes,
)
from src.topologia_checkpoint_v73 import metadados_topologia
from src.treino_vetorizado_v73 import ConfiguracaoTreinoV73, ModeloGeradorEsparsoTreinoV73
from src.versao import VERSAO_PROJETO


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_BASE = RAIZ / "modelos" / "gerador_esparso_v73_bpe8192_5x3000.pt"
DATASET_PADRAO = Path(
    os.environ.get(
        "AI_ESPARSA_DATASET",
        str(RAIZ / "dados" / "bpe_codigo_8192"),
    )
)
SAIDA_PADRAO = Path(
    os.environ.get(
        "AI_ESPARSA_SAIDA",
        str(RAIZ / "resultados" / "v7_bpe8192_continuacao"),
    )
)
# A seed da topologia precisa acompanhar os pesos no checkpoint. Ela e
# deliberadamente separada da seed que embaralha os lotes do experimento.
SEMENTE_TOPOLOGIA_V73 = 20260728


@dataclass(frozen=True)
class ConfiguracaoExperimento:
    """Parametros reproduziveis do ciclo solicitado."""

    epocas: int = 5
    # ``None`` significa uma passagem integral por todos os segmentos.
    passos_por_epoca: int | None = None
    tamanho_lote: int = 32
    maximo_contexto: int = 640
    taxa_aprendizado: float = 3e-4
    weight_decay: float = 0.01
    warmup_passos: int = 100
    fator_minimo_lr: float = 0.1
    gradiente_maximo: float = 1.0
    semente: int = 20260802
    maximo_tokens_geracao: int = 256
    tipo_tokenizador: str = "bpe"
    tamanho_vocabulario_bpe: int = 2048
    implementacao_treino: str = "banco_denso_completo"
    compilacao: str = "reduce-overhead"
    usar_prefetch: bool = True
    usar_baldes: bool = True
    adamw_fundido: bool = True
    projecao_logits: str = "auto"
    peso_eos: float = 1.0
    peso_repeticao: float = 0.0
    janela_repeticao: int = 4
    quantidade_prompts_geracao: int = 8
    arquitetura_modelo: str = "v73"
    dimensao_modelo: int = 128
    camadas_modelo: int = 3
    top_k_atencao_modelo: int = 32
    fan_in_qk_modelo: int = 32
    especialistas_modelo: int = 384
    fan_in_ffn_modelo: int = 32
    fan_out_ffn_modelo: int = 48
    top_k_ffn_modelo: int = 64

    def validar(self) -> None:
        if self.epocas <= 0:
            raise ValueError("epocas deve ser positivo")
        if (
            self.passos_por_epoca is not None
            and self.passos_por_epoca <= 0
        ) or self.tamanho_lote <= 0:
            raise ValueError("passos e lote devem ser positivos")
        if not 32 <= self.maximo_contexto <= 640:
            raise ValueError("contexto deve ficar entre 32 e 640")
        if self.taxa_aprendizado <= 0 or self.warmup_passos < 0:
            raise ValueError("taxa e warmup invalidos")
        if not 0 < self.fator_minimo_lr <= 1:
            raise ValueError("fator minimo de LR deve ficar em (0, 1]")
        if self.tipo_tokenizador not in {"bpe", "bpe_codigo", "bytes"}:
            raise ValueError("tokenizador deve ser bpe, bpe_codigo ou bytes")
        if self.arquitetura_modelo != "v73":
            raise ValueError("somente a arquitetura V7.3 BPE e suportada")
        if self.tamanho_vocabulario_bpe < 260:
            raise ValueError("vocabulario BPE deve possuir pelo menos 260 IDs")
        if self.implementacao_treino not in {
            "coo_agregado",
            "microblocos_densos",
            "banco_denso_completo",
        }:
            raise ValueError("backend de treino V7.3 invalido")
        if self.compilacao not in {
            "desligada",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError("modo de compilacao invalido")
        if self.projecao_logits not in {
            "auto",
            "completa",
            "somente_alvos",
        }:
            raise ValueError("modo de projecao de logits invalido")
        if self.peso_eos < 1:
            raise ValueError("peso de EOS deve ser >= 1")
        if self.peso_repeticao < 0:
            raise ValueError("peso de repeticao nao pode ser negativo")
        if self.janela_repeticao <= 0:
            raise ValueError("janela de repeticao deve ser positiva")
        if self.quantidade_prompts_geracao <= 0:
            raise ValueError("quantidade de prompts deve ser positiva")


TokenizadorTextual = TokenizadorBytes | TokenizadorBPEBytes | TokenizadorBPECodigo


def carregar_registros(caminho: Path) -> list[tuple[str, str]]:
    """Lê e valida pares conversacionais sem antecipar a tokenização."""

    registros: list[tuple[str, str]] = []
    with caminho.open("r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, 1):
            item = json.loads(linha)
            prompt = item.get("prompt")
            resposta = item.get("resposta")
            if not isinstance(prompt, str) or not isinstance(resposta, str):
                raise TypeError(
                    f"prompt/resposta invalidos em {caminho.name}:"
                    f"{numero_linha}"
                )
            prompt = prompt.strip()
            resposta = resposta.strip()
            if not prompt or not resposta:
                raise ValueError(
                    f"registro vazio em {caminho.name}:{numero_linha}"
                )
            registros.append((prompt, resposta))
    if not registros:
        raise ValueError(f"dataset vazio: {caminho}")
    return registros


def criar_tokenizador(
    registros_treino: list[tuple[str, str]],
    configuracao: ConfiguracaoExperimento,
) -> TokenizadorTextual:
    """Aprende BPE somente no treino ou mantém o fallback byte-level."""

    if configuracao.tipo_tokenizador == "bytes":
        return TokenizadorBytes()
    textos = (
        f"usuario: {prompt}\nassistente: {resposta}"
        for prompt, resposta in registros_treino
    )
    return TokenizadorBPEBytes(
        textos,
        tamanho_vocabulario=configuracao.tamanho_vocabulario_bpe,
    )


class CorpusConversacionalV7:
    """Segmenta respostas sem descartar tokens ou ensinar EOS artificial."""

    def __init__(
        self,
        caminho: Path,
        tokenizador: TokenizadorTextual,
        maximo_contexto: int,
        registros: list[tuple[str, str]] | None = None,
    ) -> None:
        self.caminho = caminho
        self.tokenizador = tokenizador
        self.maximo_contexto = maximo_contexto
        self.exemplos: list[tuple[list[int], list[int]]] = []
        pares = registros if registros is not None else carregar_registros(caminho)
        prompts_truncados = respostas_segmentadas = 0
        tokens_alvo = bytes_alvo = 0
        for prompt, resposta in pares:
            segmentos, auditoria = self._preparar_exemplo(prompt, resposta)
            self.exemplos.extend(segmentos)
            prompts_truncados += int(auditoria["prompt_truncado"])
            respostas_segmentadas += int(auditoria["resposta_segmentada"])
            tokens_alvo += int(auditoria["tokens_alvo"])
            bytes_alvo += int(auditoria["bytes_alvo"])
        self.estatisticas = {
            "registros": len(pares),
            "segmentos": len(self.exemplos),
            "prompts_truncados": prompts_truncados,
            "respostas_segmentadas": respostas_segmentadas,
            "tokens_alvo": tokens_alvo,
            "bytes_alvo": bytes_alvo,
            "tokens_descartados": 0,
            "eos_artificiais": 0,
        }

    @staticmethod
    def _preservar_extremos(tokens: list[int], limite: int) -> list[int]:
        """Mantém instrução inicial e entrada final quando o prompt é longo."""

        if len(tokens) <= limite:
            return tokens
        cabeca = limite // 2
        return tokens[:cabeca] + tokens[-(limite - cabeca):]

    def _preparar_exemplo(
        self,
        prompt: str,
        resposta: str,
    ) -> tuple[list[tuple[list[int], list[int]]], dict[str, int | bool]]:
        prefixo_original = self.tokenizador.codificar(
            f"usuario: {prompt}\nassistente: ", bos=True, eos=False
        )
        alvo = self.tokenizador.codificar(resposta, bos=False, eos=True)
        # Metade da janela fica disponível para os alvos de cada segmento; a
        # outra metade carrega prompt ou continuação anterior causal.
        alvos_por_segmento = max(8, self.maximo_contexto // 2)
        contexto_disponivel = self.maximo_contexto - alvos_por_segmento
        prefixo = self._preservar_extremos(
            prefixo_original,
            contexto_disponivel,
        )
        sequencia = prefixo + alvo
        inicio_resposta = len(prefixo)
        inicio_alvo = inicio_resposta
        segmentos: list[tuple[list[int], list[int]]] = []
        while inicio_alvo < len(sequencia):
            fim_alvo = min(
                len(sequencia),
                inicio_alvo + alvos_por_segmento,
            )
            inicio_contexto = max(
                0,
                inicio_alvo - contexto_disponivel,
            )
            janela = sequencia[inicio_contexto:fim_alvo]
            entrada = janela[:-1]
            labels = [
                sequencia[indice + 1]
                if indice + 1 >= inicio_alvo
                else -100
                for indice in range(inicio_contexto, fim_alvo - 1)
            ]
            if not labels or len(entrada) != len(labels):
                raise RuntimeError("segmento nao possui alvo supervisionado")
            segmentos.append((entrada, labels))
            inicio_alvo = fim_alvo
        comprimentos = self.tokenizador.comprimentos_bytes
        return segmentos, {
            "prompt_truncado": len(prefixo) < len(prefixo_original),
            "resposta_segmentada": len(segmentos) > 1,
            "tokens_alvo": len(alvo),
            "bytes_alvo": sum(
                comprimentos[token_id]
                for token_id in alvo
            ),
        }

    def comprimento_exemplo(self, indice: int) -> int:
        """Devolve o comprimento causal sem materializar o lote."""

        return len(self.exemplos[indice][0])

    def preparar_lote(
        self,
        indices: list[int],
        *,
        comprimento_alvo: int | None = None,
        fixar_memoria: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Monta o lote na CPU para permitir prefetch concorrente."""

        comprimento_real = max(
            len(self.exemplos[indice][0]) for indice in indices
        )
        comprimento = comprimento_alvo or comprimento_real
        if comprimento < comprimento_real:
            raise ValueError("comprimento_alvo menor que o maior exemplo")
        if comprimento > self.maximo_contexto:
            raise ValueError("lote excede o contexto maximo")
        tokens = torch.full(
            (len(indices), comprimento),
            self.tokenizador.pad_id,
            dtype=torch.long,
            pin_memory=fixar_memoria,
        )
        labels = torch.full_like(tokens, -100)
        for linha, indice in enumerate(indices):
            entrada, alvo = self.exemplos[indice]
            tokens[linha, : len(entrada)] = torch.tensor(entrada)
            labels[linha, : len(alvo)] = torch.tensor(alvo)
        return tokens, labels

    def lote(
        self,
        indices: list[int],
        dispositivo: torch.device,
        *,
        comprimento_alvo: int | None = None,
        fixar_memoria: bool = False,
    ) -> tuple[Tensor, Tensor]:
        tokens, labels = self.preparar_lote(
            indices,
            comprimento_alvo=comprimento_alvo,
            fixar_memoria=fixar_memoria,
        )
        assincrono = fixar_memoria and dispositivo.type == "cuda"
        return (
            tokens.to(dispositivo, non_blocking=assincrono),
            labels.to(dispositivo, non_blocking=assincrono),
        )


BALDES_CONTEXTO = (128, 256, 384, 512, 640)


def comprimento_balde(
    corpus: CorpusConversacionalV7 | CorpusShardsV7,
    indices: Sequence[int],
    usar_baldes: bool,
) -> int | None:
    if not usar_baldes:
        return None
    comprimento = max(corpus.comprimento_exemplo(indice) for indice in indices)
    for balde in BALDES_CONTEXTO:
        if balde >= comprimento and balde <= corpus.maximo_contexto:
            return balde
    return corpus.maximo_contexto


def iterar_lotes(
    corpus: CorpusConversacionalV7 | CorpusShardsV7,
    lotes: Sequence[list[int]],
    dispositivo: torch.device,
    *,
    usar_prefetch: bool,
    usar_baldes: bool,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Sobrepoe a montagem do proximo lote ao calculo da GPU atual."""

    fixar_memoria = usar_prefetch and dispositivo.type == "cuda"

    def preparar(indices: list[int]) -> tuple[Tensor, Tensor]:
        return corpus.preparar_lote(
            indices,
            comprimento_alvo=comprimento_balde(
                corpus,
                indices,
                usar_baldes,
            ),
            fixar_memoria=fixar_memoria,
        )

    if not lotes:
        return
    if not usar_prefetch:
        for indices in lotes:
            tokens, labels = preparar(indices)
            yield tokens.to(dispositivo), labels.to(dispositivo)
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        futuro = executor.submit(preparar, lotes[0])
        for indices_seguintes in lotes[1:]:
            tokens, labels = futuro.result()
            futuro = executor.submit(preparar, indices_seguintes)
            yield (
                tokens.to(dispositivo, non_blocking=True),
                labels.to(dispositivo, non_blocking=True),
            )
        tokens, labels = futuro.result()
        yield (
            tokens.to(dispositivo, non_blocking=True),
            labels.to(dispositivo, non_blocking=True),
        )


def sha256_arquivo(caminho: Path) -> str:
    """Calcula identidade do artefato sem guardar dados no checkpoint."""

    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def salvar_atomico(conteudo: Any, caminho: Path, *, json_: bool) -> None:
    """Evita artefatos parciais em caso de interrupcao."""

    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    if json_:
        temporario.write_text(
            json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        torch.save(conteudo, temporario)
    os.replace(temporario, caminho)


def transferir_base(
    modelo,
    caminho: Path,
    tokenizador: TokenizadorTextual | None = None,
) -> dict[str, Any]:
    """Transfere pesos V7.3 e valida o vocabulario quando compativel."""

    checkpoint = torch.load(caminho, map_location="cpu", weights_only=True)
    modelos_v73 = {
        "gerador-esparso-v73-base",
        "gerador-esparso-v73-treino-fp32",
        "gerador-esparso-v7-textual-experimental",
        "gerador-esparso-v7-textual-byte-experimental",
    }
    modelo_origem = checkpoint.get("modelo")
    if modelo_origem not in modelos_v73:
        raise ValueError("checkpoint de transferencia nao pertence a V7.3")
    configuracao_origem = checkpoint.get("configuracao")
    if configuracao_origem is not None and configuracao_origem != asdict(modelo.configuracao):
        raise ValueError("arquitetura do checkpoint de transferencia diverge")
    if int(checkpoint.get("semente", SEMENTE_TOPOLOGIA_V73)) != SEMENTE_TOPOLOGIA_V73:
        raise ValueError("seed topologica do checkpoint de transferencia diverge")

    estado_origem = checkpoint["estado_modelo"]
    destino = modelo.state_dict()
    vocabulario_origem = checkpoint.get("vocabulario")
    embedding_compativel = (
        "embedding.weight" in estado_origem
        and "embedding.weight" in destino
        and estado_origem["embedding.weight"].shape == destino["embedding.weight"].shape
    )
    if embedding_compativel and modelo_origem != "gerador-esparso-v73-base":
        if tokenizador is None:
            raise ValueError(
                "tokenizador e obrigatorio ao transferir vocabulario experimental"
            )
        if vocabulario_origem is None or tuple(vocabulario_origem) != tuple(
            tokenizador.id_para_token
        ):
            raise ValueError("vocabulario do checkpoint de transferencia diverge")

    compativeis = {
        chave: tensor
        for chave, tensor in estado_origem.items()
        if chave in destino and destino[chave].shape == tensor.shape
    }
    reinicializados = sorted(
        chave
        for chave, tensor in estado_origem.items()
        if chave not in destino or destino[chave].shape != tensor.shape
    )
    destino.update(compativeis)
    modelo.load_state_dict(destino, strict=True)
    return {
        "checkpoint": caminho.name,
        "sha256": sha256_arquivo(caminho),
        "modelo_origem": modelo_origem,
        "epoca_origem": checkpoint.get("epoca"),
        "tensores_transferidos": len(compativeis),
        "tensores_reinicializados": reinicializados,
        "vocabulario_verificado": bool(
            embedding_compativel and tokenizador is not None
        ),
        "estrategia": "v73_estrita",
    }


def criar_modelos(
    tokenizador: TokenizadorTextual,
    implementacao_treino: str = "banco_denso_completo",
    arquitetura_modelo: str = "v73",
    configuracao_experimento: ConfiguracaoExperimento | None = None,
):
    """Constroi exclusivamente a arquitetura V7.3 usada pelo modelo BPE."""

    if arquitetura_modelo != "v73":
        raise ValueError("somente a arquitetura V7.3 BPE e suportada")
    roteamento = ConfiguracaoRoteamentoV73()
    treino = ConfiguracaoTreinoV73(
        implementacao=implementacao_treino,
        atencao="matriz_densa_topk",
        projecoes_qk="densas_temporarias",
    )
    if configuracao_experimento is None:
        arquitetura = ConfiguracaoGeradorEsparso()
    else:
        arquitetura = ConfiguracaoGeradorEsparso(
            dimensao=configuracao_experimento.dimensao_modelo,
            camadas=configuracao_experimento.camadas_modelo,
            top_k_atencao=configuracao_experimento.top_k_atencao_modelo,
            fan_in_qk=configuracao_experimento.fan_in_qk_modelo,
            especialistas=configuracao_experimento.especialistas_modelo,
            fan_in_ffn=configuracao_experimento.fan_in_ffn_modelo,
            fan_out_ffn=configuracao_experimento.fan_out_ffn_modelo,
            top_k_ffn=configuracao_experimento.top_k_ffn_modelo,
            maximo_contexto=configuracao_experimento.maximo_contexto,
            bloco_consultas_atencao=min(
                32,
                configuracao_experimento.maximo_contexto,
            ),
        )
    arquitetura.validar()
    roteamento.validar(arquitetura)
    modelo = ModeloGeradorEsparsoTreinoV73(
        tokenizador.tamanho,
        tokenizador.pad_id,
        arquitetura,
        SEMENTE_TOPOLOGIA_V73,
        roteamento,
        treino,
    )
    return modelo, arquitetura, roteamento, treino


def indices_epoca(
    tamanho: int,
    configuracao: ConfiguracaoExperimento,
    epoca: int,
) -> list[list[int]]:
    """Cobre todo o corpus por padrão ou limita passos explicitamente."""

    gerador = random.Random(configuracao.semente + epoca)
    ordem = list(range(tamanho))
    gerador.shuffle(ordem)
    if configuracao.passos_por_epoca is None:
        return [
            ordem[inicio : inicio + configuracao.tamanho_lote]
            for inicio in range(0, tamanho, configuracao.tamanho_lote)
        ]
    necessarios = configuracao.passos_por_epoca * configuracao.tamanho_lote
    if necessarios > tamanho:
        ordem = (ordem * math.ceil(necessarios / tamanho))[:necessarios]
    else:
        ordem = ordem[:necessarios]
    return [
        ordem[inicio : inicio + configuracao.tamanho_lote]
        for inicio in range(0, necessarios, configuracao.tamanho_lote)
    ]


def perda_resposta(
    logits: Tensor,
    labels: Tensor,
    *,
    eos_id: int | None = None,
    peso_eos: float = 1.0,
) -> tuple[Tensor, int]:
    """Calcula CE nos alvos, com ponderacao opcional do EOS real."""

    if peso_eos < 1:
        raise ValueError("peso_eos deve ser >= 1")
    mascara = labels.ne(-100)
    quantidade = int(mascara.sum())
    alvos = labels[mascara]
    if logits.ndim == 3:
        logits_validos = logits[mascara]
    elif logits.ndim == 2:
        logits_validos = logits
        if logits.shape[0] != alvos.numel():
            raise ValueError("logits filtrados divergem dos alvos")
    else:
        raise ValueError("logits deve possuir duas ou tres dimensoes")
    perdas = F.cross_entropy(logits_validos, alvos, reduction="none")
    if eos_id is None or peso_eos == 1.0:
        return perdas.mean(), quantidade
    pesos = torch.ones_like(perdas)
    pesos = torch.where(
        alvos.eq(eos_id),
        pesos.new_full((), peso_eos),
        pesos,
    )
    return (perdas * pesos).sum() / pesos.sum().clamp_min(1.0), quantidade


def perda_repeticao_incorreta(
    logits: Tensor,
    labels: Tensor,
    *,
    janela: int = 4,
) -> tuple[Tensor, int]:
    """Penaliza somente previsoes erradas que repetem previsoes recentes."""

    if janela <= 0:
        raise ValueError("janela de repeticao deve ser positiva")
    mascara = labels.ne(-100)
    if logits.ndim == 3:
        previstos = torch.where(
            mascara,
            logits.detach().argmax(-1),
            torch.full_like(labels, -1),
        )
    elif logits.ndim == 2:
        if logits.shape[0] != int(mascara.sum()):
            raise ValueError("logits filtrados divergem dos alvos")
        previstos = torch.full_like(labels, -1)
        previstos[mascara] = logits.detach().argmax(-1)
    else:
        raise ValueError("logits deve possuir duas ou tres dimensoes")

    repetidos = torch.zeros_like(mascara)
    limite = min(janela, max(previstos.shape[1] - 1, 0))
    for atraso in range(1, limite + 1):
        atuais = previstos[:, atraso:]
        anteriores = previstos[:, :-atraso]
        repetidos[:, atraso:] |= (
            atuais.ge(0)
            & anteriores.ge(0)
            & atuais.eq(anteriores)
        )
    penalizados = repetidos & mascara & previstos.ne(labels)
    quantidade = int(penalizados.sum())
    if quantidade == 0:
        return logits.new_zeros(()), 0

    if logits.ndim == 3:
        logits_penalizados = logits[penalizados]
    else:
        logits_penalizados = logits[penalizados[mascara]]
    ids_penalizados = previstos[penalizados]
    probabilidades = logits_penalizados.softmax(-1).gather(
        -1,
        ids_penalizados.unsqueeze(-1),
    ).squeeze(-1)
    perda = -torch.log1p(
        -probabilidades.clamp(max=1.0 - 1e-6)
    ).mean()
    return perda, quantidade


class ExecutorAlvosV7(torch.nn.Module):
    """Modulo estreito para compilar somente o caminho usado no treino."""

    def __init__(
        self,
        modelo: ModeloGeradorEsparsoTreinoV73,
        *,
        somente_alvos: bool,
    ) -> None:
        super().__init__()
        self.modelo = modelo
        self.somente_alvos = somente_alvos

    def forward(
        self,
        tokens: Tensor,
        mascara_alvos: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.somente_alvos:
            return self.modelo.forward_alvos(tokens, mascara_alvos)
        return self.modelo(tokens)


class ExecutorComFallbackV7:
    """Desativa torch.compile automaticamente caso o backend recuse o grafo."""

    def __init__(
        self,
        modelo: ModeloGeradorEsparsoTreinoV73,
        modo: str,
        *,
        somente_alvos: bool,
    ) -> None:
        self.referencia = ExecutorAlvosV7(
            modelo,
            somente_alvos=somente_alvos,
        )
        self.modo_solicitado = modo
        self.somente_alvos = somente_alvos
        self.ativo = modo != "desligada"
        self.erro: str | None = None
        self.executor = self.referencia
        if self.ativo and torch.cuda.is_available():
            try:
                triton_disponivel = importlib.util.find_spec("triton") is not None
            except (ImportError, ValueError):
                triton_disponivel = False
            if not triton_disponivel:
                self.ativo = False
                self.erro = "Triton indisponivel; execucao eager selecionada"
        if self.ativo:
            try:
                self.executor = torch.compile(
                    self.referencia,
                    mode=modo,
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as exc:  # pragma: no cover - depende do ambiente
                self.ativo = False
                self.erro = f"{type(exc).__name__}: {exc}"
                self.executor = self.referencia

    def train(self) -> None:
        self.referencia.train()
        self.executor.train()

    def eval(self) -> None:
        self.referencia.eval()
        self.executor.eval()

    def __call__(
        self,
        tokens: Tensor,
        mascara_alvos: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.ativo:
            return self.referencia(tokens, mascara_alvos)
        try:
            return self.executor(tokens, mascara_alvos)
        except Exception as exc:  # pragma: no cover - depende do compilador
            self.ativo = False
            self.erro = f"{type(exc).__name__}: {exc}"
            self.executor = self.referencia
            try:
                torch.compiler.reset()
            except AttributeError:
                pass
            return self.referencia(tokens, mascara_alvos)

    def auditoria(self) -> dict[str, Any]:
        return {
            "solicitada": self.modo_solicitado,
            "ativa": self.ativo,
            "fallback": self.erro,
            "projecao_logits": (
                "somente_alvos" if self.somente_alvos else "completa"
            ),
        }


def criar_otimizador(
    modelo: ModeloGeradorEsparsoTreinoV73,
    configuracao: ConfiguracaoExperimento,
    *,
    solicitar_fundido: bool,
) -> tuple[AdamW, bool, str | None]:
    argumentos = {
        "lr": configuracao.taxa_aprendizado,
        "weight_decay": configuracao.weight_decay,
    }
    if solicitar_fundido:
        try:
            return AdamW(modelo.parameters(), fused=True, **argumentos), True, None
        except (TypeError, RuntimeError) as exc:
            erro = f"{type(exc).__name__}: {exc}"
    else:
        erro = None
    return AdamW(modelo.parameters(), **argumentos), False, erro



def selecionar_projecao_logits(
    modelo: ModeloGeradorEsparsoTreinoV73,
    entrada: Tensor,
    labels: Tensor,
    solicitado: str,
) -> tuple[str, dict[str, Any]]:
    """Escolhe a projecao mais rapida e recua para alvos em caso de OOM."""

    if solicitado != "auto":
        return solicitado, {
            "solicitada": solicitado,
            "selecionada": solicitado,
            "benchmark": None,
        }
    mascara = labels.ne(-100)
    if not bool(mascara.any()):
        raise ValueError("lote de calibracao nao possui alvos")
    modelo.train()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        estados_base = modelo.codificar_estados(entrada).detach()
    modelo.zero_grad(set_to_none=True)

    def medir(somente_alvos: bool) -> float:
        tempos: list[float] = []
        repeticoes = 8
        for indice in range(repeticoes + 2):
            modelo.zero_grad(set_to_none=True)
            estados = estados_base.detach().requires_grad_(True)
            torch.cuda.synchronize()
            inicio = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = modelo.projetar_logits(
                    estados,
                    mascara if somente_alvos else None,
                )
                if somente_alvos:
                    perda = F.cross_entropy(logits, labels[mascara])
                else:
                    perda = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
            perda.backward()
            torch.cuda.synchronize()
            if indice >= 2:
                tempos.append(time.perf_counter() - inicio)
        return statistics.median(tempos)

    resultados: dict[str, float | None] = {
        "completa_segundos": None,
        "somente_alvos_segundos": None,
    }
    try:
        resultados["completa_segundos"] = medir(False)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
    resultados["somente_alvos_segundos"] = medir(True)
    modelo.zero_grad(set_to_none=True)
    completa = resultados["completa_segundos"]
    seletiva = resultados["somente_alvos_segundos"]
    selecionada = (
        "somente_alvos"
        if completa is None or float(seletiva) < float(completa)
        else "completa"
    )
    return selecionada, {
        "solicitada": solicitado,
        "selecionada": selecionada,
        "benchmark": resultados,
        "fracao_tokens_alvo": float(mascara.float().mean()),
    }


@torch.inference_mode()
def avaliar(executor, corpus, configuracao, dispositivo) -> dict[str, float]:
    """Mede PPL/BPB projetando somente os tokens supervisionados."""

    executor.eval()
    soma_nll, tokens, bytes_alvo = 0.0, 0, 0
    comprimentos_bytes = torch.tensor(
        corpus.tokenizador.comprimentos_bytes,
        dtype=torch.long,
        device=dispositivo,
    )
    lotes = [
        list(
            range(
                posicao,
                min(
                    posicao + configuracao.tamanho_lote,
                    len(corpus.exemplos),
                ),
            )
        )
        for posicao in range(
            0,
            len(corpus.exemplos),
            configuracao.tamanho_lote,
        )
    ]
    inicio = time.perf_counter()
    for entrada, labels in iterar_lotes(
        corpus,
        lotes,
        dispositivo,
        usar_prefetch=configuracao.usar_prefetch,
        usar_baldes=configuracao.usar_baldes,
    ):
        mascara = labels.ne(-100)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = executor(entrada, mascara)
            alvos_validos = labels[mascara]
            if logits.ndim == 2:
                nll = F.cross_entropy(
                    logits,
                    alvos_validos,
                    reduction="sum",
                )
            else:
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
        soma_nll += float(nll)
        tokens += int(alvos_validos.numel())
        bytes_alvo += int(comprimentos_bytes[alvos_validos].sum())
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    perda = soma_nll / max(tokens, 1)
    return {
        "loss_resposta": perda,
        "ppl_token": math.exp(min(perda, 20.0)),
        "bits_por_byte": soma_nll / math.log(2.0) / max(bytes_alvo, 1),
        "tokens_resposta": float(tokens),
        "bytes_resposta": float(bytes_alvo),
        "duracao_segundos": duracao,
        "tokens_resposta_por_segundo": tokens / max(duracao, 1e-9),
    }


@torch.inference_mode()
def benchmark_forward(executor, corpus, configuracao, dispositivo) -> dict[str, Any]:
    """Mede o forward otimizado usado pelo treino, sem geracao autoregressiva."""

    executor.eval()
    quantidade = min(configuracao.tamanho_lote, len(corpus.exemplos))
    indices = list(range(quantidade))
    entrada, labels = corpus.lote(
        indices,
        dispositivo,
        comprimento_alvo=comprimento_balde(
            corpus,
            indices,
            configuracao.usar_baldes,
        ),
        fixar_memoria=configuracao.usar_prefetch,
    )
    mascara = labels.ne(-100)
    for _ in range(3):
        executor(entrada, mascara)
    medidas = []
    for _ in range(10):
        torch.cuda.synchronize()
        inicio = time.perf_counter()
        executor(entrada, mascara)
        torch.cuda.synchronize()
        medidas.append(time.perf_counter() - inicio)
    taxas_entrada = [entrada.numel() / duracao for duracao in medidas]
    supervisionados = int(mascara.sum())
    taxas_alvo = [supervisionados / duracao for duracao in medidas]
    return {
        "tokens_entrada_por_segundo_mediana": statistics.median(taxas_entrada),
        "tokens_entrada_por_segundo_media": statistics.fmean(taxas_entrada),
        "tokens_alvo_por_segundo_mediana": statistics.median(taxas_alvo),
        "repeticoes": len(medidas),
        "tokens_entrada_por_repeticao": entrada.numel(),
        "tokens_alvo_por_repeticao": supervisionados,
        "projecao_logits": (
            "somente_alvos" if executor.somente_alvos else "completa"
        ),
    }


PROMPTS_AVALIACAO_GERACAO = (
    "Explique de forma simples o que e inteligencia artificial.",
    "Como organizar uma rotina de estudos?",
    "Explique como funciona a fotossintese.",
    "Compare energia solar e energia eolica.",
    "Escreva uma funcao Python que some os numeros pares de uma lista.",
    "De tres sugestoes para melhorar a concentracao.",
    "Resuma a importancia da agua para a vida.",
    "Crie uma historia curta sobre uma biblioteca abandonada.",
    "Explique a diferenca entre memoria RAM e armazenamento.",
    "Como preparar uma apresentacao clara e objetiva?",
    "Liste cuidados basicos para proteger uma conta online.",
    "Explique por que o ceu parece azul durante o dia.",
)


def gerar_amostras(modelo_treino, tokenizador, configuracao, dispositivo) -> dict[str, Any]:
    """Compara greedy puro e amostragem controlada nos mesmos prompts."""

    runtime = ModeloGeradorEsparsoV73(
        tokenizador.tamanho,
        tokenizador.pad_id,
        modelo_treino.configuracao,
        SEMENTE_TOPOLOGIA_V73,
        modelo_treino.configuracao_roteamento,
        ConfiguracaoRuntimeCondicionalV73(implementacao="hibrido_agrupado"),
        ConfiguracaoRuntimeV73(exigir_kernel_cuda=False),
    ).to(dispositivo)
    runtime.load_state_dict(modelo_treino.state_dict(), strict=True)
    runtime.eval()
    prompts = PROMPTS_AVALIACAO_GERACAO[: configuracao.quantidade_prompts_geracao]
    configuracoes = {
        "greedy": ConfiguracaoDecodificacaoV73.greedy_puro(),
        "amostragem": ConfiguracaoDecodificacaoV73(
            modo="amostragem",
            temperatura=0.8,
            top_k=40,
            top_p=0.90,
            penalidade_repeticao=1.15,
            ngrama_sem_repeticao=3,
            maximo_repeticao_consecutiva=3,
            semente=configuracao.semente,
        ),
    }
    modos: dict[str, Any] = {}
    for nome, configuracao_decodificacao in configuracoes.items():
        amostras = []
        for indice, prompt in enumerate(prompts):
            prefixo = f"usuario: {prompt}\nassistente: "
            ids = tokenizador.codificar(prefixo, bos=True, eos=False)
            entrada = torch.tensor([ids], dtype=torch.long, device=dispositivo)
            configuracao_amostra = ConfiguracaoDecodificacaoV73(
                **{
                    **asdict(configuracao_decodificacao),
                    "semente": configuracao_decodificacao.semente + indice,
                }
            )
            with torch.inference_mode():
                gerados, medidas = gerar_v73(
                    runtime,
                    entrada,
                    configuracao.maximo_tokens_geracao,
                    tokenizador.eos_id,
                    configuracao_amostra,
                )
            amostras.append({
                "prompt": prompt,
                "texto": tokenizador.decodificar(gerados[0].tolist()),
                **medidas,
            })
        modos[nome] = {
            "configuracao": asdict(configuracao_decodificacao),
            "amostras": amostras,
            "resumo": resumir_metricas_geracao(amostras),
        }
    return {
        "prompts": len(prompts),
        "modos": modos,
    }


def estado_checkpoint(
    modelo,
    otimizador,
    agendador,
    epoca,
    configuracao,
    arquitetura,
    roteamento,
    treino,
    hashes_dataset,
    tokenizador: TokenizadorTextual,
    elegivel_comparacao: bool,
) -> dict[str, Any]:
    """Serializa estado treinavel, configuracao e apenas hashes dos dados."""

    return {
        "modelo": "gerador-esparso-v7-textual-experimental",
        "versao": VERSAO_PROJETO,
        "status": "experimental_nao_promovido",
        "epoca": epoca,
        "epocas_planejadas": configuracao.epocas,
        # O loader usa esta seed para reconstruir os indices COO nao
        # persistentes. Gravar a seed do shuffle aqui reconectaria os pesos.
        "semente": SEMENTE_TOPOLOGIA_V73,
        "semente_experimento": configuracao.semente,
        "configuracao": asdict(arquitetura),
        "configuracao_roteamento": asdict(roteamento),
        "configuracao_treino": asdict(treino),
        "configuracao_experimento": asdict(configuracao),
        "tokenizador": (
            {
                "tipo": tokenizador.TIPO,
                "serializado": tokenizador.serializado,
            }
            if isinstance(tokenizador, TokenizadorBPECodigo)
            else {
                "tipo": tokenizador.TIPO,
                "mesclas": (
                    tokenizador.mesclas
                    if isinstance(tokenizador, TokenizadorBPEBytes)
                    else None
                ),
            }
        ),
        "vocabulario": tokenizador.id_para_token,
        "estado_modelo": modelo.state_dict(),
        "topologia": metadados_topologia(modelo),
        "estado_otimizador": otimizador.state_dict(),
        "estado_agendador": agendador.state_dict(),
        "dataset": {
            "treino": "treino.jsonl",
            "validacao": "validacao.jsonl",
            "sha256": hashes_dataset,
        },
        "elegivel_comparacao": elegivel_comparacao,
        "elegivel_promocao": False,
        "checkpoint_base_preservado": True,
    }


def executar(args: argparse.Namespace) -> Path:
    """Executa cinco epocas e devolve o relatorio consolidado."""

    configuracao = ConfiguracaoExperimento(
        epocas=args.epocas,
        passos_por_epoca=args.passos_por_epoca,
        tamanho_lote=args.lote,
        maximo_contexto=args.contexto,
        taxa_aprendizado=args.taxa,
        warmup_passos=args.warmup,
        fator_minimo_lr=args.fator_minimo_lr,
        maximo_tokens_geracao=args.max_tokens_geracao,
        tipo_tokenizador=args.tokenizador,
        tamanho_vocabulario_bpe=args.tamanho_vocabulario_bpe,
        implementacao_treino=args.backend_treino,
        compilacao=args.compilacao,
        usar_prefetch=args.prefetch,
        usar_baldes=args.baldes,
        adamw_fundido=args.adamw_fundido,
        projecao_logits=args.projecao_logits,
        peso_eos=args.peso_eos,
        peso_repeticao=args.peso_repeticao,
        janela_repeticao=args.janela_repeticao,
        quantidade_prompts_geracao=args.quantidade_prompts_geracao,
        arquitetura_modelo=args.arquitetura_modelo,
        dimensao_modelo=args.dimensao_modelo,
        camadas_modelo=args.camadas_modelo,
        top_k_atencao_modelo=args.top_k_atencao_modelo,
        fan_in_qk_modelo=args.fan_in_qk_modelo,
        especialistas_modelo=args.especialistas_modelo,
        fan_in_ffn_modelo=args.fan_in_ffn_modelo,
        fan_out_ffn_modelo=args.fan_out_ffn_modelo,
        top_k_ffn_modelo=args.top_k_ffn_modelo,
    )
    configuracao.validar()
    if not torch.cuda.is_available():
        raise RuntimeError("o experimento exige o ambiente CUDA existente")
    # Um novo ciclo nunca pode substituir checkpoints experimentais anteriores.
    if args.saida.exists() and any(args.saida.glob("epoca_*.pt")):
        raise FileExistsError(f"a saida ja contem checkpoints: {args.saida}")
    dispositivo = torch.device("cuda")
    random.seed(configuracao.semente)
    torch.manual_seed(configuracao.semente)
    torch.cuda.manual_seed_all(configuracao.semente)
    manifesto_path = args.dataset / "manifesto.json"
    if manifesto_path.exists():
        corpus_treino = CorpusShardsV7(
            args.dataset,
            "treino",
            configuracao.maximo_contexto,
        )
        corpus_validacao = CorpusShardsV7(
            args.dataset,
            "validacao",
            configuracao.maximo_contexto,
        )
        tokenizador = corpus_treino.tokenizador
        if tokenizador.tamanho != configuracao.tamanho_vocabulario_bpe:
            raise ValueError(
                "vocabulario solicitado diverge do manifesto de shards"
            )
        tokenizador_path = args.dataset / "tokenizador.json"
        hashes_dataset = {
            "manifesto.json": sha256_arquivo(manifesto_path),
            "tokenizador.json": sha256_arquivo(tokenizador_path),
        }
        arquivos_relativos = ["manifesto.json", "tokenizador.json"]
    else:
        treino_path = args.dataset / "treino.jsonl"
        validacao_path = args.dataset / "validacao.jsonl"
        hashes_dataset = {
            "treino.jsonl": sha256_arquivo(treino_path),
            "validacao.jsonl": sha256_arquivo(validacao_path),
        }
        registros_treino = carregar_registros(treino_path)
        registros_validacao = carregar_registros(validacao_path)
        tokenizador = criar_tokenizador(registros_treino, configuracao)
        corpus_treino = CorpusConversacionalV7(
            treino_path,
            tokenizador,
            configuracao.maximo_contexto,
            registros_treino,
        )
        corpus_validacao = CorpusConversacionalV7(
            validacao_path,
            tokenizador,
            configuracao.maximo_contexto,
            registros_validacao,
        )
        arquivos_relativos = ["treino.jsonl", "validacao.jsonl"]
    modelo, arquitetura, roteamento, treino = criar_modelos(
        tokenizador,
        configuracao.implementacao_treino,
        configuracao.arquitetura_modelo,
        configuracao,
    )
    transferencia = transferir_base(modelo, args.checkpoint_base, tokenizador)
    modelo = modelo.to(dispositivo)
    indices_calibracao = indices_epoca(
        len(corpus_treino.exemplos),
        configuracao,
        0,
    )[0]
    entrada_calibracao, labels_calibracao = corpus_treino.lote(
        indices_calibracao,
        dispositivo,
        comprimento_alvo=comprimento_balde(
            corpus_treino,
            indices_calibracao,
            configuracao.usar_baldes,
        ),
    )
    modo_projecao, auditoria_projecao = selecionar_projecao_logits(
        modelo,
        entrada_calibracao,
        labels_calibracao,
        configuracao.projecao_logits,
    )
    del entrada_calibracao, labels_calibracao
    executor = ExecutorComFallbackV7(
        modelo,
        configuracao.compilacao,
        somente_alvos=modo_projecao == "somente_alvos",
    )
    otimizador, adamw_fundido_ativo, erro_adamw_fundido = criar_otimizador(
        modelo,
        configuracao,
        solicitar_fundido=configuracao.adamw_fundido,
    )
    passos_efetivos_por_epoca = len(
        indices_epoca(len(corpus_treino.exemplos), configuracao, 1)
    )
    total_passos = configuracao.epocas * passos_efetivos_por_epoca
    elegivel_comparacao = (
        configuracao.epocas == 5
        and configuracao.passos_por_epoca is None
    )

    def fator_lr(passo: int) -> float:
        if passo < configuracao.warmup_passos:
            return (passo + 1) / max(configuracao.warmup_passos, 1)
        progresso = (passo - configuracao.warmup_passos) / max(
            total_passos - configuracao.warmup_passos, 1
        )
        piso = configuracao.fator_minimo_lr
        return piso + (1.0 - piso) * 0.5 * (1.0 + math.cos(math.pi * progresso))

    agendador = torch.optim.lr_scheduler.LambdaLR(otimizador, fator_lr)
    relatorio: dict[str, Any] = {
        "experimento": "v73_bpe_codigo_8192",
        "status": "em_execucao",
        "versao_projeto": VERSAO_PROJETO,
        "configuracao": asdict(configuracao),
        "sementes": {
            "topologia": SEMENTE_TOPOLOGIA_V73,
            "experimento": configuracao.semente,
        },
        "arquitetura": asdict(arquitetura),
        "auditoria_arquitetura": modelo.auditoria(),
        "roteamento": asdict(roteamento),
        "executor_treino": asdict(treino),
        "transferencia": transferencia,
        "dataset": {
            "treino": corpus_treino.estatisticas,
            "validacao": corpus_validacao.estatisticas,
            "arquivos_relativos": arquivos_relativos,
            "sha256": hashes_dataset,
        },
        "tokenizador": {
            "tipo": tokenizador.TIPO,
            "tamanho": tokenizador.tamanho,
            "mesclas": (
                len(tokenizador.mesclas)
                if isinstance(tokenizador, TokenizadorBPEBytes)
                else 0
            ),
        },
        "cobertura_treino": {
            "passos_efetivos_por_epoca": passos_efetivos_por_epoca,
            "epoca_integral": configuracao.passos_por_epoca is None,
            "segmentos_por_epoca": (
                len(corpus_treino.exemplos)
                if configuracao.passos_por_epoca is None
                else min(
                    len(corpus_treino.exemplos),
                    configuracao.passos_por_epoca
                    * configuracao.tamanho_lote,
                )
            ),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_mib": (
                torch.cuda.get_device_properties(0).total_memory / 1048576
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "precisao": "bf16",
        },
        "otimizacoes_treino": {
            "projecao_logits": auditoria_projecao,
            "backend": configuracao.implementacao_treino,
            "compilacao": executor.auditoria(),
            "prefetch_cpu": configuracao.usar_prefetch,
            "baldes_contexto": (
                list(BALDES_CONTEXTO) if configuracao.usar_baldes else []
            ),
            "adamw_fundido_solicitado": configuracao.adamw_fundido,
            "adamw_fundido_ativo": adamw_fundido_ativo,
            "adamw_fundido_fallback": erro_adamw_fundido,
        },
        "epocas": [],
        "checkpoint_base_preservado": True,
        "elegivel_comparacao": elegivel_comparacao,
        "elegivel_promocao": False,
    }
    relatorio_path = args.saida / "relatorio.json"
    salvar_atomico(relatorio, relatorio_path, json_=True)

    for epoca in range(1, configuracao.epocas + 1):
        executor.train()
        torch.cuda.reset_peak_memory_stats()
        inicio_epoca = time.perf_counter()
        perdas, perdas_lm, perdas_roteamento, perdas_repeticao = [], [], [], []
        tokens_processados = tokens_supervisionados = 0
        tokens_repeticao_penalizados = 0
        lotes_epoca = indices_epoca(
            len(corpus_treino.exemplos), configuracao, epoca
        )
        lotes_carregados = iterar_lotes(
            corpus_treino,
            lotes_epoca,
            dispositivo,
            usar_prefetch=configuracao.usar_prefetch,
            usar_baldes=configuracao.usar_baldes,
        )
        for passo, (entrada, labels) in enumerate(lotes_carregados, 1):
            otimizador.zero_grad(set_to_none=True)
            mascara_alvos = labels.ne(-100)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, auxiliares = executor(entrada, mascara_alvos)
                perda_lm, quantidade_alvos = perda_resposta(
                    logits,
                    labels,
                    eos_id=tokenizador.eos_id,
                    peso_eos=configuracao.peso_eos,
                )
                if configuracao.peso_repeticao > 0:
                    perda_repeticao, quantidade_repeticao = (
                        perda_repeticao_incorreta(
                            logits,
                            labels,
                            janela=configuracao.janela_repeticao,
                        )
                    )
                else:
                    perda_repeticao = logits.new_zeros(())
                    quantidade_repeticao = 0
                perda = (
                    perda_lm
                    + auxiliares["perda_roteamento"]
                    + configuracao.peso_repeticao * perda_repeticao
                )
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), configuracao.gradiente_maximo)
            otimizador.step()
            agendador.step()
            perdas.append(float(perda.detach()))
            perdas_lm.append(float(perda_lm.detach()))
            perdas_roteamento.append(float(auxiliares["perda_roteamento"].detach()))
            perdas_repeticao.append(float(perda_repeticao.detach()))
            tokens_processados += entrada.numel()
            tokens_supervisionados += quantidade_alvos
            tokens_repeticao_penalizados += quantidade_repeticao
            if passo % 100 == 0:
                print(
                    f"epoca={epoca} passo={passo}/{len(lotes_epoca)} "
                    f"loss={statistics.fmean(perdas[-100:]):.4f}", flush=True
                )
        torch.cuda.synchronize()
        duracao_treino = time.perf_counter() - inicio_epoca
        pico_vram = torch.cuda.max_memory_allocated() / 1048576
        validacao = avaliar(
            executor, corpus_validacao, configuracao, dispositivo
        )
        forward = benchmark_forward(
            executor, corpus_validacao, configuracao, dispositivo
        )
        geracao = gerar_amostras(modelo, tokenizador, configuracao, dispositivo)
        checkpoint_path = args.saida / f"epoca_{epoca:02d}.pt"
        salvar_atomico(
            estado_checkpoint(
                modelo, otimizador, agendador, epoca, configuracao,
                arquitetura, roteamento, treino, hashes_dataset,
                tokenizador, elegivel_comparacao,
            ),
            checkpoint_path,
            json_=False,
        )
        registro = {
            "epoca": epoca,
            "passos": len(lotes_epoca),
            "treino": {
                "loss_total_media": statistics.fmean(perdas),
                "loss_resposta_media": statistics.fmean(perdas_lm),
                "loss_roteamento_media": statistics.fmean(perdas_roteamento),
                "loss_repeticao_media": statistics.fmean(perdas_repeticao),
                "tokens_repeticao_penalizados": tokens_repeticao_penalizados,
                "duracao_segundos": duracao_treino,
                "tokens_processados": tokens_processados,
                "tokens_supervisionados": tokens_supervisionados,
                "tokens_processados_por_segundo": tokens_processados / duracao_treino,
                "vram_pico_mib": pico_vram,
                "taxa_final": otimizador.param_groups[0]["lr"],
            },
            "validacao": validacao,
            "forward": forward,
            "geracao_autorregressiva": geracao,
            "roteamento": {
                "rotas_unicas_por_camada": [
                    int(torch.unique(mascara).numel())
                    for mascara in modelo.mascaras_ultima_execucao()
                ]
            },
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256_arquivo(checkpoint_path),
        }
        relatorio["epocas"].append(registro)
        relatorio["otimizacoes_treino"]["compilacao"] = executor.auditoria()
        salvar_atomico(relatorio, relatorio_path, json_=True)
        print(
            f"epoca={epoca} concluida ppl={validacao['ppl_token']:.4f} "
            f"treino_tok_s={registro['treino']['tokens_processados_por_segundo']:.2f}",
            flush=True,
        )

    relatorio["status"] = "concluido"
    relatorio["melhor_epoca_ppl"] = min(
        relatorio["epocas"], key=lambda item: item["validacao"]["ppl_token"]
    )["epoca"]
    relatorio["duracao_total_treino_segundos"] = sum(
        item["treino"]["duracao_segundos"] for item in relatorio["epocas"]
    )
    salvar_atomico(relatorio, relatorio_path, json_=True)
    return relatorio_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--checkpoint-base", type=Path, default=CHECKPOINT_BASE)
    parser.add_argument(
        "--arquitetura-modelo", choices=("v73",), default="v73"
    )
    parser.add_argument("--dimensao-modelo", type=int, default=128)
    parser.add_argument("--camadas-modelo", type=int, default=3)
    parser.add_argument("--top-k-atencao-modelo", type=int, default=32)
    parser.add_argument(
        "--fan-in-qk-modelo",
        "--fan-in-qkvo",
        dest="fan_in_qk_modelo",
        type=int,
        default=32,
        help="fan-in das projecoes Q/K COO",
    )
    parser.add_argument("--especialistas-modelo", type=int, default=384)
    parser.add_argument("--fan-in-ffn-modelo", type=int, default=32)
    parser.add_argument("--fan-out-ffn-modelo", type=int, default=48)
    parser.add_argument("--top-k-ffn-modelo", type=int, default=64)
    parser.add_argument("--epocas", type=int, default=5)
    parser.add_argument(
        "--passos-por-epoca",
        type=int,
        default=None,
        help="limite experimental; omitido percorre todos os segmentos",
    )
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--contexto", type=int, default=640)
    parser.add_argument("--taxa", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--fator-minimo-lr", type=float, default=0.1)
    parser.add_argument("--max-tokens-geracao", type=int, default=256)
    parser.add_argument(
        "--tokenizador",
        choices=("bpe_codigo",),
        default="bpe_codigo",
    )
    parser.add_argument(
        "--tamanho-vocabulario-bpe",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--backend-treino",
        choices=(
            "coo_agregado",
            "microblocos_densos",
            "banco_denso_completo",
        ),
        default="banco_denso_completo",
    )
    parser.add_argument(
        "--compilacao",
        choices=("desligada", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--baldes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adamw-fundido",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--projecao-logits",
        choices=("auto", "completa", "somente_alvos"),
        default="auto",
    )
    parser.add_argument("--peso-eos", type=float, default=1.0)
    parser.add_argument("--peso-repeticao", type=float, default=0.0)
    parser.add_argument("--janela-repeticao", type=int, default=4)
    parser.add_argument("--quantidade-prompts-geracao", type=int, default=8)
    args = parser.parse_args()
    print(executar(args))


if __name__ == "__main__":
    main()
