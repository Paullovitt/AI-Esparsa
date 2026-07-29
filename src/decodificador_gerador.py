"""Decodificacao causal com bloqueio de repeticao e encerramento de frase."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class TokenizadorDecodificacao(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int
    id_para_token: tuple[str, ...]

    def codificar(self, texto: str, *, bos: bool = True, eos: bool = True) -> list[int]: ...
    def decodificar(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class ConfiguracaoDecodificacao:
    maximo_tokens: int = 36
    minimo_tokens_frase: int = 2
    ngrama_bloqueado: int = 2
    penalidade_repeticao: float = 1.25
    janela_penalidade: int = 8
    temperatura: float = 0.75
    top_k_amostragem: int = 8
    parar_apos_frase: bool = True
    frases_para_encerrar: int = 1
    bloquear_trigrama_adicional: bool = True


def bloquear_proximo_ngram_repetido(
    logits: Tensor,
    continuacao: list[int],
    tamanho: int,
) -> None:
    """Bloqueia candidatos que repetiriam um n-grama ja gerado."""
    if tamanho < 2 or len(continuacao) < tamanho - 1:
        return
    prefixo = tuple(continuacao[-(tamanho - 1) :])
    limite = len(continuacao) - tamanho + 1
    for inicio in range(max(0, limite)):
        if tuple(continuacao[inicio : inicio + tamanho - 1]) == prefixo:
            logits[continuacao[inicio + tamanho - 1]] = -torch.inf


def aplicar_controle_repeticao(
    logits: Tensor,
    continuacao: list[int],
    configuracao: ConfiguracaoDecodificacao,
) -> None:
    if continuacao:
        for token_id in set(continuacao[-configuracao.janela_penalidade :]):
            logits[token_id] -= configuracao.penalidade_repeticao
    if configuracao.ngrama_bloqueado >= 2:
        bloquear_proximo_ngram_repetido(
            logits,
            continuacao,
            configuracao.ngrama_bloqueado,
        )
    if (
        configuracao.bloquear_trigrama_adicional
        and configuracao.ngrama_bloqueado != 3
    ):
        bloquear_proximo_ngram_repetido(logits, continuacao, 3)


def _encerrou_frase(
    tokenizador: TokenizadorDecodificacao,
    continuacao: list[int],
    configuracao: ConfiguracaoDecodificacao,
) -> bool:
    if not configuracao.parar_apos_frase:
        return False
    if len(continuacao) < configuracao.minimo_tokens_frase:
        return False
    token = tokenizador.id_para_token[continuacao[-1]]
    if token not in {".", "!", "?"}:
        return False
    terminadores = sum(
        tokenizador.id_para_token[token_id] in {".", "!", "?"}
        for token_id in continuacao
    )
    return terminadores >= max(1, configuracao.frases_para_encerrar)


@torch.inference_mode()
def gerar_controlado(
    modelo: torch.nn.Module,
    tokenizador: TokenizadorDecodificacao,
    prompt: str,
    dispositivo: torch.device,
    *,
    configuracao: ConfiguracaoDecodificacao = ConfiguracaoDecodificacao(),
    amostrar: bool = False,
    semente: int = 20260807,
    metricas_desempenho: dict[str, float] | None = None,
    usar_cache_incremental: bool = True,
) -> str:
    """Gera uma continuacao sem repetir n-gramas e encerra na primeira frase."""
    modelo.eval()
    ids = tokenizador.codificar(prompt, eos=False)
    continuacao: list[int] = []
    gerador = torch.Generator(device=dispositivo).manual_seed(semente)
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)
    inicio = time.perf_counter()
    instante_primeiro_token: float | None = None
    cache: object | None = None
    logits_cache: Tensor | None = None
    iniciar_cache = getattr(modelo, "iniciar_cache_geracao", None)
    avancar_cache = getattr(modelo, "avancar_cache_geracao", None)
    cache_disponivel = (
        usar_cache_incremental
        and callable(iniciar_cache)
        and callable(avancar_cache)
    )
    if cache_disponivel and configuracao.maximo_tokens > 0:
        entrada_inicial = torch.tensor(
            [ids[-modelo.configuracao.maximo_contexto :]],
            dtype=torch.long,
            device=dispositivo,
        )
        logits_cache, cache = iniciar_cache(entrada_inicial)

    for _ in range(configuracao.maximo_tokens):
        if cache_disponivel:
            if logits_cache is None or cache is None:
                raise RuntimeError("cache incremental nao foi inicializado")
            logits = logits_cache
        else:
            entrada = torch.tensor(
                [ids[-modelo.configuracao.maximo_contexto :]],
                dtype=torch.long,
                device=dispositivo,
            )
            logits, _ = modelo(entrada)
        atual = logits[0, -1].float().clone()
        atual[tokenizador.pad_id] = -torch.inf
        atual[tokenizador.bos_id] = -torch.inf
        aplicar_controle_repeticao(atual, continuacao, configuracao)

        if amostrar:
            quantidade = min(configuracao.top_k_amostragem, atual.numel())
            valores, indices = atual.topk(quantidade)
            probabilidades = torch.softmax(valores / configuracao.temperatura, dim=-1)
            proximo = int(indices[torch.multinomial(probabilidades, 1, generator=gerador)])
        else:
            proximo = int(atual.argmax())

        ids.append(proximo)
        continuacao.append(proximo)
        if instante_primeiro_token is None:
            if dispositivo.type == "cuda":
                torch.cuda.synchronize(dispositivo)
            instante_primeiro_token = time.perf_counter()
        if proximo == tokenizador.eos_id or _encerrou_frase(
            tokenizador,
            continuacao,
            configuracao,
        ):
            break
        if cache_disponivel:
            token_incremental = torch.tensor(
                [[proximo]],
                dtype=torch.long,
                device=dispositivo,
            )
            logits_cache, cache = avancar_cache(
                token_incremental,
                cache,
            )

    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)
    fim = time.perf_counter()
    if metricas_desempenho is not None:
        duracao = max(fim - inicio, 1e-12)
        quantidade = len(continuacao)
        metricas_desempenho.update(
            {
                "tokens_gerados": float(quantidade),
                "tempo_total_segundos": duracao,
                "latencia_primeiro_token_ms": (
                    (
                        (instante_primeiro_token or fim) - inicio
                    )
                    * 1000.0
                ),
                "latencia_media_token_ms": (
                    duracao * 1000.0 / max(1, quantidade)
                ),
                "tokens_por_segundo": quantidade / duracao,
                "cache_incremental_utilizado": (
                    1.0 if cache_disponivel else 0.0
                ),
            }
        )
    return tokenizador.decodificar(ids)
