"""Executa a V7.3 textual sem depender do corpus de treinamento removido.

O executor faz geração greedy pura. Ele rejeita tokens desconhecidos para não
ocultar perda de informação atrás de ``<unk>`` e usa o cache causal da V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from src.kernel_cuda_v73 import carregar_kernel_cuda_v73
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v73 import (
    ConfiguracaoRuntimeV73,
    ModeloGeradorEsparsoV73,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.runtime_condicional_v73 import ConfiguracaoRuntimeCondicionalV73
from src.tokenizador_palavras import TokenizadorPalavras


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_PADRAO = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"
MODELOS_V73_COMPATIVEIS = {
    "gerador-esparso-v73-base",
    "gerador-esparso-v73-treino-fp32",
}


def carregar_v73(
    caminho: Path,
    dispositivo: torch.device,
    *,
    exigir_kernel_cuda: bool = True,
) -> tuple[ModeloGeradorEsparsoV73, TokenizadorPalavras, dict]:
    """Carrega estritamente um checkpoint V7.3 com ``weights_only``."""

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    if checkpoint.get("modelo") not in MODELOS_V73_COMPATIVEIS:
        raise ValueError("checkpoint nao pertence a V7.3")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    modelo = ModeloGeradorEsparsoV73(
        tokenizador.tamanho,
        tokenizador.pad_id,
        ConfiguracaoGeradorEsparso(**checkpoint["configuracao"]),
        int(checkpoint["semente"]),
        ConfiguracaoRoteamentoV73(
            **checkpoint["configuracao_roteamento"]
        ),
        ConfiguracaoRuntimeCondicionalV73(
            implementacao="hibrido_agrupado"
        ),
        ConfiguracaoRuntimeV73(
            exigir_kernel_cuda=exigir_kernel_cuda,
        ),
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()
    if exigir_kernel_cuda:
        carregar_kernel_cuda_v73(obrigatorio=True)
    return modelo, tokenizador, checkpoint


@torch.inference_mode()
def gerar_greedy_v73(
    modelo: ModeloGeradorEsparsoV73,
    tokens: torch.Tensor,
    maximo_novos_tokens: int,
    eos_id: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gera incrementalmente e mede latência percebida."""

    if maximo_novos_tokens <= 0:
        raise ValueError("maximo_novos_tokens deve ser positivo")
    inicio = time.perf_counter()
    logits, cache = modelo.iniciar_cache_geracao(tokens)
    primeiro = None
    gerados = tokens.clone()
    for _ in range(maximo_novos_tokens):
        proximo = logits[:, -1].argmax(dim=-1, keepdim=True)
        gerados = torch.cat((gerados, proximo), dim=1)
        if primeiro is None:
            primeiro = time.perf_counter()
        if bool(proximo.eq(eos_id).all()):
            break
        logits, cache = modelo.avancar_cache_geracao(proximo, cache)
    fim = time.perf_counter()
    novos = gerados.shape[1] - tokens.shape[1]
    duracao = max(fim - inicio, 1e-9)
    return gerados, {
        "tokens_por_segundo": novos / duracao,
        "latencia_primeiro_token_ms": (
            ((primeiro or fim) - inicio) * 1000.0
        ),
        "tokens_gerados": float(novos),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PADRAO)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--permitir-fallback",
        action="store_true",
        help="permite o runtime PyTorch quando o kernel CUDA nao compilar",
    )
    args = parser.parse_args()
    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    modelo, tokenizador, _ = carregar_v73(
        args.checkpoint,
        dispositivo,
        exigir_kernel_cuda=(
            dispositivo.type == "cuda" and not args.permitir_fallback
        ),
    )
    tokenizador.validar_texto_no_vocabulario(args.prompt)
    prefixo = tokenizador.codificar(args.prompt, bos=True, eos=False)
    tokens = torch.tensor([prefixo], device=dispositivo)
    gerados, medidas = gerar_greedy_v73(
        modelo,
        tokens,
        args.max_tokens,
        tokenizador.eos_id,
    )
    print(tokenizador.decodificar(gerados[0].tolist()))
    print(
        f"\n[V7.3] {medidas['tokens_por_segundo']:.2f} tokens/s; "
        "primeiro token="
        f"{medidas['latencia_primeiro_token_ms']:.2f} ms"
    )


if __name__ == "__main__":
    main()
