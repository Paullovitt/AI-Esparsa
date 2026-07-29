"""Executa a V7.3 com caminho incremental fundido em CUDA.

O checkpoint continua contendo somente os 165.443 parâmetros da arquitetura
combinatória. O binário CUDA é compilado no cache local e nunca é persistido.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from executar_gerador_esparso import PROMPT_PADRAO
from src.kernel_cuda_v73 import carregar_kernel_cuda_v73
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v73 import (
    ConfiguracaoRuntimeV73,
    ModeloGeradorEsparsoV73,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.runtime_condicional_v73 import ConfiguracaoRuntimeCondicionalV73
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_gerador_esparso import (
    benchmark_autorregressivo,
    gerar_relato_validado,
    validar_prompt_publico,
)


RAIZ = Path(__file__).resolve().parent
MODELO_CHECKPOINT = "gerador-esparso-v73-base"
MODELOS_V73_COMPATIVEIS = {
    MODELO_CHECKPOINT,
    "gerador-esparso-v73-treino-fp32",
}
CHECKPOINT_PADRAO = (
    RAIZ
    / "modelos"
    / "gerador_esparso_v73_base.pt"
)


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
        raise ValueError("checkpoint nao pertence à V7.3")
    epoca = int(checkpoint["epoca"])
    epocas_planejadas = int(checkpoint.get("epocas_planejadas", 5))
    base_historica = (
        checkpoint.get("modelo") == MODELO_CHECKPOINT
        and epocas_planejadas == 2
        and epoca in {1, 2}
    )
    candidato_controlado = epocas_planejadas == 5 and 1 <= epoca <= 5
    if not (base_historica or candidato_controlado):
        raise ValueError(
            "checkpoint V7.3 fora da base historica ou do ciclo de cinco "
            "epocas"
        )
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
    # A compilação/cache não faz parte da latência de geração reportada.
    carregar_kernel_cuda_v73(obrigatorio=exigir_kernel_cuda)
    return modelo, tokenizador, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PADRAO)
    parser.add_argument("--prompt", type=str, default=PROMPT_PADRAO)
    parser.add_argument(
        "--permitir-fallback",
        action="store_true",
        help="permite o runtime PyTorch quando o kernel não compilar",
    )
    args = parser.parse_args()
    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    modelo, tokenizador, _ = carregar_v73(
        args.checkpoint,
        dispositivo,
        exigir_kernel_cuda=not args.permitir_fallback,
    )
    campos = validar_prompt_publico(args.prompt, tokenizador)
    texto, medidas = gerar_relato_validado(
        modelo,
        tokenizador,
        args.prompt,
        campos,
        dispositivo,
    )
    # O validador pode executar retentativas; medimos separadamente uma
    # geração greedy completa para não somar tentativas à latência exibida.
    desempenho = benchmark_autorregressivo(
        modelo,
        tokenizador,
        args.prompt,
        dispositivo,
    )
    print(texto)
    print(
        "\n[V7.3] "
        f"{desempenho['tokens_por_segundo']:.2f} tokens/s; "
        "primeiro token="
        f"{desempenho['latencia_primeiro_token_ms']:.2f} ms; "
        f"cobertura={medidas['cobertura_palavras_chave']:.0%}; "
        f"retentativas={medidas['retentativas']}"
    )


if __name__ == "__main__":
    main()
