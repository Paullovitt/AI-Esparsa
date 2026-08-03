"""Executa o modelo V7.3 principal com tokenizador BPE de 8.192 tokens.

Oferece greedy puro para auditoria e amostragem controlada para uso interativo.
Rejeita tokens desconhecidos e usa o cache causal da V7.3.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from src.kernel_cuda_v73 import carregar_kernel_cuda_v73
from src.geracao_textual_v73 import (
    ConfiguracaoDecodificacaoV73,
    gerar_greedy_v73,
    gerar_v73,
)
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v73 import (
    ConfiguracaoRuntimeV73,
    ModeloGeradorEsparsoV73,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.runtime_condicional_v73 import ConfiguracaoRuntimeCondicionalV73
from src.tokenizador_palavras import tokenizador_de_checkpoint
from src.topologia_checkpoint_v73 import validar_topologia_checkpoint


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_PADRAO = (
    RAIZ / "modelos" / "gerador_esparso_v73_bpe8192_refino_eos3_ul005_topologia.pt"
)
MODELOS_V73_COMPATIVEIS = {
    "gerador-esparso-v73-base",
    "gerador-esparso-v73-treino-fp32",
    "gerador-esparso-v7-textual-experimental",
    # Compatibilidade somente para auditoria do primeiro ensaio byte-level.
    "gerador-esparso-v7-textual-byte-experimental",
}


def carregar_v73(
    caminho: Path,
    dispositivo: torch.device,
    *,
    exigir_kernel_cuda: bool = True,
    permitir_checkpoint_legado: bool = False,
) -> tuple[ModeloGeradorEsparsoV73, object, dict]:
    """Carrega estritamente um checkpoint V7.3 com ``weights_only``."""

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    if checkpoint.get("modelo") not in MODELOS_V73_COMPATIVEIS:
        raise ValueError("checkpoint nao pertence a V7.3")
    tokenizador = tokenizador_de_checkpoint(checkpoint)
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
    validar_topologia_checkpoint(
        checkpoint,
        modelo,
        permitir_legado=permitir_checkpoint_legado,
    )
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()
    if exigir_kernel_cuda:
        carregar_kernel_cuda_v73(obrigatorio=True)
    return modelo, tokenizador, checkpoint


def formatar_prompt(prompt: str, *, bruto: bool = False) -> str:
    """Aplica o mesmo contrato conversacional usado no treinamento."""

    if bruto:
        return prompt
    return f"usuario: {prompt}\nassistente: "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PADRAO)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--prompt-cru",
        action="store_true",
        help="nao aplica o template usuario/assistente do treinamento",
    )
    parser.add_argument(
        "--permitir-checkpoint-legado",
        action="store_true",
        help="aceita checkpoint sem hash de topologia (nao recomendado)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--decodificacao",
        choices=("greedy", "amostragem"),
        default="amostragem",
    )
    parser.add_argument("--temperatura", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--penalidade-repeticao", type=float, default=1.15)
    parser.add_argument("--ngrama-sem-repeticao", type=int, default=3)
    parser.add_argument("--max-repeticao-consecutiva", type=int, default=3)
    parser.add_argument("--semente-geracao", type=int, default=20260802)
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
        permitir_checkpoint_legado=args.permitir_checkpoint_legado,
    )
    texto_prompt = formatar_prompt(args.prompt, bruto=args.prompt_cru)
    tokenizador.validar_texto_no_vocabulario(texto_prompt)
    prefixo = tokenizador.codificar(texto_prompt, bos=True, eos=False)
    tokens = torch.tensor([prefixo], device=dispositivo)
    configuracao_geracao = ConfiguracaoDecodificacaoV73(
        modo=args.decodificacao,
        temperatura=args.temperatura,
        top_k=args.top_k,
        top_p=args.top_p,
        penalidade_repeticao=args.penalidade_repeticao,
        ngrama_sem_repeticao=args.ngrama_sem_repeticao,
        maximo_repeticao_consecutiva=args.max_repeticao_consecutiva,
        semente=args.semente_geracao,
    )
    gerados, medidas = gerar_v73(
        modelo,
        tokens,
        args.max_tokens,
        tokenizador.eos_id,
        configuracao_geracao,
    )
    novos_ids = gerados[0, tokens.shape[1] :].tolist()
    print(tokenizador.decodificar(novos_ids).strip())
    print(
        "\n[V7.3] "
        f"modo={configuracao_geracao.modo}; "
        f"{medidas['tokens_por_segundo']:.2f} tokens/s; "
        f"primeiro token={medidas['latencia_primeiro_token_ms']:.2f} ms"
    )
    print(json.dumps(medidas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
