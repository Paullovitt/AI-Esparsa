"""Compara V6.2, V7.3 e denso com qualidade e benchmark auditáveis.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from executar_gerador_esparso import carregar_gerador
from executar_gerador_esparso_v73 import carregar_v73
from src.avaliacao_decodificacao import avaliar_geracao_greedy
from src.benchmark_estatistico import executar_benchmark_estatistico
from src.modelo_gerador_denso import (
    ConfiguracaoGeradorDenso,
    ModeloGeradorDenso,
)
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from src.relatorio_v73 import (
    gerar_exemplos,
    gerar_markdown,
    sincronizar_documentacao,
)
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_gerador_esparso import (
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    benchmark_autorregressivo,
    codificar_registros,
    gerar_divisoes_gerador,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados" / "v73_base"
CHECKPOINT_V62 = RAIZ / "modelos" / "gerador_esparso_base.pt"
CHECKPOINT_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"
CHECKPOINT_DENSO = (
    RAIZ
    / "resultados"
    / "comparacao_esparso_denso_50k"
    / "denso"
    / "epoca_05.pt"
)
METRICAS_TREINO_V73 = RESULTADOS / "treino_base.json"
RELATORIO_JSON = RESULTADOS / "comparacao.json"
RELATORIO_MARKDOWN = RESULTADOS / "RELATORIO_V73.md"
EXEMPLO_MARKDOWN = RESULTADOS / "EXEMPLOS_COMPARATIVOS.md"

# Compatibilidade com testes e integrações que já importavam estes nomes.
_gerar_markdown = gerar_markdown
_gerar_exemplos = gerar_exemplos


def _sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _liberar_gpu() -> None:
    """Libera referências para que cada pico de VRAM seja isolado."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _carregar_denso(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[ModeloGeradorDenso, TokenizadorPalavras, dict[str, Any]]:
    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    if (
        checkpoint.get("modelo") != "baseline-comparativa-densa"
        or int(checkpoint["epoca"]) != 5
    ):
        raise ValueError("checkpoint denso comparativo inválido")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    modelo = ModeloGeradorDenso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        ConfiguracaoGeradorDenso(**checkpoint["configuracao"]),
        int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["estado_modelo"], strict=True)
    modelo.eval()
    return modelo, tokenizador, checkpoint


def _medir(
    nome: str,
    modelo: torch.nn.Module,
    tokenizador: TokenizadorPalavras,
    checkpoint: dict[str, Any],
    caminho_checkpoint: Path,
    codificados_teste: list[tuple[list[int], list[float]]],
    teste: list[dict[str, object]],
    dispositivo: torch.device,
    *,
    repeticoes: int,
    aquecimentos: int,
) -> dict[str, object]:
    """Mede pesos, sistema controlado e desempenho estatístico."""

    inicio = time.perf_counter()
    linguagem = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        100,
        dispositivo,
    )
    geracao_modelo_puro = avaliar_geracao_greedy(
        modelo,
        tokenizador,
        teste,
        dispositivo,
        quantidade=24,
        restricoes=False,
    )
    geracao_greedy_restrita = avaliar_geracao_greedy(
        modelo,
        tokenizador,
        teste,
        dispositivo,
        quantidade=24,
        restricoes=True,
    )
    geracao_sistema = avaliar_geracao_livre(
        modelo,
        tokenizador,
        teste,
        dispositivo,
        quantidade=24,
    )
    forward = executar_benchmark_estatistico(
        lambda: benchmark(
            modelo,
            tokenizador.tamanho,
            dispositivo,
        ),
        repeticoes=repeticoes,
        aquecimentos=aquecimentos,
    )
    prompt_benchmark = str(teste[0]["pedido"])
    autorregressivo = executar_benchmark_estatistico(
        lambda: benchmark_autorregressivo(
            modelo,
            tokenizador,
            prompt_benchmark,
            dispositivo,
        ),
        repeticoes=repeticoes,
        aquecimentos=aquecimentos,
    )
    return {
        "nome": nome,
        "epoca": int(checkpoint["epoca"]),
        "epocas_planejadas": int(
            checkpoint.get("epocas_planejadas", checkpoint["epoca"])
        ),
        "identificador_checkpoint": checkpoint.get("modelo"),
        "parametros": sum(
            parametro.numel() for parametro in modelo.parameters()
        ),
        "avaliacao_teste": linguagem,
        "geracao_modelo_puro": geracao_modelo_puro,
        "geracao_greedy_restrita": geracao_greedy_restrita,
        "geracao_livre": geracao_sistema,
        "desempenho_forward": forward,
        "desempenho_autorregressivo": autorregressivo,
        "checkpoint_bytes": caminho_checkpoint.stat().st_size,
        "checkpoint_sha256": _sha256(caminho_checkpoint),
        "caminho_checkpoint": str(caminho_checkpoint.resolve()),
        "tempo_avaliacao_segundos": time.perf_counter() - inicio,
    }


def _argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara V6.2, um checkpoint V7.3 configurável e o controle denso."
        )
    )
    parser.add_argument("--checkpoint-v62", type=Path, default=CHECKPOINT_V62)
    parser.add_argument("--checkpoint-v73", type=Path, default=CHECKPOINT_V73)
    parser.add_argument(
        "--checkpoint-denso",
        type=Path,
        default=CHECKPOINT_DENSO,
    )
    parser.add_argument(
        "--metricas-treino-v73",
        type=Path,
        default=METRICAS_TREINO_V73,
    )
    parser.add_argument("--resultados", type=Path, default=RESULTADOS)
    parser.add_argument("--repeticoes", type=int, default=5)
    parser.add_argument("--aquecimentos", type=int, default=2)
    parser.add_argument(
        "--permitir-fallback-v73",
        action="store_true",
        help="permite medir a V7.3 sem o kernel CUDA compilado",
    )
    parser.add_argument(
        "--somente-documentos",
        action="store_true",
        help="não mede modelos; regenera Markdown e docs a partir do JSON",
    )
    parser.add_argument(
        "--nao-sincronizar-documentacao",
        action="store_true",
        help="útil ao avaliar um candidato ainda não promovido",
    )
    return parser.parse_args()


def _salvar_documentos(
    relatorio: dict[str, object],
    resultados: Path,
    *,
    sincronizar: bool,
) -> None:
    resultados.mkdir(parents=True, exist_ok=True)
    (resultados / "RELATORIO_V73.md").write_text(
        gerar_markdown(relatorio),
        encoding="utf-8",
    )
    (resultados / "EXEMPLOS_COMPARATIVOS.md").write_text(
        gerar_exemplos(relatorio),
        encoding="utf-8",
    )
    if sincronizar:
        sincronizar_documentacao(relatorio, RAIZ)


def main() -> None:
    args = _argumentos()
    relatorio_json = args.resultados / "comparacao.json"
    sincronizar = not args.nao_sincronizar_documentacao
    if args.somente_documentos:
        if not relatorio_json.exists():
            raise FileNotFoundError(relatorio_json)
        relatorio = json.loads(relatorio_json.read_text(encoding="utf-8"))
        _salvar_documentos(
            relatorio,
            args.resultados,
            sincronizar=sincronizar,
        )
        print(args.resultados / "RELATORIO_V73.md")
        return

    if args.repeticoes < 2:
        raise ValueError("--repeticoes deve ser pelo menos 2")
    if args.aquecimentos < 1:
        raise ValueError("--aquecimentos deve ser pelo menos 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA é obrigatória para a comparação completa")
    for caminho in (
        args.checkpoint_v62,
        args.checkpoint_v73,
        args.checkpoint_denso,
    ):
        if not caminho.exists():
            raise FileNotFoundError(caminho)

    dispositivo = torch.device("cuda")
    _, _, teste = gerar_divisoes_gerador()
    modelo_v62, tokenizador, checkpoint_v62 = carregar_gerador(
        args.checkpoint_v62,
        dispositivo,
        classe_modelo=ModeloGeradorEsparsoV62,
    )
    codificados = codificar_registros(teste, tokenizador)
    dados_v62 = _medir(
        "v62",
        modelo_v62,
        tokenizador,
        checkpoint_v62,
        args.checkpoint_v62,
        codificados,
        teste,
        dispositivo,
        repeticoes=args.repeticoes,
        aquecimentos=args.aquecimentos,
    )
    del modelo_v62
    _liberar_gpu()

    modelo_v73, tokenizador_v73, checkpoint_v73 = carregar_v73(
        args.checkpoint_v73,
        dispositivo,
        exigir_kernel_cuda=not args.permitir_fallback_v73,
    )
    if tokenizador_v73.id_para_token != tokenizador.id_para_token:
        raise RuntimeError("vocabulários V6.2 e V7.3 divergentes")
    dados_v73 = _medir(
        "v73",
        modelo_v73,
        tokenizador_v73,
        checkpoint_v73,
        args.checkpoint_v73,
        codificados,
        teste,
        dispositivo,
        repeticoes=args.repeticoes,
        aquecimentos=args.aquecimentos,
    )
    del modelo_v73
    _liberar_gpu()

    modelo_denso, tokenizador_denso, checkpoint_denso = _carregar_denso(
        args.checkpoint_denso,
        dispositivo,
    )
    if tokenizador_denso.id_para_token != tokenizador.id_para_token:
        raise RuntimeError("vocabulários V6.2 e denso divergentes")
    dados_denso = _medir(
        "denso",
        modelo_denso,
        tokenizador_denso,
        checkpoint_denso,
        args.checkpoint_denso,
        codificados,
        teste,
        dispositivo,
        repeticoes=args.repeticoes,
        aquecimentos=args.aquecimentos,
    )
    del modelo_denso
    _liberar_gpu()

    modelos = {
        "v62": dados_v62,
        "v73": dados_v73,
        "denso": dados_denso,
    }
    qualidade_integral = all(
        float(dados["geracao_livre"]["taxa_aprovacao"]) == 1.0
        for dados in modelos.values()
    )
    relatorio: dict[str, object] = {
        "experimento": "comparacao_bases_mantidas_auditavel",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "hardware": torch.cuda.get_device_name(0),
        "modelos_mantidos": ["v62", "v73", "denso"],
        "protocolo_benchmark": {
            "aquecimentos": args.aquecimentos,
            "repeticoes": args.repeticoes,
            "agregacao_principal": "mediana",
        },
        "treino_v73": (
            json.loads(args.metricas_treino_v73.read_text(encoding="utf-8"))
            if args.metricas_treino_v73.exists()
            else None
        ),
        "modelos": modelos,
        "conclusao": {
            "base_desenvolvimento": "v73",
            "base_estavel": "v62",
            "controle": "denso",
            "vantagem_geracao_v73_sobre_denso": (
                float(
                    dados_v73["desempenho_autorregressivo"][
                        "tokens_por_segundo"
                    ]
                )
                / float(
                    dados_denso["desempenho_autorregressivo"][
                        "tokens_por_segundo"
                    ]
                )
            ),
            "vantagem_geracao_v73_sobre_v62": (
                float(
                    dados_v73["desempenho_autorregressivo"][
                        "tokens_por_segundo"
                    ]
                )
                / float(
                    dados_v62["desempenho_autorregressivo"][
                        "tokens_por_segundo"
                    ]
                )
            ),
            "vantagem_forward_denso_sobre_v73": (
                float(
                    dados_denso["desempenho_forward"]["tokens_por_segundo"]
                )
                / float(
                    dados_v73["desempenho_forward"]["tokens_por_segundo"]
                )
            ),
            "qualidade_textual_integral": qualidade_integral,
            "v73_greedy_24": (
                float(
                    dados_v73["geracao_greedy_restrita"]["taxa_aprovacao"]
                )
                == 1.0
            ),
            "artefato_bf16_presente": False,
        },
        "observacoes": [
            "modelo puro e decodificador controlado foram medidos separados",
            "as 24 saídas de cada modo e modelo estão armazenadas no JSON",
            "o benchmark registra mediana, média, desvio-padrão e amostras",
            "clocks e energia são registrados quando nvidia-smi os fornece",
        ],
    }
    args.resultados.mkdir(parents=True, exist_ok=True)
    relatorio_json.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _salvar_documentos(
        relatorio,
        args.resultados,
        sincronizar=sincronizar,
    )
    print(args.resultados / "RELATORIO_V73.md")


if __name__ == "__main__":
    main()
