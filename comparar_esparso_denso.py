"""Treina e compara modelos esparso e denso sob o mesmo protocolo.

O experimento cria artefatos separados e nunca altera o checkpoint oficial.
Ambos os modelos usam os mesmos registros, tokenizador, cinco épocas, lote,
500 passos por época, otimizador, agenda de aprendizado e decodificador.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from src.modelo_gerador_denso import (
    ConfiguracaoGeradorDenso,
    ModeloGeradorDenso,
)
from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from src.relatorio_esparso_denso import (
    _protocolo,
    construir_comparacao,
    gerar_markdown_comparacao,
)
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from executar_gerador_esparso import carregar_gerador
from treinar_gerador_esparso import (
    FRASES_GERACAO,
    MINIMO_CARACTERES_GERACAO,
    PASSOS_POR_EPOCA_PADRAO,
    QUANTIDADES_PADRAO,
    SEMENTE,
    _agendamento_taxa,
    avaliar_geracao_livre,
    avaliar_linguagem,
    benchmark,
    benchmark_autorregressivo,
    codificar_registros,
    criar_lotes,
    gerar_divisoes_gerador,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS_PADRAO = (
    RAIZ / "resultados" / "comparacao_esparso_denso_50k"
)
RELATORIO_ESPARSO_OFICIAL = (
    RAIZ / "resultados" / "gerador_esparso_base_50k" / "relatorio.json"
)
CHECKPOINT_ESPARSO_OFICIAL = (
    RAIZ / "modelos" / "gerador_esparso_base.pt"
)
EPOCAS = 5
TAMANHO_LOTE = 100
PASSOS_POR_EPOCA = PASSOS_POR_EPOCA_PADRAO
TAXA_APRENDIZADO = 0.0018


def _sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _ambiente() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def _salvar_checkpoint(
    caminho: Path,
    *,
    nome_modelo: str,
    modelo: nn.Module,
    configuracao: object,
    tokenizador: TokenizadorPalavras,
    epoca: int,
    metricas: dict[str, float],
    auditoria: dict[str, object],
) -> None:
    """Salva somente valores compatíveis com ``weights_only=True``."""

    torch.save(
        {
            "modelo": nome_modelo,
            "versao": VERSAO_PROJETO,
            "experimento": "comparacao-esparso-denso-parametros",
            "epoca": epoca,
            "epocas_planejadas": EPOCAS,
            "semente": SEMENTE,
            "configuracao": asdict(configuracao),
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "metricas_validacao": metricas,
            "auditoria": auditoria,
            "dados_treino": QUANTIDADES_PADRAO[0],
            "tamanho_lote": TAMANHO_LOTE,
            "passos_por_epoca": PASSOS_POR_EPOCA,
            "checkpoint_oficial_preservado": True,
            "baseline_experimental": nome_modelo.endswith("densa"),
        },
        caminho,
    )


def _treinar_modelo(
    *,
    nome: str,
    nome_checkpoint: str,
    fabrica: Callable[[], nn.Module],
    configuracao: object,
    tokenizador: TokenizadorPalavras,
    codificados_treino: list[tuple[list[int], list[float]]],
    codificados_validacao: list[tuple[list[int], list[float]]],
    codificados_teste: list[tuple[list[int], list[float]]],
    registros_teste: list[dict[str, object]],
    diretorio: Path,
    dispositivo: torch.device,
) -> dict[str, object]:
    """Executa o protocolo idêntico e grava as cinco épocas."""

    diretorio.mkdir(parents=True, exist_ok=False)
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)
    modelo = fabrica().to(dispositivo)
    auditoria = modelo.auditoria()
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    total_passos = EPOCAS * PASSOS_POR_EPOCA
    aquecimento = min(150, max(10, total_passos // 20))
    agendador = torch.optim.lr_scheduler.LambdaLR(
        otimizador,
        lambda passo: _agendamento_taxa(
            passo,
            total_passos,
            aquecimento,
        ),
    )

    inicio_experimento = time.perf_counter()
    avaliacao_inicial = avaliar_linguagem(
        modelo,
        codificados_validacao,
        tokenizador.pad_id,
        TAMANHO_LOTE,
        dispositivo,
    )
    print(
        f"[{nome}] parametros={auditoria['parametros']} "
        f"PPL_inicial={avaliacao_inicial['ppl']:.3f}",
        flush=True,
    )

    historico: list[dict[str, object]] = []
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        torch.cuda.reset_peak_memory_stats(dispositivo)
        torch.cuda.synchronize(dispositivo)
        inicio_epoca = time.perf_counter()
        perda_ponderada_total = 0.0
        tokens_total = 0
        for passo, (entradas, alvos, pesos) in enumerate(
            criar_lotes(
                codificados_treino,
                tokenizador.pad_id,
                TAMANHO_LOTE,
                dispositivo,
                embaralhar=True,
                semente=SEMENTE + epoca,
            ),
            start=1,
        ):
            logits, _ = modelo(entradas)
            perdas = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                alvos.reshape(-1),
                reduction="none",
                ignore_index=tokenizador.pad_id,
            ).view_as(alvos)
            mascara = alvos.ne(tokenizador.pad_id)
            pesos_validos = pesos * mascara.to(pesos.dtype)
            perda = (
                (perdas * pesos_validos).sum()
                / pesos_validos.sum().clamp_min(1.0)
            )
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agendador.step()
            quantidade = int(mascara.sum())
            perda_ponderada_total += float(perda.detach()) * quantidade
            tokens_total += quantidade
            if passo % 75 == 0:
                print(
                    f"[{nome} epoca {epoca}/5] "
                    f"passo={passo}/{PASSOS_POR_EPOCA} "
                    f"perda={perda_ponderada_total/tokens_total:.4f} "
                    f"lr={agendador.get_last_lr()[0]:.6f}",
                    flush=True,
                )
        if passo != PASSOS_POR_EPOCA:
            raise RuntimeError(
                f"{nome}: passos divergentes {passo} != "
                f"{PASSOS_POR_EPOCA}"
            )

        validacao = avaliar_linguagem(
            modelo,
            codificados_validacao,
            tokenizador.pad_id,
            TAMANHO_LOTE,
            dispositivo,
        )
        torch.cuda.synchronize(dispositivo)
        registro = {
            "epoca": epoca,
            "perda_treino_ponderada": (
                perda_ponderada_total / tokens_total
            ),
            "validacao": validacao,
            "tempo_segundos": time.perf_counter() - inicio_epoca,
            "vram_pico_mib": (
                torch.cuda.max_memory_allocated(dispositivo) / 1024**2
            ),
        }
        historico.append(registro)
        _salvar_checkpoint(
            diretorio / f"epoca_{epoca:02d}.pt",
            nome_modelo=nome_checkpoint,
            modelo=modelo,
            configuracao=configuracao,
            tokenizador=tokenizador,
            epoca=epoca,
            metricas=validacao,
            auditoria=auditoria,
        )
        print(
            f"[{nome} epoca {epoca}/5] "
            f"PPL={validacao['ppl']:.4f} "
            f"acc={validacao['acuracia_token']:.2%} "
            f"tempo={registro['tempo_segundos']:.1f}s "
            f"VRAM={registro['vram_pico_mib']:.1f}MiB",
            flush=True,
        )

    avaliacao_final = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        TAMANHO_LOTE,
        dispositivo,
    )
    geracao = avaliar_geracao_livre(
        modelo,
        tokenizador,
        registros_teste,
        dispositivo,
        quantidade=24,
    )
    desempenho_forward = benchmark(
        modelo,
        tokenizador.tamanho,
        dispositivo,
    )
    desempenho_autorregressivo = benchmark_autorregressivo(
        modelo,
        tokenizador,
        str(registros_teste[0]["pedido"]),
        dispositivo,
    )
    checkpoint_final = diretorio / "epoca_05.pt"
    relatorio: dict[str, object] = {
        "experimento": f"comparacao_{nome}_50k",
        "versao": VERSAO_PROJETO,
        "autor": "Paulo Augusto",
        "ano": 2026,
        "papel": (
            "modelo_oficial_retreinado_para_comparacao"
            if nome == "esparso"
            else "baseline_densa_experimental"
        ),
        "protocolo": _protocolo(),
        "ambiente": _ambiente(),
        "tokenizador": {
            "tipo": "palavras_e_pontuacao",
            "vocabulario": tokenizador.tamanho,
            "unk_nas_divisoes": 0,
        },
        "configuracao": asdict(configuracao),
        "arquitetura": auditoria,
        "avaliacao_inicial": avaliacao_inicial,
        "historico": historico,
        "avaliacao_final": avaliacao_final,
        "geracao_livre": geracao,
        "desempenho_forward": desempenho_forward,
        "desempenho_autorregressivo": desempenho_autorregressivo,
        "tempo_treino_total_segundos": sum(
            float(item["tempo_segundos"]) for item in historico
        ),
        "vram_treino_pico_mib": max(
            float(item["vram_pico_mib"]) for item in historico
        ),
        "tempo_experimento_total_segundos": (
            time.perf_counter() - inicio_experimento
        ),
        "checkpoint_final": str(checkpoint_final.resolve()),
        "checkpoint_final_bytes": checkpoint_final.stat().st_size,
        "checkpoint_final_sha256": _sha256(checkpoint_final),
        "checkpoint_oficial_modificado": False,
        "treino_reexecutado_nesta_rodada": True,
    }
    (diretorio / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{nome}] concluido PPL={avaliacao_final['ppl']:.4f} "
        f"geracao={geracao['taxa_aprovacao']:.2%}",
        flush=True,
    )

    del modelo, otimizador, agendador
    gc.collect()
    torch.cuda.empty_cache()
    return relatorio


def _carregar_referencia_esparsa(
    *,
    tokenizador: TokenizadorPalavras,
    registros_teste: list[dict[str, object]],
    diretorio: Path,
    dispositivo: torch.device,
) -> dict[str, object]:
    """Reusa o treino oficial equivalente e refaz benchmarks no código atual."""

    if (
        not RELATORIO_ESPARSO_OFICIAL.exists()
        or not CHECKPOINT_ESPARSO_OFICIAL.exists()
    ):
        raise FileNotFoundError("artefatos oficiais esparsos ausentes")
    oficial = json.loads(
        RELATORIO_ESPARSO_OFICIAL.read_text(encoding="utf-8")
    )
    modelo, tokenizador_checkpoint, _ = carregar_gerador(
        CHECKPOINT_ESPARSO_OFICIAL,
        dispositivo,
        classe_modelo=ModeloGeradorEsparsoV62,
    )
    if tokenizador_checkpoint.id_para_token != tokenizador.id_para_token:
        raise RuntimeError(
            "tokenizador oficial diverge do protocolo comparativo"
        )
    desempenho_forward = benchmark(
        modelo,
        tokenizador.tamanho,
        dispositivo,
    )
    desempenho_autorregressivo = benchmark_autorregressivo(
        modelo,
        tokenizador,
        str(registros_teste[0]["pedido"]),
        dispositivo,
    )
    historico = oficial["historico"]
    relatorio: dict[str, object] = {
        "experimento": "comparacao_esparso_50k",
        "versao": VERSAO_PROJETO,
        "autor": "Paulo Augusto",
        "ano": 2026,
        "papel": "modelo_oficial_com_treino_existente_equivalente",
        "protocolo": _protocolo(),
        "ambiente": oficial.get(
            "ambiente_revalidacao",
            _ambiente(),
        ),
        "tokenizador": oficial["tokenizador"],
        "configuracao": oficial["configuracao"],
        "arquitetura": modelo.auditoria(),
        "avaliacao_inicial": oficial["avaliacao_inicial"],
        "historico": historico,
        "avaliacao_final": oficial["avaliacao_final"],
        "geracao_livre": oficial["geracao_livre"],
        "desempenho_forward": desempenho_forward,
        "desempenho_autorregressivo": desempenho_autorregressivo,
        "tempo_treino_total_segundos": sum(
            float(item["tempo_segundos"]) for item in historico
        ),
        "vram_treino_pico_mib": max(
            float(item["vram_pico_mib"]) for item in historico
        ),
        "tempo_experimento_total_segundos": float(
            oficial["tempo_total_segundos"]
        ),
        "checkpoint_final": str(
            CHECKPOINT_ESPARSO_OFICIAL.resolve()
        ),
        "checkpoint_final_bytes": (
            CHECKPOINT_ESPARSO_OFICIAL.stat().st_size
        ),
        "checkpoint_final_sha256": _sha256(
            CHECKPOINT_ESPARSO_OFICIAL
        ),
        "checkpoint_oficial_modificado": False,
        "fonte_treino": str(
            RELATORIO_ESPARSO_OFICIAL.relative_to(RAIZ)
        ),
        "treino_reexecutado_nesta_rodada": False,
    }
    diretorio.mkdir(parents=True, exist_ok=False)
    (diretorio / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del modelo
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "[esparso] treino oficial equivalente reutilizado; "
        f"forward={desempenho_forward['tokens_por_segundo']:.1f} tok/s",
        flush=True,
    )
    return relatorio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTADOS_PADRAO,
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para a comparacao")
    if args.resultados.exists():
        raise FileExistsError(
            "o diretorio comparativo ja existe; escolha outro caminho"
        )
    if args.resultados.resolve().is_relative_to(
        (RAIZ / "modelos").resolve()
    ):
        raise ValueError("a comparacao nao pode gravar em modelos oficiais")

    torch.backends.cuda.matmul.allow_tf32 = True
    dispositivo = torch.device("cuda")
    treino, validacao, teste = gerar_divisoes_gerador(
        QUANTIDADES_PADRAO
    )
    tokenizador = TokenizadorPalavras(
        str(registro["texto"]) for registro in treino
    )
    codificados_treino = codificar_registros(treino, tokenizador)
    codificados_validacao = codificar_registros(validacao, tokenizador)
    codificados_teste = codificar_registros(teste, tokenizador)
    if any(
        tokenizador.unk_id in ids
        for divisao in (
            codificados_treino,
            codificados_validacao,
            codificados_teste,
        )
        for ids, _ in divisao
    ):
        raise RuntimeError("as divisoes possuem token desconhecido")

    configuracao_esparsa = ConfiguracaoGeradorEsparso()
    configuracao_densa = ConfiguracaoGeradorDenso()
    modelo_esparso_teste = ModeloGeradorEsparso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao_esparsa,
        SEMENTE,
    )
    modelo_denso_teste = ModeloGeradorDenso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao_densa,
        SEMENTE,
    )
    parametros_esparsos = sum(
        parametro.numel() for parametro in modelo_esparso_teste.parameters()
    )
    parametros_densos = sum(
        parametro.numel() for parametro in modelo_denso_teste.parameters()
    )
    diferenca = abs(parametros_esparsos / parametros_densos - 1.0)
    if diferenca > 0.01:
        raise RuntimeError(
            "baseline fora da tolerancia de 1% de parametros"
        )
    del modelo_esparso_teste, modelo_denso_teste

    args.resultados.mkdir(parents=True)
    (args.resultados / "protocolo.json").write_text(
        json.dumps(
            {
                **_protocolo(),
                "ambiente": _ambiente(),
                "parametros_planejados": {
                    "esparso": parametros_esparsos,
                    "denso": parametros_densos,
                    "diferenca_percentual": diferenca * 100.0,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    relatorio_esparso = _carregar_referencia_esparsa(
        tokenizador=tokenizador,
        registros_teste=teste,
        diretorio=args.resultados / "esparso",
        dispositivo=dispositivo,
    )
    relatorio_denso = _treinar_modelo(
        nome="denso",
        nome_checkpoint="baseline-comparativa-densa",
        fabrica=lambda: ModeloGeradorDenso(
            tokenizador.tamanho,
            tokenizador.pad_id,
            configuracao_densa,
            SEMENTE,
        ),
        configuracao=configuracao_densa,
        tokenizador=tokenizador,
        codificados_treino=codificados_treino,
        codificados_validacao=codificados_validacao,
        codificados_teste=codificados_teste,
        registros_teste=teste,
        diretorio=args.resultados / "denso",
        dispositivo=dispositivo,
    )
    comparacao = construir_comparacao(
        relatorio_esparso,
        relatorio_denso,
    )
    (args.resultados / "comparacao.json").write_text(
        json.dumps(comparacao, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.resultados / "COMPARACAO.md").write_text(
        gerar_markdown_comparacao(comparacao),
        encoding="utf-8",
    )
    print(
        json.dumps(comparacao, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
