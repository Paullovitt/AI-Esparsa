"""Treina e valida o gerador esparso-base condicionado por prompt.

O experimento usa cinco epocas, salva cada checkpoint separadamente e mede
PPL, geracao livre, recuperacao dos campos do pedido, velocidade e VRAM. O
treino-base cobre 50 mil relatos em exatamente 500 passos por epoca. Os
checkpoints oficiais nunca sao modificados por este script.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from src.avaliacao_linguagem import avaliar_linguagem
from src.benchmark_gerador import benchmark, benchmark_autorregressivo
from src.dados_gerador import (
    _indices_historia,
    codificar_registros,
    criar_lotes,
    gerar_divisoes_gerador,
    gerar_registro_extenso,
)
from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from src.validacao_gerador import (
    PADRAO_PEDIDO,
    _medir_continuacao,
    acoes_consistentes,
    avaliar_geracao_livre,
    concordancia_de_genero_consistente,
    extrair_campos_pedido,
    extrair_continuacao,
    frequencia_maxima_trigrama,
    gerar_relato_validado,
    local_do_objeto_consistente,
    problema_recuperado,
    taxa_trigramas_repetidos,
    termo_recuperado,
    validar_prompt_publico,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS_PADRAO = RAIZ / "resultados" / "gerador_esparso_base_50k"
SEMENTE = 20260728
QUANTIDADES_PADRAO = (50_000, 1_000, 1_000)
INICIOS_PADRAO = (40_000_000, 50_000_000, 60_000_000)
PASSOS_POR_EPOCA_PADRAO = 500
MINIMO_CARACTERES_GERACAO = 2_000
FRASES_GERACAO = 24


def salvar_checkpoint(
    caminho: Path,
    modelo: ModeloGeradorEsparso,
    configuracao: ConfiguracaoGeradorEsparso,
    tokenizador: TokenizadorPalavras,
    epoca: int,
    metricas: dict[str, float],
    *,
    tamanho_treino: int,
    tamanho_lote: int,
    passos_por_epoca: int,
) -> None:
    """Salva apenas tensores e metadados aceitos por ``weights_only``."""

    torch.save(
        {
            "modelo": "gerador-esparso-coerente-base",
            "versao": VERSAO_PROJETO,
            "epoca": epoca,
            "epocas_planejadas": 5,
            "semente": SEMENTE,
            "configuracao": asdict(configuracao),
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "metricas_validacao": metricas,
            "auditoria": modelo.auditoria(),
            "dados_treino": tamanho_treino,
            "tamanho_lote": tamanho_lote,
            "passos_por_epoca": passos_por_epoca,
            "checkpoint_oficial_preservado": True,
        },
        caminho,
    )


def _agendamento_taxa(
    passo: int,
    total_passos: int,
    aquecimento: int,
) -> float:
    if passo < aquecimento:
        return (passo + 1) / max(1, aquecimento)
    progresso = (passo - aquecimento) / max(
        1,
        total_passos - aquecimento,
    )
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progresso))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTADOS_PADRAO,
    )
    parser.add_argument("--epocas", type=int, default=5)
    parser.add_argument("--lote", type=int, default=100)
    parser.add_argument("--treino", type=int, default=50_000)
    parser.add_argument("--validacao", type=int, default=1_000)
    parser.add_argument("--teste", type=int, default=1_000)
    parser.add_argument(
        "--passos-por-epoca",
        type=int,
        default=PASSOS_POR_EPOCA_PADRAO,
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria")
    if args.epocas != 5:
        raise ValueError("os experimentos do projeto devem usar cinco epocas")
    if args.passos_por_epoca <= 0:
        raise ValueError("passos-por-epoca deve ser positivo")
    if args.treino != args.lote * args.passos_por_epoca:
        raise ValueError(
            "treino deve ser igual a lote x passos-por-epoca para que "
            "cada amostra seja processada exatamente uma vez por epoca"
        )
    if args.resultados.resolve() == (RAIZ / "modelos").resolve():
        raise ValueError("resultados nao podem apontar para modelos oficiais")
    existentes = list(args.resultados.glob("epoca_*.pt"))
    if existentes:
        raise FileExistsError(
            "a pasta ja possui checkpoints; use outro diretorio para "
            "nao sobrescrever experimentos"
        )

    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)
    torch.backends.cuda.matmul.allow_tf32 = True
    dispositivo = torch.device("cuda")

    treino, validacao, teste = gerar_divisoes_gerador(
        (args.treino, args.validacao, args.teste)
    )
    tokenizador = TokenizadorPalavras(
        str(registro["texto"]) for registro in treino
    )
    codificados_treino = codificar_registros(treino, tokenizador)
    codificados_validacao = codificar_registros(
        validacao,
        tokenizador,
    )
    codificados_teste = codificar_registros(teste, tokenizador)
    todos_ids = [
        ids
        for divisao in (
            codificados_treino,
            codificados_validacao,
            codificados_teste,
        )
        for ids, _ in divisao
    ]
    maior = max(map(len, todos_ids))
    configuracao = ConfiguracaoGeradorEsparso()
    if maior > configuracao.maximo_contexto + 1:
        raise RuntimeError(
            "sequencia excede o contexto planejado: "
            f"{maior} > {configuracao.maximo_contexto + 1}"
        )
    if any(
        tokenizador.unk_id in ids
        for ids in todos_ids
    ):
        raise RuntimeError("as divisoes possuem token desconhecido")

    modelo = ModeloGeradorEsparso(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        SEMENTE,
    ).to(dispositivo)
    auditoria = modelo.auditoria()
    if auditoria["lineares_densas_internas"] != 0:
        raise RuntimeError("foi introduzida uma linear densa interna")
    args.resultados.mkdir(parents=True, exist_ok=True)

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=0.0018,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    passos_epoca = math.ceil(len(codificados_treino) / args.lote)
    if passos_epoca != args.passos_por_epoca:
        raise RuntimeError(
            f"passos por epoca divergentes: {passos_epoca} != "
            f"{args.passos_por_epoca}"
        )
    total_passos = passos_epoca * 5
    aquecimento = min(150, max(10, total_passos // 20))
    agendador = torch.optim.lr_scheduler.LambdaLR(
        otimizador,
        lambda passo: _agendamento_taxa(
            passo,
            total_passos,
            aquecimento,
        ),
    )

    inicial = avaliar_linguagem(
        modelo,
        codificados_validacao,
        tokenizador.pad_id,
        args.lote,
        dispositivo,
    )
    print(
        f"[dados] treino={len(treino)} validacao={len(validacao)} "
        f"teste={len(teste)} vocab={tokenizador.tamanho} "
        f"max_tokens={maior} parametros={auditoria['parametros']}",
        flush=True,
    )
    print(
        f"[antes] PPL={inicial['ppl']:.3f} "
        f"acc={inicial['acuracia_token']:.2%}",
        flush=True,
    )

    historico: list[dict[str, object]] = []
    inicio_total = time.perf_counter()
    for epoca in range(1, 6):
        modelo.train()
        torch.cuda.reset_peak_memory_stats()
        inicio_epoca = time.perf_counter()
        perda_ponderada_total = 0.0
        tokens_total = 0
        for passo, (entradas, alvos, pesos) in enumerate(
            criar_lotes(
                codificados_treino,
                tokenizador.pad_id,
                args.lote,
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
                    f"[epoca {epoca}/5] passo={passo}/{passos_epoca} "
                    f"perda={perda_ponderada_total/tokens_total:.4f} "
                    f"lr={agendador.get_last_lr()[0]:.6f}",
                    flush=True,
                )

        validacao_epoca = avaliar_linguagem(
            modelo,
            codificados_validacao,
            tokenizador.pad_id,
            args.lote,
            dispositivo,
        )
        registro = {
            "epoca": epoca,
            "perda_treino_ponderada": (
                perda_ponderada_total / tokens_total
            ),
            "validacao": validacao_epoca,
            "tempo_segundos": time.perf_counter() - inicio_epoca,
            "vram_pico_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
            ),
        }
        historico.append(registro)
        salvar_checkpoint(
            args.resultados / f"epoca_{epoca:02d}.pt",
            modelo,
            configuracao,
            tokenizador,
            epoca,
            validacao_epoca,
            tamanho_treino=len(treino),
            tamanho_lote=args.lote,
            passos_por_epoca=passos_epoca,
        )
        print(
            f"[gerador] epoca {epoca}/5 "
            f"treino={registro['perda_treino_ponderada']:.4f} "
            f"PPL={validacao_epoca['ppl']:.3f} "
            f"acc={validacao_epoca['acuracia_token']:.2%} "
            f"tempo={registro['tempo_segundos']:.1f}s "
            f"VRAM={registro['vram_pico_mib']:.1f}MiB",
            flush=True,
        )

    avaliacao_final = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        args.lote,
        dispositivo,
    )
    geracao_livre = avaliar_geracao_livre(
        modelo,
        tokenizador,
        teste,
        dispositivo,
    )
    desempenho = benchmark(
        modelo,
        tokenizador.tamanho,
        dispositivo,
    )
    desempenho_autorregressivo = benchmark_autorregressivo(
        modelo,
        tokenizador,
        str(teste[0]["pedido"]),
        dispositivo,
    )
    relatorio = {
        "experimento": "gerador_esparso_base_50k",
        "versao": VERSAO_PROJETO,
        "autor": "Paulo Augusto",
        "ano": 2026,
        "objetivo": (
            "gerar relatos completos e coerentes preservando os campos "
            "fornecidos no pedido"
        ),
        "dados": {
            "treino": len(treino),
            "validacao": len(validacao),
            "teste": len(teste),
            "combinacoes_disjuntas": True,
            "sequencia_maxima": maior,
            "relatos_extensos": True,
            "frases_por_relato": FRASES_GERACAO,
        },
        "epocas": 5,
        "passos_por_epoca": passos_epoca,
        "passos_totais": passos_epoca * 5,
        "tamanho_lote": args.lote,
        "tokenizador": {
            "tipo": "palavras_e_pontuacao",
            "vocabulario": tokenizador.tamanho,
            "unk_nas_divisoes": 0,
        },
        "configuracao": asdict(configuracao),
        "arquitetura": auditoria,
        "avaliacao_inicial": inicial,
        "historico": historico,
        "avaliacao_final": avaliacao_final,
        "geracao_livre": geracao_livre,
        "recuperacao": {
            "campos_do_pedido": geracao_livre[
                "recuperacao_campos_pedido"
            ],
        },
        "desempenho_forward": desempenho,
        "desempenho_autorregressivo": desempenho_autorregressivo,
        "tempo_total_segundos": time.perf_counter() - inicio_total,
        "checkpoint_final": str(
            (args.resultados / "epoca_05.pt").resolve()
        ),
        "criterio_minimo_caracteres": MINIMO_CARACTERES_GERACAO,
        "limite": (
            "O resultado mede o dominio procedural treinado e nao comprova "
            "linguagem aberta geral."
        ),
    }
    (args.resultados / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "avaliacao_final": avaliacao_final,
                "geracao_livre": {
                    chave: valor
                    for chave, valor in geracao_livre.items()
                    if chave != "exemplos"
                },
                "desempenho_forward": desempenho,
                "desempenho_autorregressivo": (
                    desempenho_autorregressivo
                ),
                "exemplos": geracao_livre["exemplos"][:3],
                "relatorio": str(
                    (args.resultados / "relatorio.json").resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
