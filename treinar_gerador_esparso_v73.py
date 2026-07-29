"""Treina a V7.3 em FP32 sem tocar nos checkpoints-base existentes.

O protocolo usa os mesmos 50 mil relatos, lote 100, 500 passos por época,
semente, AdamW e pesos da loss. O backend escolhido executa somente os
quatro microgrupos roteados, materializando seus pequenos pesos temporarios,
e vetoriza a atencao Top-32 em uma matriz causal completa durante o treino.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.treino_vetorizado_v73 import (
    ConfiguracaoTreinoV73,
    ModeloGeradorEsparsoTreinoV73,
)
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from treinar_gerador_esparso import (
    PASSOS_POR_EPOCA_PADRAO,
    QUANTIDADES_PADRAO,
    SEMENTE,
    _agendamento_taxa,
    avaliar_linguagem,
    codificar_registros,
    criar_lotes,
    gerar_divisoes_gerador,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS_PADRAO = (
    RAIZ / "resultados" / "v73_base" / "treino_fp32_5epocas"
)
EPOCAS_EXPERIMENTO = 5
TAMANHO_LOTE = 100
TAXA_APRENDIZADO = 0.0018
CONFIGURACAO_TREINO = ConfiguracaoTreinoV73(
    implementacao="microblocos_densos",
    atencao="matriz_densa_topk",
)


def salvar_checkpoint_v73(
    caminho: Path,
    modelo: ModeloGeradorEsparsoTreinoV73,
    tokenizador: TokenizadorPalavras,
    epoca: int,
    metricas: dict[str, float],
) -> None:
    """Salva apenas tipos seguros e nunca escreve no checkpoint oficial."""

    torch.save(
        {
            "modelo": "gerador-esparso-v73-treino-fp32",
            "versao": VERSAO_PROJETO,
            "experimento": "v73_treino_fp32_5epocas",
            "epoca": epoca,
            "epocas_planejadas": EPOCAS_EXPERIMENTO,
            "semente": SEMENTE,
            "configuracao": asdict(modelo.configuracao),
            "configuracao_roteamento": asdict(
                modelo.configuracao_roteamento
            ),
            "configuracao_treino": asdict(modelo.configuracao_treino),
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "metricas_validacao": metricas,
            "auditoria": modelo.auditoria(),
            "dados_treino": QUANTIDADES_PADRAO[0],
            "tamanho_lote": TAMANHO_LOTE,
            "passos_por_epoca": PASSOS_POR_EPOCA_PADRAO,
            "checkpoint_oficial_preservado": True,
            # A época final vira candidata, mas ainda depende da comparação
            # controlada e de uma promoção explícita.
            "elegivel_promocao": epoca == EPOCAS_EXPERIMENTO,
        },
        caminho,
    )


def _registrar(mensagem: str, caminho_log: Path) -> None:
    print(mensagem, flush=True)
    with caminho_log.open("a", encoding="utf-8") as arquivo:
        arquivo.write(mensagem + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTADOS_PADRAO,
    )
    parser.add_argument("--epocas", type=int, default=EPOCAS_EXPERIMENTO)
    args = parser.parse_args()
    if args.epocas != EPOCAS_EXPERIMENTO:
        raise ValueError("o treino-base V7.3 exige exatamente cinco epocas")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para o treino V7.3")
    if args.resultados.resolve() == (RAIZ / "modelos").resolve():
        raise ValueError("o treino V7.3 nao pode sobrescrever modelos")
    if list(args.resultados.glob("epoca_*.pt")):
        raise FileExistsError(
            "a pasta ja possui checkpoints V7.3; escolha outro destino"
        )

    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)
    torch.backends.cuda.matmul.allow_tf32 = True
    dispositivo = torch.device("cuda")
    args.resultados.mkdir(parents=True, exist_ok=True)
    caminho_log = args.resultados / "log_treino.txt"

    treino, validacao, _ = gerar_divisoes_gerador(QUANTIDADES_PADRAO)
    tokenizador = TokenizadorPalavras(
        str(registro["texto"]) for registro in treino
    )
    codificados_treino = codificar_registros(treino, tokenizador)
    codificados_validacao = codificar_registros(validacao, tokenizador)
    configuracao = ConfiguracaoGeradorEsparso()
    configuracao_roteamento = ConfiguracaoRoteamentoV73()
    modelo = ModeloGeradorEsparsoTreinoV73(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        SEMENTE,
        configuracao_roteamento,
        CONFIGURACAO_TREINO,
    ).to(dispositivo)
    auditoria = modelo.auditoria()
    if auditoria["checkpoint_oficial"]:
        raise RuntimeError("o treino V7.3 nao pode se declarar oficial")
    if not auditoria["roteamento_logico_somente_grupos_selecionados"]:
        raise RuntimeError("a mascara condicional da V7.3 foi perdida")

    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=TAXA_APRENDIZADO,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    total_passos = PASSOS_POR_EPOCA_PADRAO * EPOCAS_EXPERIMENTO
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
        TAMANHO_LOTE,
        dispositivo,
    )
    _registrar(
        (
            f"[V7.3 dados] treino={len(treino)} "
            f"validacao={len(validacao)} vocab={tokenizador.tamanho} "
            f"parametros={auditoria['parametros']} "
            f"ffn={CONFIGURACAO_TREINO.implementacao} "
            f"atencao={CONFIGURACAO_TREINO.atencao}"
        ),
        caminho_log,
    )
    _registrar(
        (
            f"[V7.3 antes] PPL={inicial['ppl']:.4f} "
            f"acc={inicial['acuracia_token']:.2%}"
        ),
        caminho_log,
    )

    historico: list[dict[str, object]] = []
    inicio_total = time.perf_counter()
    for epoca in range(1, EPOCAS_EXPERIMENTO + 1):
        modelo.train()
        torch.cuda.reset_peak_memory_stats()
        inicio_epoca = time.perf_counter()
        perda_linguagem_total = 0.0
        perda_auxiliar_total = 0.0
        entropia_total = 0.0
        balanceamento_total = 0.0
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
            logits, auxiliares = modelo(entradas)
            perdas = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                alvos.reshape(-1),
                reduction="none",
                ignore_index=tokenizador.pad_id,
            ).view_as(alvos)
            mascara = alvos.ne(tokenizador.pad_id)
            pesos_validos = pesos * mascara.to(pesos.dtype)
            perda_linguagem = (
                (perdas * pesos_validos).sum()
                / pesos_validos.sum().clamp_min(1.0)
            )
            perda_auxiliar = auxiliares["perda_roteamento"]
            perda = perda_linguagem + perda_auxiliar
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agendador.step()

            quantidade = int(mascara.sum())
            tokens_total += quantidade
            perda_linguagem_total += (
                float(perda_linguagem.detach()) * quantidade
            )
            perda_auxiliar_total += float(perda_auxiliar.detach())
            entropia_total += float(
                auxiliares["entropia_roteamento"].detach()
            )
            balanceamento_total += float(
                auxiliares["balanceamento_roteamento"].detach()
            )
            if passo % 25 == 0:
                _registrar(
                    (
                        f"[V7.3 epoca {epoca}/{EPOCAS_EXPERIMENTO}] "
                        f"passo={passo}/{PASSOS_POR_EPOCA_PADRAO} "
                        f"loss={perda_linguagem_total/tokens_total:.4f} "
                        f"aux={perda_auxiliar_total/passo:.4f} "
                        f"entropia={entropia_total/passo:.4f} "
                        f"lr={agendador.get_last_lr()[0]:.6f}"
                    ),
                    caminho_log,
                )

        torch.cuda.synchronize()
        tempo_treino = time.perf_counter() - inicio_epoca
        validacao_epoca = avaliar_linguagem(
            modelo,
            codificados_validacao,
            tokenizador.pad_id,
            TAMANHO_LOTE,
            dispositivo,
        )
        registro = {
            "epoca": epoca,
            "perda_treino_ponderada": (
                perda_linguagem_total / tokens_total
            ),
            "perda_auxiliar_media": (
                perda_auxiliar_total / PASSOS_POR_EPOCA_PADRAO
            ),
            "entropia_roteamento_media": (
                entropia_total / PASSOS_POR_EPOCA_PADRAO
            ),
            "balanceamento_medio": (
                balanceamento_total / PASSOS_POR_EPOCA_PADRAO
            ),
            "validacao": validacao_epoca,
            "tempo_segundos": tempo_treino,
            "vram_pico_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
            ),
        }
        historico.append(registro)
        salvar_checkpoint_v73(
            args.resultados / f"epoca_{epoca:02d}.pt",
            modelo,
            tokenizador,
            epoca,
            validacao_epoca,
        )
        _registrar(
            (
                f"[V7.3 epoca {epoca}/{EPOCAS_EXPERIMENTO} concluida] "
                f"loss={registro['perda_treino_ponderada']:.4f} "
                f"PPL={validacao_epoca['ppl']:.4f} "
                f"acc={validacao_epoca['acuracia_token']:.2%} "
                f"tempo={tempo_treino:.1f}s "
                f"VRAM={registro['vram_pico_mib']:.1f}MiB"
            ),
            caminho_log,
        )

    relatorio = {
        "experimento": "v73_treino_fp32_5epocas",
        "versao": VERSAO_PROJETO,
        "autor": "Paulo Augusto",
        "ano": 2026,
        "status": "candidato_validacao_pendente",
        "protocolo": {
            "treino": QUANTIDADES_PADRAO[0],
            "validacao": QUANTIDADES_PADRAO[1],
            "teste": QUANTIDADES_PADRAO[2],
            "epocas": EPOCAS_EXPERIMENTO,
            "lote": TAMANHO_LOTE,
            "passos_por_epoca": PASSOS_POR_EPOCA_PADRAO,
            "passos_totais": total_passos,
            "semente": SEMENTE,
            "otimizador": "AdamW",
            "taxa_aprendizado": TAXA_APRENDIZADO,
            "aquecimento_passos": aquecimento,
        },
        "configuracao": asdict(configuracao),
        "configuracao_roteamento": asdict(configuracao_roteamento),
        "configuracao_treino": asdict(CONFIGURACAO_TREINO),
        "arquitetura": auditoria,
        "avaliacao_inicial": inicial,
        "historico": historico,
        "tempo_treino_total_segundos": sum(
            float(item["tempo_segundos"]) for item in historico
        ),
        "vram_treino_pico_mib": max(
            float(item["vram_pico_mib"]) for item in historico
        ),
        "checkpoint_final": str(
            (args.resultados / "epoca_05.pt").resolve()
        ),
        "checkpoint_final_bytes": (
            args.resultados / "epoca_05.pt"
        ).stat().st_size,
        "checkpoint_oficial_preservado": True,
        "elegivel_promocao": True,
        "tempo_total_segundos": time.perf_counter() - inicio_total,
    }
    (args.resultados / "relatorio_treino.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _registrar(
        (
            "[V7.3 concluida] "
            f"treino={relatorio['tempo_treino_total_segundos']:.1f}s "
            f"relatorio={args.resultados / 'relatorio_treino.json'}"
        ),
        caminho_log,
    )


if __name__ == "__main__":
    main()
