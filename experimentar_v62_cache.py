"""Valida e mede o runtime-base V6.2 com cache causal incremental e CSR.

O experimento reutiliza estritamente os pesos oficiais, nao treina, nao cria
checkpoint e nao altera ``modelos/gerador_esparso_base.pt``. A V6.1 sem cache
e executada pelo mesmo objeto para isolar somente o caminho de inferencia.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from src.decodificador_gerador import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_gerador_esparso_v62 import ModeloGeradorEsparsoV62
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO
from treinar_gerador_esparso import (
    FRASES_GERACAO,
    QUANTIDADES_PADRAO,
    avaliar_geracao_livre,
    avaliar_linguagem,
    codificar_registros,
    extrair_continuacao,
    gerar_divisoes_gerador,
)


RAIZ = Path(__file__).resolve().parent
CHECKPOINT_OFICIAL = RAIZ / "modelos" / "gerador_esparso_base.pt"
RELATORIO_OFICIAL = (
    RAIZ
    / "resultados"
    / "gerador_esparso_base_50k"
    / "relatorio.json"
)
RESULTADOS_PADRAO = RAIZ / "resultados" / "v62_base_runtime"


def _sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _ambiente(dispositivo: torch.device) -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dispositivo": str(dispositivo),
        "gpu": (
            torch.cuda.get_device_name(dispositivo)
            if dispositivo.type == "cuda"
            else None
        ),
    }


def carregar_v62(
    caminho: Path,
    dispositivo: torch.device,
) -> tuple[
    ModeloGeradorEsparsoV62,
    TokenizadorPalavras,
    dict[str, object],
]:
    """Reconstroi a V6.2 usando somente os tensores seguros da V6.1."""

    checkpoint = torch.load(
        caminho,
        map_location=dispositivo,
        weights_only=True,
    )
    if checkpoint.get("modelo") != "gerador-esparso-coerente-base":
        raise ValueError("o experimento exige o checkpoint oficial esparso")
    if int(checkpoint.get("epoca", 0)) != 5:
        raise ValueError("o checkpoint oficial deve corresponder a epoca 5")
    tokenizador = TokenizadorPalavras.de_vocabulario(
        checkpoint["vocabulario"]
    )
    configuracao = ConfiguracaoGeradorEsparso(
        **checkpoint["configuracao"]
    )
    modelo = ModeloGeradorEsparsoV62(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao,
        int(checkpoint["semente"]),
    ).to(dispositivo)
    modelo.load_state_dict(
        checkpoint["estado_modelo"],
        strict=True,
    )
    modelo.eval()
    return modelo, tokenizador, checkpoint


def _configuracao_relato() -> ConfiguracaoDecodificacao:
    return ConfiguracaoDecodificacao(
        maximo_tokens=500,
        minimo_tokens_frase=300,
        ngrama_bloqueado=0,
        penalidade_repeticao=0.0,
        janela_penalidade=14,
        temperatura=0.58,
        top_k_amostragem=5,
        parar_apos_frase=True,
        frases_para_encerrar=FRASES_GERACAO,
        bloquear_trigrama_adicional=False,
    )


@torch.inference_mode()
def validar_equivalencia_logits(
    modelo: ModeloGeradorEsparsoV62,
    tokenizador: TokenizadorPalavras,
    dispositivo: torch.device,
) -> dict[str, object]:
    """Compara prefill e passos incrementais com o forward V6.1 herdado."""

    gerador = torch.Generator(device=dispositivo).manual_seed(620_2026)
    diferencas_prefill: list[dict[str, float | int]] = []
    for comprimento in (8, 32, 128, 512):
        tokens = torch.randint(
            4,
            tokenizador.tamanho,
            (1, comprimento),
            generator=gerador,
            device=dispositivo,
        )
        referencia, _ = modelo(tokens)
        cacheado, _ = modelo.iniciar_cache_geracao(tokens)
        diferenca = float(
            (referencia[:, -1:] - cacheado).abs().max()
        )
        diferencas_prefill.append(
            {
                "comprimento": comprimento,
                "diferenca_maxima_absoluta": diferenca,
            }
        )

    historico = torch.randint(
        4,
        tokenizador.tamanho,
        (1, 24),
        generator=gerador,
        device=dispositivo,
    )
    cacheado, cache = modelo.iniciar_cache_geracao(historico)
    maior_incremental = 0.0
    for _ in range(64):
        referencia, _ = modelo(historico)
        maior_incremental = max(
            maior_incremental,
            float((referencia[:, -1:] - cacheado).abs().max()),
        )
        novo_token = torch.randint(
            4,
            tokenizador.tamanho,
            (1, 1),
            generator=gerador,
            device=dispositivo,
        )
        historico = torch.cat((historico, novo_token), dim=1)
        cacheado, cache = modelo.avancar_cache_geracao(
            novo_token,
            cache,
        )

    maior_prefill = max(
        float(item["diferenca_maxima_absoluta"])
        for item in diferencas_prefill
    )
    tolerancia = 2e-5
    return {
        "tolerancia": tolerancia,
        "prefill": diferencas_prefill,
        "diferenca_maxima_prefill": maior_prefill,
        "diferenca_maxima_incremental": maior_incremental,
        "aprovado": (
            maior_prefill <= tolerancia
            and maior_incremental <= tolerancia
        ),
    }


def _limpar_medicao(dispositivo: torch.device) -> None:
    gc.collect()
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(dispositivo)
        torch.cuda.reset_peak_memory_stats(dispositivo)


@torch.inference_mode()
def medir_geracao(
    modelo: ModeloGeradorEsparsoV62,
    tokenizador: TokenizadorPalavras,
    prompt: str,
    dispositivo: torch.device,
    *,
    usar_cache: bool,
) -> tuple[str, dict[str, float]]:
    """Mede uma geracao completa e inclui o pico de memoria alocada."""

    _limpar_medicao(dispositivo)
    metricas: dict[str, float] = {}
    texto = gerar_controlado(
        modelo,
        tokenizador,
        prompt,
        dispositivo,
        configuracao=_configuracao_relato(),
        amostrar=False,
        metricas_desempenho=metricas,
        usar_cache_incremental=usar_cache,
    )
    if dispositivo.type == "cuda":
        metricas["vram_pico_mib"] = (
            torch.cuda.max_memory_allocated(dispositivo) / 1024**2
        )
    else:
        metricas["vram_pico_mib"] = 0.0
    metricas["caracteres_gerados"] = float(
        len(extrair_continuacao(texto))
    )
    return texto, metricas


def _comparar_geracoes_oficiais(
    atual: dict[str, object],
    oficial: dict[str, object],
) -> dict[str, object]:
    exemplos_atuais = atual["exemplos"]
    exemplos_oficiais = oficial["exemplos"]
    comparados = min(len(exemplos_atuais), len(exemplos_oficiais))
    textos_identicos = sum(
        atual_item["pedido"] == oficial_item["pedido"]
        and atual_item["texto"] == oficial_item["texto"]
        for atual_item, oficial_item in zip(
            exemplos_atuais[:comparados],
            exemplos_oficiais[:comparados],
            strict=True,
        )
    )
    return {
        "comparados": comparados,
        "textos_identicos": textos_identicos,
        "todos_identicos": textos_identicos == comparados,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_OFICIAL,
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTADOS_PADRAO,
    )
    args = parser.parse_args()
    if not args.checkpoint.exists() or not RELATORIO_OFICIAL.exists():
        raise FileNotFoundError(
            "checkpoint ou relatorio oficial nao encontrado"
        )
    if args.resultados.exists():
        raise FileExistsError(
            "diretorio de resultados ja existe; escolha outro caminho"
        )

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    hash_antes = _sha256(args.checkpoint)
    modelo, tokenizador, checkpoint = carregar_v62(
        args.checkpoint,
        dispositivo,
    )
    oficial = json.loads(
        RELATORIO_OFICIAL.read_text(encoding="utf-8")
    )

    print("[V6.2] validando equivalencia numerica...", flush=True)
    equivalencia = validar_equivalencia_logits(
        modelo,
        tokenizador,
        dispositivo,
    )
    if not equivalencia["aprovado"]:
        raise RuntimeError("cache incremental divergiu da referencia")

    print("[V6.2] reconstruindo corpus de teste...", flush=True)
    _, _, registros_teste = gerar_divisoes_gerador(
        QUANTIDADES_PADRAO
    )
    codificados_teste = codificar_registros(
        registros_teste,
        tokenizador,
    )
    avaliacao = avaliar_linguagem(
        modelo,
        codificados_teste,
        tokenizador.pad_id,
        100,
        dispositivo,
    )

    print("[V6.2] revalidando 24 geracoes com cache...", flush=True)
    geracao = avaliar_geracao_livre(
        modelo,
        tokenizador,
        registros_teste,
        dispositivo,
        quantidade=24,
    )
    equivalencia_geracoes = _comparar_geracoes_oficiais(
        geracao,
        oficial["geracao_livre"],
    )

    prompt_benchmark = str(registros_teste[0]["pedido"])
    # Um pre-aquecimento curto materializa as matrizes COO de inferencia antes
    # de ambas as medicoes, sem favorecer um dos caminhos comparados.
    gerar_controlado(
        modelo,
        tokenizador,
        prompt_benchmark,
        dispositivo,
        configuracao=ConfiguracaoDecodificacao(
            maximo_tokens=8,
            parar_apos_frase=False,
        ),
        usar_cache_incremental=True,
    )
    print("[V6.2] medindo recalculo completo sem cache...", flush=True)
    texto_referencia, sem_cache = medir_geracao(
        modelo,
        tokenizador,
        prompt_benchmark,
        dispositivo,
        usar_cache=False,
    )
    print("[V6.2] medindo cache incremental CSR...", flush=True)
    texto_cache, com_cache = medir_geracao(
        modelo,
        tokenizador,
        prompt_benchmark,
        dispositivo,
        usar_cache=True,
    )

    if texto_referencia != texto_cache:
        raise RuntimeError("texto cacheado divergiu da referencia")
    hash_depois = _sha256(args.checkpoint)
    if hash_antes != hash_depois:
        raise RuntimeError("checkpoint oficial foi modificado")

    relatorio: dict[str, object] = {
        "experimento": "v62_base_runtime_csr",
        "versao_projeto": VERSAO_PROJETO,
        "versao_runtime": "6.2",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "objetivo": (
            "reduzir o custo autorregressivo preservando integralmente "
            "pesos, topologia esparsa e saida da V6.1"
        ),
        "ambiente": _ambiente(dispositivo),
        "checkpoint": {
            "origem": str(args.checkpoint.resolve()),
            "modelo": checkpoint["modelo"],
            "epoca": int(checkpoint["epoca"]),
            "carregado_weights_only": True,
            "strict": True,
            "sha256_antes": hash_antes,
            "sha256_depois": hash_depois,
            "oficial_preservado": hash_antes == hash_depois,
        },
        "treino": {
            "executado": False,
            "motivo": (
                "otimizacao exclusiva de runtime, sem novos parametros "
                "ou alteracao de pesos"
            ),
            "ppl_requer_novo_treino": False,
        },
        "configuracao": asdict(modelo.configuracao),
        "arquitetura": modelo.auditoria(),
        "equivalencia_numerica": equivalencia,
        "avaliacao_final": avaliacao,
        "comparacao_ppl_oficial": {
            "v61": float(oficial["avaliacao_final"]["ppl"]),
            "v62": float(avaliacao["ppl"]),
            "diferenca_absoluta": abs(
                float(oficial["avaliacao_final"]["ppl"])
                - float(avaliacao["ppl"])
            ),
        },
        "geracao_livre": geracao,
        "equivalencia_geracoes_oficiais": equivalencia_geracoes,
        "benchmark_autorregressivo": {
            "prompt": prompt_benchmark,
            "texto_identico": True,
            "recalculo_completo_sem_cache": sem_cache,
            "v62_cache_incremental_csr": com_cache,
            "aceleracao": (
                float(com_cache["tokens_por_segundo"])
                / float(sem_cache["tokens_por_segundo"])
            ),
            "reducao_tempo_percentual": (
                1.0
                - float(com_cache["tempo_total_segundos"])
                / float(sem_cache["tempo_total_segundos"])
            )
            * 100.0,
            "variacao_vram_percentual": (
                float(com_cache["vram_pico_mib"])
                / float(sem_cache["vram_pico_mib"])
                - 1.0
            )
            * 100.0,
        },
        "limitacoes": [
            "o cache acelera geracao, nao o treino nem o forward paralelo",
            "ao ultrapassar 640 tokens o cache refaz a janela para preservar "
            "as posicoes da referencia",
            "a medicao e especifica do hardware e ambiente registrados",
        ],
    }
    args.resultados.mkdir(parents=True, exist_ok=False)
    (args.resultados / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.resultados / "texto_benchmark.txt").write_text(
        extrair_continuacao(texto_cache) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "equivalencia": equivalencia["aprovado"],
                "ppl": avaliacao["ppl"],
                "geracoes_aprovadas": (
                    f"{int(geracao['taxa_aprovacao'] * 24)}/24"
                ),
                "textos_oficiais_identicos": equivalencia_geracoes,
                "sem_cache_tokens_s": sem_cache["tokens_por_segundo"],
                "com_cache_tokens_s": com_cache["tokens_por_segundo"],
                "aceleracao": relatorio[
                    "benchmark_autorregressivo"
                ]["aceleracao"],
                "resultado": str(
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
