"""Treina a V8 multimodal a partir de manifestos locais explícitos.

O script nunca baixa dados, nunca sobrescreve modelos existentes e salva uma
época por checkpoint. Cinco épocas tornam um resultado elegível para futura
comparação, mas promoção continua exigindo validação e autorização separadas.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dados_multimodais_v8 import (
    DatasetMultimodalV8,
    agrupar_lote_multimodal,
    carregar_manifesto_multimodal,
    criar_tokenizador_multimodal,
)
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_multimodal_v8 import (
    ConfiguracaoMultimodalV8,
    ModeloMultimodalEsparsoTreinoV8,
    ModeloMultimodalEsparsoV8,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.treino_vetorizado_v73 import ConfiguracaoTreinoV73
from src.versao import VERSAO_PROJETO


RAIZ = Path(__file__).resolve().parent
SEMENTE = 20260801
EPOCAS_PROMOVIVEIS = 5
CHECKPOINT_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"
CONFIGURACAO_TREINO = ConfiguracaoTreinoV73(
    implementacao="microblocos_densos",
    atencao="matriz_densa_topk",
    projecoes_qk="densas_temporarias",
)


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def inicializar_backbone_v73(
    modelo: ModeloMultimodalEsparsoV8,
    caminho: Path,
) -> dict[str, int]:
    """Transfere somente tensores V7.3 de shape compatível para a V8."""

    checkpoint = torch.load(caminho, map_location="cpu", weights_only=True)
    if not str(checkpoint.get("modelo", "")).startswith(
        "gerador-esparso-v73"
    ):
        raise ValueError("checkpoint de inicializacao nao pertence a V7.3")
    destino = modelo.state_dict()
    origem = checkpoint["estado_modelo"]
    transferiveis = {
        nome: tensor
        for nome, tensor in origem.items()
        if nome in destino and destino[nome].shape == tensor.shape
    }
    resultado = modelo.load_state_dict(transferiveis, strict=False)
    return {
        "tensores_transferidos": len(transferiveis),
        "tensores_inicializados_v8": len(resultado.missing_keys),
        "tensores_inesperados": len(resultado.unexpected_keys),
    }


def _perda_resposta(
    logits: Tensor,
    alvos: Tensor,
    pad_id: int,
) -> tuple[Tensor, int]:
    perdas = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        alvos.reshape(-1),
        ignore_index=pad_id,
        reduction="sum",
    )
    tokens = int(alvos.ne(pad_id).sum())
    return perdas, tokens


def _validar_comprimentos_dataset(
    dataset: DatasetMultimodalV8,
    maximo_tokens_texto: int,
    nome_split: str,
) -> None:
    """Falha antes do treino se algum texto não couber após os patches."""

    excedentes = [
        exemplo.identificador
        for exemplo in dataset.exemplos
        if len(exemplo.entradas) > maximo_tokens_texto
    ]
    if excedentes:
        amostra = ", ".join(excedentes[:5])
        raise ValueError(
            f"{nome_split} possui sequencias acima de "
            f"{maximo_tokens_texto} tokens: {amostra}"
        )


@torch.inference_mode()
def avaliar(
    modelo: ModeloMultimodalEsparsoV8,
    carregador: DataLoader,
    pad_id: int,
    dispositivo: torch.device,
) -> dict[str, float]:
    modelo.eval()
    perda_total = 0.0
    tokens_total = 0
    acertos = 0
    for entradas, alvos, imagens in carregador:
        entradas = entradas.to(dispositivo)
        alvos = alvos.to(dispositivo)
        imagens = imagens.to(dispositivo)
        logits, _ = modelo(entradas, imagens)
        perda, tokens = _perda_resposta(logits, alvos, pad_id)
        mascara = alvos.ne(pad_id)
        acertos += int(logits.argmax(-1).eq(alvos).logical_and(mascara).sum())
        perda_total += float(perda)
        tokens_total += tokens
    if tokens_total == 0:
        raise ValueError("validacao nao possui tokens de resposta")
    perda_media = perda_total / tokens_total
    return {
        "loss": perda_media,
        "ppl": float(torch.exp(torch.tensor(perda_media))),
        "acuracia_token": acertos / tokens_total,
        "tokens_avaliados": float(tokens_total),
    }


def salvar_checkpoint(
    caminho: Path,
    modelo: ModeloMultimodalEsparsoV8,
    tokenizador,
    epoca: int,
    epocas_planejadas: int,
    metricas: dict[str, float],
    inicializacao: dict[str, int] | None,
    manifesto_treino: Path,
    manifesto_validacao: Path,
) -> None:
    """Persiste somente tensores e metadados seguros, sem copiar imagens."""

    auditoria = modelo.auditoria()
    auditoria["checkpoint_treinado"] = True
    torch.save(
        {
            "modelo": "gerador-esparso-v8-multimodal",
            "versao": VERSAO_PROJETO,
            "epoca": epoca,
            "epocas_planejadas": epocas_planejadas,
            "semente": SEMENTE,
            "configuracao": asdict(modelo.configuracao),
            "configuracao_roteamento": asdict(
                modelo.configuracao_roteamento
            ),
            "configuracao_multimodal": asdict(
                modelo.configuracao_multimodal
            ),
            "configuracao_treino": asdict(
                modelo.configuracao_treino
            ),
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "metricas_validacao": metricas,
            "auditoria": auditoria,
            "inicializacao_v73": inicializacao,
            "manifestos": {
                "treino_sha256": _sha256(manifesto_treino),
                "validacao_sha256": _sha256(manifesto_validacao),
            },
            "dataset_incorporado_no_checkpoint": False,
            "elegivel_comparacao": (
                epoca == EPOCAS_PROMOVIVEIS
                and epocas_planejadas == EPOCAS_PROMOVIVEIS
            ),
        },
        caminho,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifesto-treino", type=Path, required=True)
    parser.add_argument("--manifesto-validacao", type=Path, required=True)
    parser.add_argument("--raiz-imagens", type=Path, required=True)
    parser.add_argument("--resultados", type=Path, required=True)
    parser.add_argument("--epocas", type=int, default=EPOCAS_PROMOVIVEIS)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--taxa", type=float, default=1e-3)
    parser.add_argument(
        "--do-zero",
        action="store_true",
        help="nao transfere os tensores compativeis do checkpoint V7.3",
    )
    parser.add_argument(
        "--checkpoint-v73",
        type=Path,
        default=CHECKPOINT_V73,
    )
    args = parser.parse_args()

    if args.epocas <= 0 or args.lote <= 0 or args.taxa <= 0:
        raise ValueError("epocas, lote e taxa devem ser positivos")
    resultados = args.resultados.resolve()
    if resultados == (RAIZ / "modelos").resolve():
        raise ValueError("resultados nao pode ser a pasta de modelos")
    if resultados.exists() and list(resultados.glob("epoca_*.pt")):
        raise FileExistsError("o destino ja possui checkpoints de epoca")

    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMENTE)
    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    treino = carregar_manifesto_multimodal(
        args.manifesto_treino,
        args.raiz_imagens,
    )
    validacao = carregar_manifesto_multimodal(
        args.manifesto_validacao,
        args.raiz_imagens,
    )
    tokenizador = criar_tokenizador_multimodal(treino)
    configuracao_base = ConfiguracaoGeradorEsparso()
    configuracao_roteamento = ConfiguracaoRoteamentoV73()
    configuracao_multimodal = ConfiguracaoMultimodalV8()
    dataset_treino = DatasetMultimodalV8(
        treino,
        tokenizador,
        configuracao_multimodal.altura_imagem,
        configuracao_multimodal.largura_imagem,
    )
    dataset_validacao = DatasetMultimodalV8(
        validacao,
        tokenizador,
        configuracao_multimodal.altura_imagem,
        configuracao_multimodal.largura_imagem,
    )
    limite_texto = (
        configuracao_base.maximo_contexto
        - configuracao_multimodal.quantidade_patches
    )
    _validar_comprimentos_dataset(
        dataset_treino,
        limite_texto,
        "treino",
    )
    _validar_comprimentos_dataset(
        dataset_validacao,
        limite_texto,
        "validacao",
    )
    agrupar = partial(
        agrupar_lote_multimodal,
        pad_id=tokenizador.pad_id,
    )
    gerador = torch.Generator().manual_seed(SEMENTE)
    carregador_treino = DataLoader(
        dataset_treino,
        batch_size=args.lote,
        shuffle=True,
        collate_fn=agrupar,
        generator=gerador,
    )
    carregador_validacao = DataLoader(
        dataset_validacao,
        batch_size=args.lote,
        shuffle=False,
        collate_fn=agrupar,
    )

    modelo = ModeloMultimodalEsparsoTreinoV8(
        tokenizador.tamanho,
        tokenizador.pad_id,
        configuracao_base,
        SEMENTE,
        configuracao_roteamento,
        configuracao_multimodal,
        CONFIGURACAO_TREINO,
    )
    inicializacao = None
    if not args.do_zero:
        inicializacao = inicializar_backbone_v73(
            modelo,
            args.checkpoint_v73,
        )
    modelo.to(dispositivo)
    otimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.taxa,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    resultados.mkdir(parents=True, exist_ok=True)
    historico = []
    inicio_total = time.perf_counter()

    for epoca in range(1, args.epocas + 1):
        modelo.train()
        perda_total = 0.0
        tokens_total = 0
        inicio_epoca = time.perf_counter()
        for entradas, alvos, imagens in carregador_treino:
            entradas = entradas.to(dispositivo)
            alvos = alvos.to(dispositivo)
            imagens = imagens.to(dispositivo)
            otimizador.zero_grad(set_to_none=True)
            logits, auxiliares = modelo(entradas, imagens)
            perda_soma, tokens = _perda_resposta(
                logits,
                alvos,
                tokenizador.pad_id,
            )
            perda = perda_soma / max(1, tokens)
            perda = perda + auxiliares["perda_roteamento"]
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            perda_total += float(perda_soma.detach())
            tokens_total += tokens

        metricas = avaliar(
            modelo,
            carregador_validacao,
            tokenizador.pad_id,
            dispositivo,
        )
        registro = {
            "epoca": epoca,
            "loss_treino": perda_total / max(1, tokens_total),
            "segundos": time.perf_counter() - inicio_epoca,
            "validacao": metricas,
        }
        historico.append(registro)
        salvar_checkpoint(
            resultados / f"epoca_{epoca:02d}.pt",
            modelo,
            tokenizador,
            epoca,
            args.epocas,
            metricas,
            inicializacao,
            args.manifesto_treino,
            args.manifesto_validacao,
        )
        print(json.dumps(registro, ensure_ascii=False), flush=True)

    relatorio = {
        "modelo": "gerador-esparso-v8-multimodal",
        "versao": VERSAO_PROJETO,
        "dispositivo": str(dispositivo),
        "epocas": args.epocas,
        "lote": args.lote,
        "registros_treino": len(treino),
        "registros_validacao": len(validacao),
        "segundos_totais": time.perf_counter() - inicio_total,
        "inicializacao_v73": inicializacao,
        "historico": historico,
        "auditoria": modelo.auditoria(),
    }
    (resultados / "relatorio.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
