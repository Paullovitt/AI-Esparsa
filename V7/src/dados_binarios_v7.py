"""Shards uint16 reversiveis para o treinamento textual da V7.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .tokenizador_palavras import TokenizadorBPECodigo


FORMATO_SHARDS = "v7_codigo_uint16_v1"
MAGIA_INDICE = b"V7IDX001"
CABECALHO_INDICE = struct.Struct("<8sQ")
ENTRADA_INDICE = struct.Struct("<QIII")


def sha256_arquivo(caminho: Path) -> str:
    """Calcula o hash usado para detectar alteracoes nos artefatos."""

    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def carregar_pares_jsonl(caminho: Path) -> list[tuple[str, str]]:
    """Le prompt/resposta sem normalizar caixa ou espacos do conteudo."""

    pares: list[tuple[str, str]] = []
    with caminho.open("r", encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, 1):
            item = json.loads(linha)
            prompt = item.get("prompt")
            resposta = item.get("resposta")
            if not isinstance(prompt, str) or not isinstance(resposta, str):
                raise TypeError(
                    f"prompt/resposta invalidos em {caminho.name}:"
                    f"{numero_linha}"
                )
            if not prompt.strip() or not resposta.strip():
                raise ValueError(
                    f"prompt/resposta vazios em {caminho.name}:"
                    f"{numero_linha}"
                )
            pares.append((prompt, resposta))
    if not pares:
        raise ValueError(f"dataset vazio: {caminho}")
    return pares


def _texto_conversa(prompt: str, resposta: str) -> str:
    """Mantem um contrato unico entre preparacao, treino e inferencia."""

    return f"usuario: {prompt}\nassistente: {resposta}"


def _escrever_ids_uint16(arquivo, ids: Iterable[int]) -> None:
    valores = array("H", ids)
    if sys.byteorder != "little":
        valores.byteswap()
    valores.tofile(arquivo)


@dataclass(frozen=True)
class EntradaRegistroShard:
    """Localizacao de uma conversa dentro de um shard de tokens."""

    shard: int
    offset: int
    comprimento: int
    inicio_resposta: int
    comprimento_resposta: int


@dataclass(frozen=True)
class SegmentoShard:
    """Janela causal que supervisiona uma faixa unica da resposta."""

    shard: int
    inicio_contexto: int
    inicio_alvo: int
    fim_alvo: int


def salvar_tokenizador(
    tokenizador: TokenizadorBPECodigo,
    caminho: Path,
) -> None:
    """Salva o tokenizador completo em JSON independente do checkpoint."""

    conteudo = {
        "tipo": tokenizador.TIPO,
        "tamanho": tokenizador.tamanho,
        "tokens_especiais": {
            "pad": tokenizador.pad_id,
            "bos": tokenizador.bos_id,
            "eos": tokenizador.eos_id,
            "unk": tokenizador.unk_id,
        },
        "serializado": tokenizador.serializado,
    }
    caminho.write_text(
        json.dumps(conteudo, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def carregar_tokenizador(caminho: Path) -> TokenizadorBPECodigo:
    """Restaura e valida o artefato BPE de codigo."""

    conteudo = json.loads(caminho.read_text(encoding="utf-8"))
    if conteudo.get("tipo") != TokenizadorBPECodigo.TIPO:
        raise ValueError("artefato nao contem BPE de codigo")
    tokenizador = TokenizadorBPECodigo.de_serializado(
        conteudo.get("serializado")
    )
    if tokenizador.tamanho != int(conteudo.get("tamanho", -1)):
        raise ValueError("tamanho do tokenizador diverge do artefato")
    return tokenizador


def _abrir_shard(
    destino: Path,
    split: str,
    numero: int,
):
    nome_tokens = f"{split}-{numero:05d}.tokens.uint16"
    nome_indice = f"{split}-{numero:05d}.idx"
    caminho_tokens = destino / nome_tokens
    caminho_indice = destino / nome_indice
    arquivo_tokens = caminho_tokens.open("wb")
    arquivo_indice = caminho_indice.open("wb+")
    arquivo_indice.write(CABECALHO_INDICE.pack(MAGIA_INDICE, 0))
    return (
        arquivo_tokens,
        arquivo_indice,
        caminho_tokens,
        caminho_indice,
    )


def _fechar_shard(
    arquivos,
    registros: int,
    tokens: int,
) -> dict[str, int | str]:
    arquivo_tokens, arquivo_indice, caminho_tokens, caminho_indice = arquivos
    arquivo_indice.seek(0)
    arquivo_indice.write(CABECALHO_INDICE.pack(MAGIA_INDICE, registros))
    arquivo_tokens.close()
    arquivo_indice.close()
    return {
        "tokens": caminho_tokens.name,
        "indice": caminho_indice.name,
        "registros": registros,
        "quantidade_tokens": tokens,
        "sha256_tokens": sha256_arquivo(caminho_tokens),
        "sha256_indice": sha256_arquivo(caminho_indice),
    }


def _converter_split(
    pares: list[tuple[str, str]],
    tokenizador: TokenizadorBPECodigo,
    destino: Path,
    split: str,
    tokens_por_shard: int,
) -> dict:
    """Converte um split sem misturar seus registros com outro split."""

    shards: list[dict[str, int | str]] = []
    arquivos = None
    numero_shard = registros_shard = tokens_shard = 0
    total_tokens = total_bytes_resposta = 0
    for prompt, resposta in pares:
        prefixo = tokenizador.codificar(
            f"usuario: {prompt}\nassistente: ",
            bos=True,
            eos=False,
        )
        alvo = tokenizador.codificar(resposta, bos=False, eos=True)
        # A verificacao integral impede aceitar qualquer normalizacao oculta.
        if tokenizador.decodificar(alvo) != resposta:
            raise RuntimeError("BPE de codigo nao reconstruiu a resposta")
        ids = prefixo + alvo
        # O registro inteiro tambem precisa voltar byte a byte ao texto
        # recebido; isso cobre caixa, indentacao e quebras de linha do prompt.
        if tokenizador.decodificar(ids) != _texto_conversa(prompt, resposta):
            raise RuntimeError("BPE de codigo nao reconstruiu a conversa")
        if max(ids) >= 65_536:
            raise ValueError("ID nao cabe em uint16")
        if (
            arquivos is not None
            and registros_shard > 0
            and tokens_shard + len(ids) > tokens_por_shard
        ):
            shards.append(
                _fechar_shard(
                    arquivos,
                    registros_shard,
                    tokens_shard,
                )
            )
            numero_shard += 1
            arquivos = None
            registros_shard = tokens_shard = 0
        if arquivos is None:
            arquivos = _abrir_shard(destino, split, numero_shard)
        arquivo_tokens, arquivo_indice, _, _ = arquivos
        offset = tokens_shard
        _escrever_ids_uint16(arquivo_tokens, ids)
        arquivo_indice.write(
            ENTRADA_INDICE.pack(
                offset,
                len(ids),
                len(prefixo),
                len(alvo),
            )
        )
        registros_shard += 1
        tokens_shard += len(ids)
        total_tokens += len(ids)
        total_bytes_resposta += len(resposta.encode("utf-8"))
    if arquivos is not None:
        shards.append(
            _fechar_shard(arquivos, registros_shard, tokens_shard)
        )
    return {
        "registros": len(pares),
        "quantidade_tokens": total_tokens,
        "bytes_resposta": total_bytes_resposta,
        "shards": shards,
    }


def preparar_dataset_shards(
    dataset: Path,
    destino: Path,
    *,
    tamanho_vocabulario: int = 8192,
    tokens_por_shard: int = 4_000_000,
) -> Path:
    """Treina no split de treino e grava ambos os splits atomicamente."""

    if tokens_por_shard <= 0:
        raise ValueError("tokens_por_shard deve ser positivo")
    if destino.exists():
        raise FileExistsError(f"destino de shards ja existe: {destino}")
    treino_path = dataset / "treino.jsonl"
    validacao_path = dataset / "validacao.jsonl"
    pares_treino = carregar_pares_jsonl(treino_path)
    pares_validacao = carregar_pares_jsonl(validacao_path)
    # Nenhum texto da validacao participa do aprendizado das mesclas.
    tokenizador = TokenizadorBPECodigo(
        (
            _texto_conversa(prompt, resposta)
            for prompt, resposta in pares_treino
        ),
        tamanho_vocabulario=tamanho_vocabulario,
    )
    temporario = destino.with_name(
        f"{destino.name}.tmp-{os.getpid()}"
    )
    if temporario.exists():
        raise FileExistsError(f"temporario ja existe: {temporario}")
    temporario.mkdir(parents=True)
    try:
        tokenizador_path = temporario / "tokenizador.json"
        salvar_tokenizador(tokenizador, tokenizador_path)
        splits = {
            "treino": _converter_split(
                pares_treino,
                tokenizador,
                temporario,
                "treino",
                tokens_por_shard,
            ),
            "validacao": _converter_split(
                pares_validacao,
                tokenizador,
                temporario,
                "validacao",
                tokens_por_shard,
            ),
        }
        manifesto = {
            "formato": FORMATO_SHARDS,
            "dtype": "uint16_le",
            "tokenizador": "tokenizador.json",
            "tamanho_vocabulario": tokenizador.tamanho,
            "treinado_apenas_no_split": "treino",
            "fontes": {
                "treino": {
                    "arquivo": "treino.jsonl",
                    "sha256": sha256_arquivo(treino_path),
                },
                "validacao": {
                    "arquivo": "validacao.jsonl",
                    "sha256": sha256_arquivo(validacao_path),
                },
            },
            "sha256_tokenizador": sha256_arquivo(tokenizador_path),
            "splits": splits,
            "reversibilidade_validada": True,
            "tokens_unk": 0,
        }
        manifesto_path = temporario / "manifesto.json"
        manifesto_path.write_text(
            json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporario, destino)
    except Exception:
        if temporario.exists():
            shutil.rmtree(temporario)
        raise
    return destino / "manifesto.json"


class CorpusShardsV7:
    """DataLoader por shards mapeados em memoria, sem materializar o corpus."""

    def __init__(
        self,
        diretorio: Path,
        split: str,
        maximo_contexto: int,
        *,
        validar_hashes: bool = True,
    ) -> None:
        if split not in {"treino", "validacao"}:
            raise ValueError("split deve ser treino ou validacao")
        self.diretorio = diretorio
        self.split = split
        self.maximo_contexto = maximo_contexto
        self.manifesto = json.loads(
            (diretorio / "manifesto.json").read_text(encoding="utf-8")
        )
        if self.manifesto.get("formato") != FORMATO_SHARDS:
            raise ValueError("formato de shards V7 desconhecido")
        caminho_tokenizador = diretorio / self.manifesto["tokenizador"]
        if (
            validar_hashes
            and sha256_arquivo(caminho_tokenizador)
            != self.manifesto["sha256_tokenizador"]
        ):
            raise ValueError("hash do tokenizador diverge")
        self.tokenizador = carregar_tokenizador(caminho_tokenizador)
        if (
            self.tokenizador.tamanho
            != int(self.manifesto["tamanho_vocabulario"])
        ):
            raise ValueError("vocabulario diverge do manifesto")
        if self.tokenizador.tamanho > 65_536:
            raise ValueError("vocabulario nao cabe em uint16")
        self._tokens: list[Tensor] = []
        self.registros: list[EntradaRegistroShard] = []
        for numero_shard, metadados in enumerate(
            self.manifesto["splits"][split]["shards"]
        ):
            caminho_tokens = diretorio / metadados["tokens"]
            caminho_indice = diretorio / metadados["indice"]
            if validar_hashes:
                if sha256_arquivo(caminho_tokens) != metadados["sha256_tokens"]:
                    raise ValueError("hash do shard de tokens diverge")
                if sha256_arquivo(caminho_indice) != metadados["sha256_indice"]:
                    raise ValueError("hash do indice diverge")
            quantidade_tokens = int(metadados["quantidade_tokens"])
            if caminho_tokens.stat().st_size != quantidade_tokens * 2:
                raise ValueError("tamanho do shard uint16 e invalido")
            tokens = torch.from_file(
                str(caminho_tokens),
                shared=False,
                size=quantidade_tokens,
                dtype=torch.uint16,
            )
            self._tokens.append(tokens)
            bruto = caminho_indice.read_bytes()
            if len(bruto) < CABECALHO_INDICE.size:
                raise ValueError("indice truncado")
            magia, quantidade = CABECALHO_INDICE.unpack_from(bruto)
            esperado = CABECALHO_INDICE.size + quantidade * ENTRADA_INDICE.size
            if magia != MAGIA_INDICE or len(bruto) != esperado:
                raise ValueError("cabecalho ou comprimento do indice invalido")
            for indice in range(quantidade):
                posicao = CABECALHO_INDICE.size + indice * ENTRADA_INDICE.size
                offset, comprimento, inicio, tamanho_alvo = (
                    ENTRADA_INDICE.unpack_from(bruto, posicao)
                )
                if (
                    offset + comprimento > quantidade_tokens
                    or inicio + tamanho_alvo != comprimento
                ):
                    raise ValueError("entrada aponta para fora do shard")
                self.registros.append(
                    EntradaRegistroShard(
                        numero_shard,
                        offset,
                        comprimento,
                        inicio,
                        tamanho_alvo,
                    )
                )
        self.exemplos: list[SegmentoShard] = []
        prompts_truncados = respostas_segmentadas = tokens_alvo = bytes_alvo = 0
        comprimentos_bytes = self.tokenizador.comprimentos_bytes
        alvos_por_segmento = max(8, maximo_contexto // 2)
        contexto_disponivel = maximo_contexto - alvos_por_segmento
        for registro in self.registros:
            inicio_registro = registro.offset
            inicio_resposta = registro.offset + registro.inicio_resposta
            fim_resposta = inicio_resposta + registro.comprimento_resposta
            inicio_alvo = inicio_resposta
            quantidade_segmentos = 0
            while inicio_alvo < fim_resposta:
                fim_alvo = min(
                    fim_resposta,
                    inicio_alvo + alvos_por_segmento,
                )
                inicio_contexto = max(
                    inicio_registro,
                    inicio_alvo - contexto_disponivel,
                )
                self.exemplos.append(
                    SegmentoShard(
                        registro.shard,
                        inicio_contexto,
                        inicio_alvo,
                        fim_alvo,
                    )
                )
                quantidade_segmentos += 1
                inicio_alvo = fim_alvo
            prompts_truncados += int(
                registro.inicio_resposta > contexto_disponivel
            )
            respostas_segmentadas += int(quantidade_segmentos > 1)
            tokens_alvo += registro.comprimento_resposta
            ids_alvo = self._tokens[registro.shard][
                inicio_resposta:fim_resposta
            ].to(torch.long)
            bytes_alvo += sum(
                comprimentos_bytes[token_id]
                for token_id in ids_alvo.tolist()
            )
        self.estatisticas = {
            "registros": len(self.registros),
            "segmentos": len(self.exemplos),
            "prompts_truncados": prompts_truncados,
            "respostas_segmentadas": respostas_segmentadas,
            "tokens_alvo": tokens_alvo,
            "bytes_alvo": bytes_alvo,
            "tokens_descartados": 0,
            "eos_artificiais": 0,
            "armazenamento": FORMATO_SHARDS,
        }

    def comprimento_exemplo(self, indice: int) -> int:
        """Devolve o comprimento causal sem copiar dados do shard."""

        exemplo = self.exemplos[indice]
        return exemplo.fim_alvo - exemplo.inicio_contexto - 1

    def preparar_lote(
        self,
        indices: list[int],
        *,
        comprimento_alvo: int | None = None,
        fixar_memoria: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Monta o lote na CPU, inclusive em memoria fixada para prefetch."""

        exemplos = [self.exemplos[indice] for indice in indices]
        comprimento_real = max(
            exemplo.fim_alvo - exemplo.inicio_contexto - 1
            for exemplo in exemplos
        )
        comprimento = comprimento_alvo or comprimento_real
        if comprimento < comprimento_real:
            raise ValueError("comprimento_alvo menor que o maior exemplo")
        if comprimento > self.maximo_contexto:
            raise ValueError("lote excede o contexto maximo")
        tokens = torch.full(
            (len(indices), comprimento),
            self.tokenizador.pad_id,
            dtype=torch.long,
            pin_memory=fixar_memoria,
        )
        labels = torch.full_like(tokens, -100)
        for linha, exemplo in enumerate(exemplos):
            janela = self._tokens[exemplo.shard][
                exemplo.inicio_contexto:exemplo.fim_alvo
            ].to(torch.long)
            entrada = janela[:-1]
            tokens[linha, : entrada.numel()] = entrada
            inicio_label = (
                exemplo.inicio_alvo - exemplo.inicio_contexto - 1
            )
            alvos = self._tokens[exemplo.shard][
                exemplo.inicio_alvo:exemplo.fim_alvo
            ].to(torch.long)
            labels[
                linha,
                inicio_label : inicio_label + alvos.numel(),
            ] = alvos
        return tokens, labels

    def lote(
        self,
        indices: list[int],
        dispositivo: torch.device,
        *,
        comprimento_alvo: int | None = None,
        fixar_memoria: bool = False,
    ) -> tuple[Tensor, Tensor]:
        tokens, labels = self.preparar_lote(
            indices,
            comprimento_alvo=comprimento_alvo,
            fixar_memoria=fixar_memoria,
        )
        assincrono = fixar_memoria and dispositivo.type == "cuda"
        return (
            tokens.to(dispositivo, non_blocking=assincrono),
            labels.to(dispositivo, non_blocking=assincrono),
        )

    def decodificar_resposta(self, indice: int) -> str:
        """Reconstrui exatamente a resposta original de um registro."""

        registro = self.registros[indice]
        inicio = registro.offset + registro.inicio_resposta
        fim = inicio + registro.comprimento_resposta
        ids = self._tokens[registro.shard][inicio:fim].to(torch.long)
        return self.tokenizador.decodificar(ids.tolist())

    def decodificar_registro(self, indice: int) -> str:
        """Reconstrui a conversa completa, incluindo formatacao do prompt."""

        registro = self.registros[indice]
        inicio = registro.offset
        fim = inicio + registro.comprimento
        ids = self._tokens[registro.shard][inicio:fim].to(torch.long)
        return self.tokenizador.decodificar(ids.tolist())
