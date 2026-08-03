"""Tokenizador determinístico de palavras e pontuação da base esparsa.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SEGMENTO_BPE_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


class TokenizadorPalavras:
    """Mapeia palavras/pontuação para IDs com vocabulário serializável."""

    TIPO = "palavras_v1"

    def __init__(self, textos: Iterable[str]) -> None:
        especiais = ["<pad>", "<bos>", "<eos>", "<unk>"]
        tokens = sorted(
            {
                token
                for texto in textos
                for token in self.tokenizar(texto)
            }
        )
        self.id_para_token = tuple(
            especiais
            + [token for token in tokens if token not in especiais]
        )
        self.token_para_id = {
            token: indice
            for indice, token in enumerate(self.id_para_token)
        }

    @classmethod
    def de_vocabulario(
        cls,
        id_para_token: Iterable[str],
    ) -> "TokenizadorPalavras":
        """Reconstrói estritamente o vocabulário salvo no checkpoint."""

        tokens = tuple(str(token) for token in id_para_token)
        especiais = ("<pad>", "<bos>", "<eos>", "<unk>")
        if tokens[: len(especiais)] != especiais:
            raise ValueError("vocabulario nao preserva os tokens especiais")
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulario contem tokens duplicados")
        tokenizador = cls.__new__(cls)
        tokenizador.id_para_token = tokens
        tokenizador.token_para_id = {
            token: indice for indice, token in enumerate(tokens)
        }
        return tokenizador

    @staticmethod
    def tokenizar(texto: str) -> list[str]:
        return TOKEN_RE.findall(texto.lower())

    def codificar(
        self,
        texto: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        ids = [
            self.token_para_id.get(token, self.unk_id)
            for token in self.tokenizar(texto)
        ]
        return (
            ([self.bos_id] if bos else [])
            + ids
            + ([self.eos_id] if eos else [])
        )

    def tokens_desconhecidos(self, texto: str) -> list[str]:
        """Lista tokens fora do vocabulário sem duplicar ocorrências."""

        desconhecidos: list[str] = []
        vistos: set[str] = set()
        for token in self.tokenizar(texto):
            if token not in self.token_para_id and token not in vistos:
                desconhecidos.append(token)
                vistos.add(token)
        return desconhecidos

    def validar_texto_no_vocabulario(self, texto: str) -> None:
        """Rejeita entradas que perderiam informação ao virar ``<unk>``."""

        desconhecidos = self.tokens_desconhecidos(texto)
        if desconhecidos:
            raise ValueError(
                "tokens fora do vocabulario fechado: "
                + ", ".join(desconhecidos)
            )

    def decodificar(self, ids: Iterable[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            token = self.id_para_token[int(token_id)]
            if token in {"<pad>", "<bos>"}:
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        texto = " ".join(tokens)
        for pontuacao in ".,;:!?":
            texto = texto.replace(" " + pontuacao, pontuacao)
        return texto[:1].upper() + texto[1:] if texto else texto

    @property
    def tamanho(self) -> int:
        return len(self.id_para_token)

    @property
    def pad_id(self) -> int:
        return self.token_para_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_para_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_para_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_para_id["<unk>"]

    @property
    def comprimentos_bytes(self) -> tuple[int, ...]:
        return tuple(
            0 if token.startswith("<") else len(token.encode("utf-8"))
            for token in self.id_para_token
        )


class TokenizadorBytes:
    """Tokenizador UTF-8 fechado com 260 IDs e cobertura textual completa.

    Bytes evitam que o embedding cresça com o corpus e representam português,
    inglês e qualquer entrada Unicode sem produzir ``<unk>``.
    """

    TIPO = "bytes_utf8_v1"

    def __init__(self) -> None:
        especiais = ("<pad>", "<bos>", "<eos>", "<unk>")
        self.id_para_token = especiais + tuple(
            f"<byte:{valor:02x}>" for valor in range(256)
        )
        self.token_para_id = {
            token: indice for indice, token in enumerate(self.id_para_token)
        }

    @classmethod
    def de_vocabulario(
        cls,
        id_para_token: Iterable[str],
    ) -> "TokenizadorBytes":
        tokenizador = cls()
        if tuple(id_para_token) != tokenizador.id_para_token:
            raise ValueError("vocabulario byte-level invalido")
        return tokenizador

    @staticmethod
    def tokenizar(texto: str) -> list[str]:
        return [f"<byte:{valor:02x}>" for valor in texto.lower().encode("utf-8")]

    def codificar(
        self,
        texto: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        ids = [4 + valor for valor in texto.lower().encode("utf-8")]
        return (
            ([self.bos_id] if bos else [])
            + ids
            + ([self.eos_id] if eos else [])
        )

    def tokens_desconhecidos(self, texto: str) -> list[str]:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")
        return []

    def validar_texto_no_vocabulario(self, texto: str) -> None:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")

    def decodificar(self, ids: Iterable[int]) -> str:
        valores = bytearray()
        for token_id in ids:
            indice = int(token_id)
            if indice in {self.pad_id, self.bos_id}:
                continue
            if indice == self.eos_id:
                break
            if 4 <= indice < 260:
                valores.append(indice - 4)
        texto = valores.decode("utf-8", errors="replace")
        return texto[:1].upper() + texto[1:] if texto else texto

    @property
    def tamanho(self) -> int:
        return 260

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def comprimentos_bytes(self) -> tuple[int, ...]:
        return (0, 0, 0, 0) + (1,) * 256


class TokenizadorBPEBytes:
    """BPE determinístico sobre UTF-8 com fallback integral em bytes.

    As fusões nunca atravessam segmentos de espaço, palavra ou pontuação.
    Isso preserva reconstrução exata, reduz o comprimento das sequências e
    mantém cobertura Unicode sem depender de uma biblioteca externa.
    """

    TIPO = "bpe_bytes_v1"

    def __init__(
        self,
        textos: Iterable[str],
        tamanho_vocabulario: int = 2048,
    ) -> None:
        if tamanho_vocabulario < 260:
            raise ValueError("tamanho_vocabulario BPE deve ser pelo menos 260")
        frequencias: Counter[bytes] = Counter()
        for texto in textos:
            if not isinstance(texto, str):
                raise TypeError("textos do BPE devem ser str")
            frequencias.update(
                segmento.encode("utf-8")
                for segmento in SEGMENTO_BPE_RE.findall(texto.lower())
            )
        if not frequencias:
            raise ValueError("nenhum texto fornecido ao BPE")

        sequencias = {
            segmento: tuple(4 + valor for valor in segmento)
            for segmento in frequencias
        }
        bytes_por_id = [b""] * 4 + [bytes((valor,)) for valor in range(256)]
        mesclas: list[tuple[int, int]] = []
        while len(bytes_por_id) < tamanho_vocabulario:
            pares: Counter[tuple[int, int]] = Counter()
            for segmento, sequencia in sequencias.items():
                frequencia = frequencias[segmento]
                for indice in range(len(sequencia) - 1):
                    pares[(sequencia[indice], sequencia[indice + 1])] += (
                        frequencia
                    )
            if not pares:
                break
            maior_frequencia = max(pares.values())
            melhor_par = min(
                par for par, frequencia in pares.items()
                if frequencia == maior_frequencia
            )
            novo_id = len(bytes_por_id)
            mesclas.append(melhor_par)
            bytes_por_id.append(
                bytes_por_id[melhor_par[0]] + bytes_por_id[melhor_par[1]]
            )
            sequencias = {
                segmento: self._aplicar_par(
                    sequencia,
                    melhor_par,
                    novo_id,
                )
                for segmento, sequencia in sequencias.items()
            }
        self._inicializar(tuple(bytes_por_id), tuple(mesclas))

    @staticmethod
    def _aplicar_par(
        sequencia: tuple[int, ...] | list[int],
        par: tuple[int, int],
        novo_id: int,
    ) -> tuple[int, ...]:
        saida: list[int] = []
        indice = 0
        while indice < len(sequencia):
            if (
                indice + 1 < len(sequencia)
                and sequencia[indice] == par[0]
                and sequencia[indice + 1] == par[1]
            ):
                saida.append(novo_id)
                indice += 2
            else:
                saida.append(int(sequencia[indice]))
                indice += 1
        return tuple(saida)

    def _inicializar(
        self,
        bytes_por_id: tuple[bytes, ...],
        mesclas: tuple[tuple[int, int], ...],
    ) -> None:
        especiais = ("<pad>", "<bos>", "<eos>", "<unk>")
        tokens_bytes = tuple(
            f"<byte:{valor:02x}>" for valor in range(256)
        )
        tokens_bpe = tuple(
            f"<bpe:{indice}:{valor.hex()}>"
            for indice, valor in enumerate(bytes_por_id[260:], start=260)
        )
        self.bytes_por_id = bytes_por_id
        self.mesclas = mesclas
        self.id_para_token = especiais + tokens_bytes + tokens_bpe
        self.token_para_id = {
            token: indice for indice, token in enumerate(self.id_para_token)
        }
        self._ranks = {
            par: (rank, 260 + rank)
            for rank, par in enumerate(mesclas)
        }
        # O corpus multimodal repete prompts e palavras em milhares de linhas.
        # O cache evita reaplicar todas as mesclas ao mesmo segmento, sem
        # alterar a sequencia deterministica produzida.
        self._cache_segmentos: dict[str, tuple[int, ...]] = {}

    @classmethod
    def de_vocabulario(
        cls,
        id_para_token: Iterable[str],
        mesclas: Iterable[Iterable[int]],
    ) -> "TokenizadorBPEBytes":
        pares = tuple(tuple(int(valor) for valor in par) for par in mesclas)
        bytes_por_id = [b""] * 4 + [bytes((valor,)) for valor in range(256)]
        for novo_id, par in enumerate(pares, start=260):
            if len(par) != 2 or min(par) < 4 or max(par) >= novo_id:
                raise ValueError("mescla BPE invalida")
            bytes_por_id.append(bytes_por_id[par[0]] + bytes_por_id[par[1]])
        tokenizador = cls.__new__(cls)
        tokenizador._inicializar(tuple(bytes_por_id), pares)
        if tuple(id_para_token) != tokenizador.id_para_token:
            raise ValueError("vocabulario BPE difere das mesclas salvas")
        return tokenizador

    def _codificar_segmento(self, segmento: str) -> tuple[int, ...]:
        armazenado = self._cache_segmentos.get(segmento)
        if armazenado is not None:
            return armazenado
        sequencia = [4 + valor for valor in segmento.encode("utf-8")]
        while len(sequencia) > 1:
            candidatos = [
                (self._ranks[(sequencia[indice], sequencia[indice + 1])], indice)
                for indice in range(len(sequencia) - 1)
                if (sequencia[indice], sequencia[indice + 1]) in self._ranks
            ]
            if not candidatos:
                break
            (_, novo_id), _ = min(candidatos)
            par = self.mesclas[novo_id - 260]
            sequencia = list(self._aplicar_par(sequencia, par, novo_id))
        resultado = tuple(sequencia)
        self._cache_segmentos[segmento] = resultado
        return resultado

    def codificar(
        self,
        texto: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")
        ids = [
            token_id
            for segmento in SEGMENTO_BPE_RE.findall(texto.lower())
            for token_id in self._codificar_segmento(segmento)
        ]
        return (
            ([self.bos_id] if bos else [])
            + ids
            + ([self.eos_id] if eos else [])
        )

    def tokenizar(self, texto: str) -> list[str]:
        return [
            self.id_para_token[token_id]
            for token_id in self.codificar(texto, bos=False, eos=False)
        ]

    def tokens_desconhecidos(self, texto: str) -> list[str]:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")
        return []

    def validar_texto_no_vocabulario(self, texto: str) -> None:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")

    def decodificar(self, ids: Iterable[int]) -> str:
        partes: list[bytes] = []
        for token_id in ids:
            indice = int(token_id)
            if indice in {self.pad_id, self.bos_id}:
                continue
            if indice == self.eos_id:
                break
            if 4 <= indice < len(self.bytes_por_id):
                partes.append(self.bytes_por_id[indice])
        texto = b"".join(partes).decode("utf-8", errors="replace")
        return texto[:1].upper() + texto[1:] if texto else texto

    @property
    def tamanho(self) -> int:
        return len(self.id_para_token)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def comprimentos_bytes(self) -> tuple[int, ...]:
        return tuple(len(valor) for valor in self.bytes_por_id)


class TokenizadorBPECodigo(TokenizadorBPEBytes):
    """BPE byte-level reversivel, case-sensitive e sem normalizacao.

    O treinamento usa contagens incrementais de pares para suportar
    vocabularios grandes sem recontar o corpus inteiro a cada mescla.
    """

    TIPO = "bpe_codigo_bytes_v1"
    FORMATO_SERIALIZADO = "bpe_codigo_bytes_v1"

    def __init__(
        self,
        textos: Iterable[str],
        tamanho_vocabulario: int = 8192,
    ) -> None:
        if tamanho_vocabulario < 260:
            raise ValueError("tamanho_vocabulario BPE deve ser pelo menos 260")
        frequencias: Counter[bytes] = Counter()
        for texto in textos:
            if not isinstance(texto, str):
                raise TypeError("textos do BPE devem ser str")
            frequencias.update(
                segmento.encode("utf-8")
                for segmento in SEGMENTO_BPE_RE.findall(texto)
            )
        if not frequencias:
            raise ValueError("nenhum texto fornecido ao BPE")

        segmentos = sorted(frequencias)
        sequencias = [
            [4 + valor for valor in segmento]
            for segmento in segmentos
        ]
        pesos = [frequencias[segmento] for segmento in segmentos]
        pares_globais: Counter[tuple[int, int]] = Counter()
        pares_por_segmento: list[Counter[tuple[int, int]]] = []
        segmentos_por_par: defaultdict[tuple[int, int], set[int]] = (
            defaultdict(set)
        )
        for indice, (sequencia, peso) in enumerate(zip(
            sequencias,
            pesos,
            strict=True,
        )):
            locais = Counter(zip(sequencia, sequencia[1:]))
            pares_por_segmento.append(locais)
            for par, quantidade in locais.items():
                pares_globais[par] += quantidade * peso
                segmentos_por_par[par].add(indice)

        heap: list[tuple[int, tuple[int, int]]] = [
            (-quantidade, par)
            for par, quantidade in pares_globais.items()
            if quantidade > 0
        ]
        heapq.heapify(heap)
        bytes_por_id = [b""] * 4 + [
            bytes((valor,)) for valor in range(256)
        ]
        mesclas: list[tuple[int, int]] = []

        while len(bytes_por_id) < tamanho_vocabulario:
            melhor_par: tuple[int, int] | None = None
            while heap:
                negativo, candidato = heapq.heappop(heap)
                atual = pares_globais.get(candidato, 0)
                if atual > 0 and -negativo == atual:
                    melhor_par = candidato
                    break
            if melhor_par is None:
                break

            novo_id = len(bytes_por_id)
            mesclas.append(melhor_par)
            bytes_por_id.append(
                bytes_por_id[melhor_par[0]]
                + bytes_por_id[melhor_par[1]]
            )
            afetados = tuple(segmentos_por_par.get(melhor_par, ()))
            for indice in afetados:
                antiga = sequencias[indice]
                nova = list(self._aplicar_par(antiga, melhor_par, novo_id))
                if nova == antiga:
                    continue
                peso = pesos[indice]
                antigos = pares_por_segmento[indice]
                novos = Counter(zip(nova, nova[1:]))
                todas = antigos.keys() | novos.keys()
                for par in todas:
                    delta = (novos.get(par, 0) - antigos.get(par, 0)) * peso
                    if delta:
                        quantidade = pares_globais.get(par, 0) + delta
                        if quantidade > 0:
                            pares_globais[par] = quantidade
                            heapq.heappush(heap, (-quantidade, par))
                        else:
                            pares_globais.pop(par, None)
                    if par in antigos and par not in novos:
                        grupo = segmentos_por_par.get(par)
                        if grupo is not None:
                            grupo.discard(indice)
                            if not grupo:
                                segmentos_por_par.pop(par, None)
                    elif par in novos and par not in antigos:
                        segmentos_por_par[par].add(indice)
                sequencias[indice] = nova
                pares_por_segmento[indice] = novos

        self._inicializar(tuple(bytes_por_id), tuple(mesclas))

    @property
    def serializado(self) -> dict[str, object] | str:
        legado = getattr(self, "_serializado_hf", None)
        if legado is not None:
            return legado
        return {
            "formato": self.FORMATO_SERIALIZADO,
            "mesclas": [list(par) for par in self.mesclas],
        }

    @classmethod
    def de_serializado(cls, conteudo: object) -> "TokenizadorBPECodigo":
        if isinstance(conteudo, str):
            # Compatibilidade com os shards BPE-8192 produzidos pela versao
            # anterior, que persistia diretamente o JSON do Hugging Face
            # Tokenizers. A delegacao preserva exatamente os IDs ja gravados.
            try:
                from tokenizers import Tokenizer
            except ImportError as erro:
                raise RuntimeError(
                    "o pacote tokenizers e necessario para ler o BPE legado"
                ) from erro
            externo = Tokenizer.from_str(conteudo)
            vocabulario = externo.get_vocab()
            tamanho = len(vocabulario)
            ids = [None] * tamanho
            for token, indice in vocabulario.items():
                if indice < 0 or indice >= tamanho or ids[indice] is not None:
                    raise ValueError("vocabulario BPE legado possui IDs invalidos")
                ids[indice] = token
            if any(token is None for token in ids):
                raise ValueError("vocabulario BPE legado possui lacunas")

            # Inverte o alfabeto byte-level usado pelo pre-tokenizador GPT-2.
            bytes_visiveis = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(0xA1, 0xAC + 1))
                + list(range(0xAE, 0xFF + 1))
            )
            codigos = list(bytes_visiveis)
            proximo = 0
            for valor in range(256):
                if valor not in bytes_visiveis:
                    bytes_visiveis.append(valor)
                    codigos.append(256 + proximo)
                    proximo += 1
            reverso = {
                chr(codigo): valor
                for valor, codigo in zip(bytes_visiveis, codigos, strict=True)
            }
            bytes_por_id: list[bytes] = []
            for indice, token in enumerate(ids):
                if indice < 4:
                    bytes_por_id.append(b"")
                    continue
                try:
                    bytes_por_id.append(bytes(reverso[caractere] for caractere in token))
                except KeyError:
                    # Alguns artefatos antigos contem o caractere de reposicao
                    # no vocabulario. Ele nao altera os IDs do corpus; para a
                    # metrica de bytes, usamos a propria decodificacao legada.
                    bytes_por_id.append(
                        externo.decode([indice], skip_special_tokens=True).encode(
                            "utf-8", errors="replace"
                        )
                    )

            tokenizador = cls.__new__(cls)
            tokenizador._tokenizador_hf = externo
            tokenizador._serializado_hf = conteudo
            tokenizador.bytes_por_id = tuple(bytes_por_id)
            tokenizador.mesclas = tuple()
            tokenizador.id_para_token = tuple(ids)
            tokenizador.token_para_id = {
                token: indice for indice, token in enumerate(tokenizador.id_para_token)
            }
            tokenizador._ranks = {}
            tokenizador._cache_segmentos = {}
            return tokenizador
        if not isinstance(conteudo, dict):
            raise TypeError("serializacao BPE de codigo deve ser um objeto ou texto")
        if conteudo.get("formato") != cls.FORMATO_SERIALIZADO:
            raise ValueError("formato BPE de codigo desconhecido")
        bruto = conteudo.get("mesclas")
        if not isinstance(bruto, list):
            raise ValueError("serializacao nao contem mesclas")
        pares: list[tuple[int, int]] = []
        bytes_por_id = [b""] * 4 + [
            bytes((valor,)) for valor in range(256)
        ]
        for novo_id, item in enumerate(bruto, start=260):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(valor, int) for valor in item)
            ):
                raise ValueError("mescla BPE de codigo invalida")
            par = (int(item[0]), int(item[1]))
            if min(par) < 4 or max(par) >= novo_id:
                raise ValueError("mescla BPE de codigo referencia ID invalido")
            pares.append(par)
            bytes_por_id.append(bytes_por_id[par[0]] + bytes_por_id[par[1]])
        tokenizador = cls.__new__(cls)
        tokenizador._inicializar(tuple(bytes_por_id), tuple(pares))
        return tokenizador

    def codificar(
        self,
        texto: str,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        if not isinstance(texto, str):
            raise TypeError("texto deve ser str")
        externo = getattr(self, "_tokenizador_hf", None)
        if externo is not None:
            ids = list(externo.encode(texto, add_special_tokens=False).ids)
        else:
            ids = [
                token_id
                for segmento in SEGMENTO_BPE_RE.findall(texto)
                for token_id in self._codificar_segmento(segmento)
            ]
        return (
            ([self.bos_id] if bos else [])
            + ids
            + ([self.eos_id] if eos else [])
        )

    def decodificar(self, ids: Iterable[int]) -> str:
        externos: list[int] = []
        partes: list[bytes] = []
        externo = getattr(self, "_tokenizador_hf", None)
        for token_id in ids:
            indice = int(token_id)
            if indice in {self.pad_id, self.bos_id}:
                continue
            if indice == self.eos_id:
                break
            if externo is not None:
                if 0 <= indice < len(self.id_para_token):
                    externos.append(indice)
            elif 4 <= indice < len(self.bytes_por_id):
                partes.append(self.bytes_por_id[indice])
        if externo is not None:
            return externo.decode(externos, skip_special_tokens=True)
        return b"".join(partes).decode("utf-8", errors="replace")


def tokenizador_de_checkpoint(checkpoint: dict):
    """Reconstrói tokenizadores novos e legados sem heurística insegura."""

    tipo = checkpoint.get("tokenizador", {}).get("tipo", "palavras_v1")
    if tipo == TokenizadorBytes.TIPO:
        return TokenizadorBytes.de_vocabulario(checkpoint["vocabulario"])
    if tipo == TokenizadorBPEBytes.TIPO:
        mesclas = checkpoint.get("tokenizador", {}).get("mesclas")
        if mesclas is None:
            raise ValueError("checkpoint BPE nao contem mesclas")
        return TokenizadorBPEBytes.de_vocabulario(
            checkpoint["vocabulario"],
            mesclas,
        )
    if tipo == TokenizadorBPECodigo.TIPO:
        serializado = checkpoint.get("tokenizador", {}).get("serializado")
        tokenizador = TokenizadorBPECodigo.de_serializado(serializado)
        if tuple(checkpoint["vocabulario"]) != tokenizador.id_para_token:
            raise ValueError("vocabulario BPE de codigo diverge da serializacao")
        return tokenizador
    if tipo == "palavras_v1":
        return TokenizadorPalavras.de_vocabulario(checkpoint["vocabulario"])
    raise ValueError(f"tipo de tokenizador desconhecido: {tipo}")
