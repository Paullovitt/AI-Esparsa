"""Mini-LM causal para comparar atencao densa e atencao esparsa por conteudo.

O experimento isola a camada de atencao: embedding, normalizacao e cabeca de
vocabulario sao identicos nos dois modelos. A variante esparsa nao usa
projecoes densas em Q/K nem um roteador denso.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ENTIDADES = ("paulo", "ana", "bruno", "carla", "diego", "elisa", "fabio", "giulia")
OBJETOS = (
    ("celular", "telefone"),
    ("livro", "volume"),
    ("chave", "chavinha"),
    ("caderno", "bloco"),
    ("relogio", "cronometro"),
    ("carteira", "documento"),
    ("oculos", "lentes"),
    ("mochila", "bolsa"),
)
LOCAIS = ("gaveta", "armario", "caixa", "mesa", "cozinha", "quarto", "estante", "escritorio")
GRAMATICA = (
    "fato",
    "pergunta",
    "resposta",
    "onde",
    "esta",
    "o",
    "a",
    "de",
    "em",
    "na",
    "no",
    "guardou",
    "?",
    ".",
)
ESPECIAIS = ("<pad>", "<bos>", "<eos>")


@dataclass(frozen=True)
class VocabularioTexto:
    """Mapeamento pequeno e deterministico usado pelo corpus controlado."""

    token_para_id: dict[str, int]
    id_para_token: tuple[str, ...]
    objeto_para_grupo: dict[str, int]

    def id(self, token: str) -> int:
        return self.token_para_id[token]

    def codificar(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_para_id[token] for token in tokens]

    def decodificar(self, ids: Iterable[int]) -> list[str]:
        return [self.id_para_token[int(token_id)] for token_id in ids]

    @property
    def tamanho(self) -> int:
        return len(self.id_para_token)

    @property
    def pad_id(self) -> int:
        return self.id("<pad>")


def criar_vocabulario() -> VocabularioTexto:
    """Cria um vocabulario sem duplicatas e com ordem estavel."""

    tokens = list(ESPECIAIS) + list(GRAMATICA) + list(ENTIDADES)
    tokens.extend(alias for grupo in OBJETOS for alias in grupo)
    tokens.extend(LOCAIS)
    token_para_id = {token: indice for indice, token in enumerate(tokens)}
    objeto_para_grupo = {
        alias: grupo for grupo, aliases in enumerate(OBJETOS) for alias in aliases
    }
    return VocabularioTexto(token_para_id, tuple(tokens), objeto_para_grupo)


@dataclass(frozen=True)
class ConfiguracaoTextoCausal:
    """Hiperparametros compartilhados pelas duas variantes de atencao."""

    dimensao_modelo: int = 48
    dimensao_features: int = 512
    features_por_token: int = 2
    janela_local: int = 8
    top_k: int = 24
    fatos_por_historia: int = 3


@dataclass
class DadosTextoCausal:
    """Tensores de uma divisao do corpus e metadados para avaliacao."""

    tokens: Tensor
    mascara_resposta_local: Tensor
    posicao_fato_origem: Tensor
    triplas: Tensor
    aliases_fato: Tensor
    candidatos: Tensor | None = None
    candidatos_validos: Tensor | None = None

    def para(self, dispositivo: torch.device | str) -> "DadosTextoCausal":
        argumentos = {
            campo: valor.to(dispositivo) if isinstance(valor, Tensor) else valor
            for campo, valor in vars(self).items()
        }
        return DadosTextoCausal(**argumentos)

    def selecionar(self, indices: Tensor | Sequence[int]) -> "DadosTextoCausal":
        argumentos = {}
        for campo, valor in vars(self).items():
            argumentos[campo] = valor[indices] if isinstance(valor, Tensor) else valor
        return DadosTextoCausal(**argumentos)

    def com_candidatos(self, candidatos: Tensor, validos: Tensor) -> "DadosTextoCausal":
        return replace(self, candidatos=candidatos, candidatos_validos=validos)

    @property
    def entradas(self) -> Tensor:
        return self.tokens[:, :-1]

    @property
    def alvos(self) -> Tensor:
        return self.tokens[:, 1:]


def _sorteio_permutacao(gerador: torch.Generator, tamanho: int, quantidade: int) -> list[int]:
    return torch.randperm(tamanho, generator=gerador)[:quantidade].tolist()


def _escolher_local(
    gerador: torch.Generator,
    entidade: int,
    objeto: int,
    usar_combinacoes_ineditas: bool,
) -> int:
    candidatos = [
        local
        for local in range(len(LOCAIS))
        if ((entidade + objeto + local) % 4 == 0) == usar_combinacoes_ineditas
    ]
    indice = int(torch.randint(len(candidatos), (1,), generator=gerador))
    return candidatos[indice]


def gerar_corpus_texto_causal(
    quantidade: int,
    semente: int,
    vocabulario: VocabularioTexto,
    configuracao: ConfiguracaoTextoCausal,
    usar_combinacoes_ineditas: bool,
) -> DadosTextoCausal:
    """Gera historias fixas com fatos e perguntas que usam aliases opostos.

    A particao e estrutural: treino usa triplas cuja soma modulo quatro nao e
    zero; validacao/teste usam apenas as combinacoes reservadas.
    """

    gerador = torch.Generator().manual_seed(semente)
    historias: list[list[int]] = []
    mascaras: list[list[bool]] = []
    origens: list[list[int]] = []
    todas_triplas: list[list[list[int]]] = []
    todos_aliases: list[list[int]] = []

    for _ in range(quantidade):
        entidades = _sorteio_permutacao(gerador, len(ENTIDADES), configuracao.fatos_por_historia)
        objetos = _sorteio_permutacao(gerador, len(OBJETOS), configuracao.fatos_por_historia)
        locais = [
            _escolher_local(gerador, entidade, objeto, usar_combinacoes_ineditas)
            for entidade, objeto in zip(entidades, objetos)
        ]
        aliases_fato = torch.randint(
            0, 2, (configuracao.fatos_por_historia,), generator=gerador
        ).tolist()

        texto = ["<bos>"]
        posicoes_fato: list[int] = []
        for entidade, objeto, local, alias_fato in zip(
            entidades, objetos, locais, aliases_fato
        ):
            texto.extend(
                (
                    "fato",
                    ENTIDADES[entidade],
                    "guardou",
                    OBJETOS[objeto][alias_fato],
                    "em",
                    LOCAIS[local],
                    ".",
                )
            )
            posicoes_fato.append(len(texto) - 2)

        posicoes_resposta: list[int] = []
        origens_resposta: list[int] = []
        for indice, (entidade, objeto, local, alias_fato) in enumerate(
            zip(entidades, objetos, locais, aliases_fato)
        ):
            alias_pergunta = 1 - alias_fato
            artigo = "a" if objeto in (2, 5, 7) else "o"
            texto.extend(
                (
                    "pergunta",
                    "onde",
                    "esta",
                    artigo,
                    OBJETOS[objeto][alias_pergunta],
                    "de",
                    ENTIDADES[entidade],
                    "?",
                    "resposta",
                    artigo,
                    OBJETOS[objeto][alias_pergunta],
                    "de",
                    ENTIDADES[entidade],
                    "esta",
                    "na",
                )
            )
            posicoes_resposta.append(len(texto))
            origens_resposta.append(posicoes_fato[indice])
            texto.extend((LOCAIS[local], "."))
        texto.append("<eos>")

        ids = vocabulario.codificar(texto)
        mascara = [False] * (len(ids) - 1)
        origem = [-1] * (len(ids) - 1)
        for posicao_resposta, posicao_fato in zip(posicoes_resposta, origens_resposta):
            # O logit na posicao anterior deve prever o local da resposta.
            mascara[posicao_resposta - 1] = True
            origem[posicao_resposta - 1] = posicao_fato

        historias.append(ids)
        mascaras.append(mascara)
        origens.append(origem)
        todas_triplas.append(
            [[entidade, objeto, local] for entidade, objeto, local in zip(entidades, objetos, locais)]
        )
        todos_aliases.append(aliases_fato)

    comprimentos = {len(historia) for historia in historias}
    if len(comprimentos) != 1:
        raise RuntimeError(f"O corpus deveria ter comprimento fixo, recebeu {comprimentos}.")

    return DadosTextoCausal(
        tokens=torch.tensor(historias, dtype=torch.long),
        mascara_resposta_local=torch.tensor(mascaras, dtype=torch.bool),
        posicao_fato_origem=torch.tensor(origens, dtype=torch.long),
        triplas=torch.tensor(todas_triplas, dtype=torch.long),
        aliases_fato=torch.tensor(todos_aliases, dtype=torch.long),
    )


class CodigoEsparsoCausal(nn.Module):
    """Codifica entidades, aliases e papeis sem um roteador neural denso."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao: ConfiguracaoTextoCausal,
        semente: int = 20260726,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        self.vocabulario = vocabulario
        if configuracao.dimensao_features < 128:
            raise ValueError("dimensao_features deve ser pelo menos 128.")

        gerador = torch.Generator().manual_seed(semente)
        codigos = torch.zeros(
            vocabulario.tamanho, configuracao.features_por_token, dtype=torch.long
        )
        limite = configuracao.dimensao_features - 2
        usados: set[tuple[int, ...]] = set()

        def novo_codigo() -> Tensor:
            while True:
                codigo = torch.randperm(limite, generator=gerador)[
                    : configuracao.features_por_token
                ] + 1
                chave = tuple(sorted(codigo.tolist()))
                if chave not in usados:
                    usados.add(chave)
                    return codigo

        for entidade in ENTIDADES:
            codigos[vocabulario.id(entidade)] = novo_codigo()
        for aliases in OBJETOS:
            codigo = novo_codigo()
            for alias in aliases:
                codigos[vocabulario.id(alias)] = codigo

        mapa_entidade = torch.zeros(vocabulario.tamanho, dtype=torch.bool)
        mapa_objeto = torch.zeros(vocabulario.tamanho, dtype=torch.bool)
        mapa_local = torch.zeros(vocabulario.tamanho, dtype=torch.bool)
        for entidade in ENTIDADES:
            mapa_entidade[vocabulario.id(entidade)] = True
        for aliases in OBJETOS:
            for alias in aliases:
                mapa_objeto[vocabulario.id(alias)] = True
        for local in LOCAIS:
            mapa_local[vocabulario.id(local)] = True
        mapa_local[vocabulario.id("na")] = True
        mapa_local[vocabulario.id("no")] = True

        self.register_buffer("codigos_token", codigos)
        self.register_buffer("mapa_entidade", mapa_entidade)
        self.register_buffer("mapa_objeto", mapa_objeto)
        self.register_buffer("mapa_local", mapa_local)
        self.register_buffer(
            "codigo_papel",
            torch.tensor(
                [configuracao.dimensao_features - 1, configuracao.dimensao_features],
                dtype=torch.long,
            ),
        )

    @property
    def largura_descritor(self) -> int:
        return self.configuracao.features_por_token * 3

    def descritores(self, tokens: Tensor) -> Tensor:
        """Monta descritores usando somente o token atual e o passado local."""

        lote, comprimento = tokens.shape
        ultimo_entidade = torch.full_like(tokens, self.vocabulario.pad_id)
        ultimo_objeto = torch.full_like(tokens, self.vocabulario.pad_id)

        for deslocamento in range(self.configuracao.janela_local):
            deslocados = torch.full_like(tokens, self.vocabulario.pad_id)
            if deslocamento == 0:
                deslocados = tokens
            elif deslocamento < comprimento:
                deslocados[:, deslocamento:] = tokens[:, :-deslocamento]

            falta_entidade = ultimo_entidade.eq(self.vocabulario.pad_id)
            falta_objeto = ultimo_objeto.eq(self.vocabulario.pad_id)
            ultimo_entidade = torch.where(
                falta_entidade & self.mapa_entidade[deslocados],
                deslocados,
                ultimo_entidade,
            )
            ultimo_objeto = torch.where(
                falta_objeto & self.mapa_objeto[deslocados],
                deslocados,
                ultimo_objeto,
            )

        codigo_entidade = self.codigos_token[ultimo_entidade]
        codigo_objeto = self.codigos_token[ultimo_objeto]
        codigo_papel = self.codigo_papel.view(1, 1, -1).expand(lote, comprimento, -1)
        codigo_papel = torch.where(
            self.mapa_local[tokens].unsqueeze(-1),
            codigo_papel,
            torch.zeros_like(codigo_papel),
        )
        return torch.cat((codigo_entidade, codigo_objeto, codigo_papel), dim=-1)


def selecionar_candidatos_causais(
    descritores: Tensor,
    top_k: int,
    tamanho_bloco: int = 128,
) -> tuple[Tensor, Tensor]:
    """Seleciona Top-K por sobreposicao de features, respeitando causalidade."""

    lote, comprimento, _ = descritores.shape
    quantidade = min(top_k, comprimento)
    candidatos_blocos: list[Tensor] = []
    validos_blocos: list[Tensor] = []
    posicoes = torch.arange(comprimento, device=descritores.device)
    origem = posicoes.view(1, 1, -1)
    consulta = posicoes.view(1, -1, 1)
    causal = origem < consulta
    causal[:, 0, 0] = True

    for inicio in range(0, lote, tamanho_bloco):
        consulta = descritores[inicio : inicio + tamanho_bloco]
        # Cada feature da consulta vota no token historico que contem a mesma feature.
        igualdade = consulta[:, :, None, :, None].eq(consulta[:, None, :, None, :])
        nao_zero = consulta[:, :, None, :, None].ne(0)
        sobreposicao = (igualdade & nao_zero).any(dim=-1).sum(dim=-1)
        pontuacao = sobreposicao.to(torch.float32)
        pontuacao = pontuacao.masked_fill(~causal, -torch.inf)
        # O pequeno desempate favorece a memoria mais recente.
        # 1e-4 permanece abaixo de um nivel de overlap, mas e grande o
        # suficiente para produzir um desempate deterministico em FP32.
        pontuacao = pontuacao + posicoes.to(torch.float32).view(1, 1, -1) * 1e-4
        candidatos = pontuacao.topk(quantidade, dim=-1, sorted=True).indices
        validos = candidatos < posicoes.view(1, -1, 1)
        validos[:, 0] = candidatos[:, 0].eq(0)
        candidatos_blocos.append(candidatos)
        validos_blocos.append(validos)

    return torch.cat(candidatos_blocos), torch.cat(validos_blocos)


class MiniLMCausal(nn.Module):
    """Mini-LM sem FFN que permite trocar apenas o mecanismo de atencao."""

    def __init__(
        self,
        vocabulario: VocabularioTexto,
        configuracao: ConfiguracaoTextoCausal,
        modo: str,
    ) -> None:
        super().__init__()
        if modo not in {"densa", "esparsa"}:
            raise ValueError("modo deve ser 'densa' ou 'esparsa'.")
        self.vocabulario = vocabulario
        self.configuracao = configuracao
        self.modo = modo

        # Estes componentes sao compartilhados para que o teste isole a atencao.
        self.embedding = nn.Embedding(
            vocabulario.tamanho, configuracao.dimensao_modelo, padding_idx=vocabulario.pad_id
        )
        self.normalizacao = nn.LayerNorm(configuracao.dimensao_modelo)
        self.saida = nn.Linear(configuracao.dimensao_modelo, vocabulario.tamanho)
        self.gate_atencao = nn.Parameter(torch.tensor(-0.5))

        if modo == "densa":
            self.q_densa = nn.Linear(
                configuracao.dimensao_modelo, configuracao.dimensao_modelo, bias=False
            )
            self.k_densa = nn.Linear(
                configuracao.dimensao_modelo, configuracao.dimensao_modelo, bias=False
            )
        else:
            self.q_feature = nn.Embedding(
                configuracao.dimensao_features + 1, 1, padding_idx=0, sparse=True
            )
            self.k_feature = nn.Embedding(
                configuracao.dimensao_features + 1, 1, padding_idx=0, sparse=True
            )
            nn.init.ones_(self.q_feature.weight)
            nn.init.ones_(self.k_feature.weight)
            with torch.no_grad():
                self.q_feature.weight[0].zero_()
                self.k_feature.weight[0].zero_()

    def estados_locais(self, tokens: Tensor, embedding: Tensor | None = None) -> Tensor:
        """Resume uma janela causal curta antes da atencao de longo alcance."""

        if embedding is None:
            embedding = self.embedding(tokens)
        soma = torch.zeros_like(embedding)
        contagem = torch.zeros(
            tokens.shape[0], tokens.shape[1], 1, device=tokens.device, dtype=embedding.dtype
        )
        for deslocamento in range(self.configuracao.janela_local):
            if deslocamento == 0:
                soma = soma + embedding
                contagem = contagem + 1
            elif deslocamento < tokens.shape[1]:
                soma[:, deslocamento:] = soma[:, deslocamento:] + embedding[:, :-deslocamento]
                contagem[:, deslocamento:] = contagem[:, deslocamento:] + 1
        return soma / contagem.clamp_min(1)

    def _atencao_densa(self, estados: Tensor, valores: Tensor) -> tuple[Tensor, Tensor]:
        q = self.q_densa(estados)
        k = self.k_densa(estados)
        pontuacao = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mascara_bloqueada = torch.ones(
            estados.shape[1], estados.shape[1], device=estados.device, dtype=torch.bool
        ).triu(0)
        mascara_bloqueada[0, 0] = False
        pontuacao = pontuacao.masked_fill(mascara_bloqueada, -torch.inf)
        pesos = F.softmax(pontuacao, dim=-1)
        contexto = torch.matmul(pesos, valores)
        return contexto, pesos.argmax(dim=-1)

    def _atencao_esparsa(
        self,
        estados: Tensor,
        valores: Tensor,
        descritores: Tensor,
        candidatos: Tensor,
        candidatos_validos: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lote = estados.shape[0]
        indice_lote = torch.arange(lote, device=estados.device).view(lote, 1, 1)
        valores_candidatos = valores[indice_lote, candidatos]
        descritores_candidatos = descritores[indice_lote, candidatos]

        q = self.q_feature(descritores).squeeze(-1)
        k = self.k_feature(descritores_candidatos).squeeze(-1)
        igualdade = descritores.unsqueeze(2).unsqueeze(-1).eq(
            descritores_candidatos.unsqueeze(-2)
        )
        nao_zero = descritores.unsqueeze(2).unsqueeze(-1).ne(0)
        contribuicoes = igualdade & nao_zero
        pontuacao = (
            contribuicoes.to(estados.dtype)
            * q.unsqueeze(2).unsqueeze(-1)
            * k.unsqueeze(-2)
        ).sum(dim=(-1, -2))
        pontuacao = pontuacao / math.sqrt(float(descritores.shape[-1]))
        pontuacao = pontuacao.masked_fill(~candidatos_validos, -torch.inf)
        pesos = F.softmax(pontuacao, dim=-1)
        contexto = (pesos.unsqueeze(-1) * valores_candidatos).sum(dim=2)
        melhor = pesos.argmax(dim=-1, keepdim=True)
        posicao = candidatos.gather(-1, melhor).squeeze(-1)
        return contexto, posicao

    def estado_contextual(
        self,
        tokens: Tensor,
        descritores: Tensor | None = None,
        candidatos: Tensor | None = None,
        candidatos_validos: Tensor | None = None,
        usar_atencao: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Retorna o estado apos atencao para composicao com outros blocos."""

        embedding = self.embedding(tokens)
        estados = self.estados_locais(tokens, embedding)
        if not usar_atencao:
            contexto = torch.zeros_like(estados)
            posicao = torch.full_like(tokens, -1)
        elif self.modo == "densa":
            contexto, posicao = self._atencao_densa(estados, embedding)
        else:
            if descritores is None or candidatos is None or candidatos_validos is None:
                raise ValueError("A atencao esparsa requer descritores e candidatos.")
            contexto, posicao = self._atencao_esparsa(
                estados, embedding, descritores, candidatos, candidatos_validos
            )

        gate = torch.sigmoid(self.gate_atencao)
        oculto = self.normalizacao(estados + gate * contexto)
        return oculto, posicao

    def forward(
        self,
        tokens: Tensor,
        descritores: Tensor | None = None,
        candidatos: Tensor | None = None,
        candidatos_validos: Tensor | None = None,
        usar_atencao: bool = True,
    ) -> tuple[Tensor, Tensor]:
        oculto, posicao = self.estado_contextual(
            tokens,
            descritores,
            candidatos,
            candidatos_validos,
            usar_atencao,
        )
        return self.saida(oculto), posicao

    def parametros_esparsos(self) -> list[nn.Parameter]:
        if self.modo != "esparsa":
            return []
        return [self.q_feature.weight, self.k_feature.weight]

    def parametros_densos(self) -> list[nn.Parameter]:
        ids_esparsos = {id(parametro) for parametro in self.parametros_esparsos()}
        return [parametro for parametro in self.parameters() if id(parametro) not in ids_esparsos]

    def auditoria(self) -> dict[str, int | float | str]:
        total = sum(parametro.numel() for parametro in self.parameters())
        if self.modo == "densa":
            qk = self.q_densa.weight.numel() + self.k_densa.weight.numel()
        else:
            qk = self.q_feature.weight.numel() + self.k_feature.weight.numel()
        return {
            "modo": self.modo,
            "parametros_totais": total,
            "parametros_qk": qk,
            "parametros_densos_qk": qk if self.modo == "densa" else 0,
            "parametros_esparsos_qk": qk if self.modo == "esparsa" else 0,
            "top_k": self.configuracao.top_k if self.modo == "esparsa" else -1,
        }


def preparar_candidatos(
    dados: DadosTextoCausal,
    codigo: CodigoEsparsoCausal,
) -> tuple[DadosTextoCausal, Tensor]:
    """Calcula descritores e indice causal uma unica vez para treino/avaliacao."""

    descritores = codigo.descritores(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores, codigo.configuracao.top_k
    )
    return dados.com_candidatos(candidatos, validos), descritores


def formatar_tokens(tokens: Sequence[str]) -> str:
    """Converte tokens do corpus em texto legivel sem depender de tokenizer."""

    texto = " ".join(tokens)
    texto = texto.replace(" .", ".").replace(" ?", "?")
    return texto.replace("<bos> ", "").replace(" <eos>", "")
