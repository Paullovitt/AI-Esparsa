"""Geração determinística do corpus e preparação dos lotes.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import torch
from torch import Tensor

from .corpus_gerador_esparso import (
    ACOES,
    LOCAIS as LOCAIS_LONGOS,
    OBJETOS,
    PESSOAS,
    PROBLEMAS,
    TAREFAS,
    TEMPOS,
    gerar_registro_curto,
)
from .tokenizador_palavras import TokenizadorPalavras


QUANTIDADES_PADRAO = (50_000, 1_000, 1_000)
INICIOS_PADRAO = (40_000_000, 50_000_000, 60_000_000)

def gerar_registro_extenso(semente: int) -> dict[str, object]:
    """Gera um relato longo com fatos estáveis e 24 frases coerentes."""

    aleatorio = random.Random(semente)
    (pessoa, pronome), (ajudante, pronome_ajudante) = aleatorio.sample(
        PESSOAS,
        2,
    )
    tarefa = aleatorio.choice(TAREFAS)
    objeto_principal, objeto_apoio = aleatorio.sample(OBJETOS, 2)
    local_principal, local_apoio = aleatorio.sample(LOCAIS_LONGOS, 2)
    problema_curto, problema_frase = aleatorio.choice(PROBLEMAS)
    acoes = aleatorio.sample(ACOES, 4)
    tempo = aleatorio.choice(TEMPOS)

    artigo, objeto, _ = objeto_principal
    artigo_apoio, nome_apoio, _ = objeto_apoio
    preposicao, local = local_principal
    preposicao_apoio, lugar_apoio = local_apoio
    colaborador = (
        "disposta" if pronome_ajudante == "ela" else "disposto"
    )
    artigo_ajudante = "A" if pronome_ajudante == "ela" else "O"
    grupo = (
        "As duas"
        if pronome == "ela" and pronome_ajudante == "ela"
        else "Os dois"
    )
    pronome_grupo = (
        "elas"
        if pronome == "ela" and pronome_ajudante == "ela"
        else "eles"
    )
    guardado = "guardada" if artigo == "a" else "guardado"
    pronome_objeto = "ela" if artigo == "a" else "ele"
    levado_apoio = "levada" if artigo_apoio == "a" else "levado"
    destino_apoio = (
        "para a" if preposicao_apoio == "na" else "para o"
    )

    pedido = (
        f"Pedido: escreva um relato sobre {pessoa}, com ajuda de "
        f"{ajudante}, para {tarefa}. Inclua {artigo} {objeto} "
        f"{preposicao} {local} e o problema {problema_curto}. Texto:"
    )
    # O molde longo mantém o mesmo fato principal do começo ao fim e varia
    # agentes, tarefa, objetos, locais, causa e quatro ações do domínio.
    frases = [
        (
            f"{tempo}, {pessoa} recebeu a responsabilidade de {tarefa} e "
            f"chamou {ajudante} para participar do trabalho"
        ),
        (
            f"{grupo} revisaram o objetivo da atividade, dividiram as "
            f"etapas e combinaram registrar cada decisão importante"
        ),
        (
            f"{pessoa.capitalize()} explicou as prioridades, enquanto "
            f"{ajudante}, {colaborador} a colaborar, preparou uma lista "
            f"com os materiais disponíveis"
        ),
        (
            f"Antes do início, a dupla inspecionou o espaço, verificou as "
            f"condições de segurança e deixou livre a área de circulação"
        ),
        (
            f"O primeiro passo foi {acoes[0]}, pois essa medida facilitaria "
            f"todas as atividades seguintes"
        ),
        (
            f"Logo depois, {ajudante} começou a {acoes[1]}, e {pessoa} "
            f"acompanhou o resultado para evitar qualquer troca indevida"
        ),
        (
            f"Para proteger o item principal, {pessoa} colocou {artigo} "
            f"{objeto} {preposicao} {local} e mostrou a posição a "
            f"{ajudante}"
        ),
        (
            f"{artigo_ajudante} ajudante confirmou a localização, anotou que "
            f"{pronome_objeto} estava {guardado} corretamente e manteve o "
            f"acesso desimpedido"
        ),
        (
            f"Em outra parte do ambiente, {artigo_apoio} {nome_apoio} foi "
            f"{levado_apoio} {destino_apoio} {lugar_apoio}, sem interferir no "
            f"local reservado para {artigo} {objeto}"
        ),
        (
            f"Durante a execução, {problema_frase}, criando uma pausa que "
            f"não estava prevista no cronograma"
        ),
        (
            f"A ocorrência exigiu calma, porque uma resposta apressada "
            f"poderia comprometer os materiais e aumentar o atraso"
        ),
        (
            f"{pessoa.capitalize()} protegeu os itens próximos, avisou a "
            f"equipe e conferiu novamente se {artigo} {objeto} permanecia "
            f"{preposicao} {local}"
        ),
        (
            f"{ajudante.capitalize()} analisou a causa da interrupção, "
            f"comparou as alternativas e propôs um caminho seguro para a "
            f"continuidade"
        ),
        (
            f"Depois de conversarem, {grupo.lower()} reorganizaram o "
            f"cronograma sem "
            f"mudar a responsabilidade de cada participante"
        ),
        (
            f"Na retomada, {pronome_grupo} decidiram {acoes[2]} e "
            f"documentaram o motivo da escolha para que ninguém perdesse o "
            f"contexto"
        ),
        (
            f"Em seguida, {pessoa} ficou responsável por {acoes[3]}, "
            f"enquanto {ajudante} verificava o andamento das etapas já "
            f"concluídas"
        ),
        (
            f"A comunicação constante permitiu corrigir uma pequena "
            f"diferença na lista antes que ela afetasse o restante do plano"
        ),
        (
            f"Quando a situação se estabilizou, {pronome} voltou ao ponto "
            f"de armazenamento e observou que {artigo} {objeto} seguia "
            f"{preposicao} {local}"
        ),
        (
            f"{ajudante.capitalize()} fez uma segunda conferência e declarou "
            f"que a informação registrada correspondia ao que estava no "
            f"ambiente"
        ),
        (
            f"A dupla então concluiu as etapas pendentes, revisou os "
            f"materiais utilizados e recolheu apenas o que não seria mais "
            f"necessário"
        ),
        (
            f"Antes de encerrar, {pessoa} explicou como o problema foi "
            f"controlado e quais cuidados evitaram novas perdas durante a "
            f"atividade"
        ),
        (
            f"No relatório final, {pronome} confirmou que {artigo} {objeto} "
            f"continuava {preposicao} {local}, exatamente como havia sido "
            f"planejado"
        ),
        (
            f"{ajudante.capitalize()} releu as anotações, concordou com a "
            f"conclusão e acrescentou as medidas preventivas adotadas pela "
            f"equipe"
        ),
        (
            f"Com a tarefa concluída, {pessoa} e {ajudante} deixaram o "
            f"espaço organizado e entregaram um registro completo do "
            f"trabalho realizado"
        ),
    ]
    historia = ". ".join(frases) + "."
    chave = (
        pessoa,
        ajudante,
        tarefa,
        objeto,
        local,
        problema_curto,
        tuple(acoes),
    )
    return {
        "pedido": pedido,
        "historia": historia,
        "texto": f"{pedido} {historia}",
        "perguntas": [],
        "palavras_chave": [
            pessoa,
            ajudante,
            tarefa,
            objeto,
            local,
            problema_curto,
        ],
        "chave": chave,
        "quantidade_frases": len(frases),
    }


def gerar_divisoes_gerador(
    quantidades: tuple[int, int, int] = QUANTIDADES_PADRAO,
    inicios: tuple[int, int, int] = INICIOS_PADRAO,
    *,
    relatos_extensos: bool = True,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Gera combinacoes disjuntas para treino, validacao e teste."""

    if len(quantidades) != 3 or len(inicios) != 3:
        raise ValueError("sao necessarias tres divisoes")
    if any(quantidade <= 0 for quantidade in quantidades):
        raise ValueError("as quantidades devem ser positivas")

    usados: set[tuple[object, ...]] = set()
    divisoes: list[list[dict[str, object]]] = []
    for quantidade, inicio in zip(quantidades, inicios):
        registros: list[dict[str, object]] = []
        tentativa = 0
        while len(registros) < quantidade:
            semente_registro = inicio + tentativa * 7_919
            registro = (
                gerar_registro_extenso(semente_registro)
                if relatos_extensos
                else gerar_registro_curto(
                    semente_registro,
                    corrigir_gramatica=True,
                )
            )
            tentativa += 1
            chave = tuple(registro["chave"])
            if chave in usados:
                continue
            usados.add(chave)
            registros.append(registro)
        divisoes.append(registros)
    return divisoes[0], divisoes[1], divisoes[2]


def _indices_historia(
    ids: list[int],
    tokenizador: TokenizadorPalavras,
) -> tuple[int, int]:
    """Localiza a historia entre ``texto:`` e o primeiro ``pergunta:``."""

    texto_id = tokenizador.token_para_id["texto"]
    pergunta_id = tokenizador.token_para_id.get("pergunta")
    dois_pontos_id = tokenizador.token_para_id[":"]
    inicio = next(
        indice + 2
        for indice in range(len(ids) - 1)
        if ids[indice] == texto_id
        and ids[indice + 1] == dois_pontos_id
    )
    fim = next(
        (
            indice
            for indice in range(inicio, len(ids))
            if pergunta_id is not None
            and ids[indice] == pergunta_id
            and indice + 1 < len(ids)
            and ids[indice + 1] == dois_pontos_id
        ),
        len(ids) - 1,
    )
    if inicio >= fim:
        raise ValueError("historia vazia ou marcadores fora de ordem")
    return inicio, fim


def codificar_registros(
    registros: list[dict[str, object]],
    tokenizador: TokenizadorPalavras,
) -> list[tuple[list[int], list[float]]]:
    """Codifica os textos e reforca a continuacao condicionada pelo pedido."""

    codificados: list[tuple[list[int], list[float]]] = []
    for registro in registros:
        ids = tokenizador.codificar(str(registro["texto"]))
        inicio_historia, fim_historia = _indices_historia(
            ids,
            tokenizador,
        )
        pesos = [1.0] * (len(ids) - 1)
        ids_pedido = set(
            tokenizador.codificar(
                str(registro["pedido"]),
                bos=False,
                eos=False,
            )
        )
        # O indice do peso corresponde ao logit anterior ao token-alvo.
        for posicao_alvo in range(inicio_historia, fim_historia):
            fator = 3.0 if ids[posicao_alvo] in ids_pedido else 2.0
            pesos[posicao_alvo - 1] = fator
        codificados.append((ids, pesos))
    return codificados


def criar_lotes(
    codificados: list[tuple[list[int], list[float]]],
    pad_id: int,
    tamanho_lote: int,
    dispositivo: torch.device,
    *,
    embaralhar: bool,
    semente: int,
) -> Iterable[tuple[Tensor, Tensor, Tensor]]:
    """Cria lotes com mascara de perda sem misturar as divisoes."""

    if tamanho_lote <= 0:
        raise ValueError("tamanho_lote deve ser positivo")
    ordem = list(range(len(codificados)))
    if embaralhar:
        random.Random(semente).shuffle(ordem)
    for inicio in range(0, len(ordem), tamanho_lote):
        itens = [
            codificados[indice]
            for indice in ordem[inicio : inicio + tamanho_lote]
        ]
        maior = max(len(ids) for ids, _ in itens)
        entradas = torch.full(
            (len(itens), maior - 1),
            pad_id,
            dtype=torch.long,
        )
        alvos = torch.full_like(entradas, pad_id)
        pesos_lote = torch.zeros_like(entradas, dtype=torch.float32)
        for linha, (ids, pesos) in enumerate(itens):
            valores = torch.tensor(ids, dtype=torch.long)
            entradas[linha, : len(ids) - 1] = valores[:-1]
            alvos[linha, : len(ids) - 1] = valores[1:]
            pesos_lote[linha, : len(pesos)] = torch.tensor(pesos)
        yield (
            entradas.to(dispositivo, non_blocking=True),
            alvos.to(dispositivo, non_blocking=True),
            pesos_lote.to(dispositivo, non_blocking=True),
        )
