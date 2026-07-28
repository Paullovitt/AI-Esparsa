"""Vocabulário e gerador procedural compartilhados pela base esparsa.

O módulo contém apenas dados e geração determinística de texto, mantendo o
treinamento independente de fontes externas e artefatos auxiliares.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import random


PESSOAS = [
    ("ana", "ela"),
    ("beatriz", "ela"),
    ("carla", "ela"),
    ("marina", "ela"),
    ("bruno", "ele"),
    ("diego", "ele"),
    ("lucas", "ele"),
    ("paulo", "ele"),
    ("renata", "ela"),
    ("tiago", "ele"),
]

TAREFAS = [
    "organizar a biblioteca do bairro",
    "preparar uma feira de ciencias",
    "restaurar uma sala de estudos",
    "montar uma exposicao de fotografias",
    "planejar uma horta comunitaria",
    "revisar um arquivo historico",
    "preparar uma oficina de leitura",
    "organizar uma pequena mostra cultural",
]

OBJETOS = [
    ("o", "livro", "guardado"),
    ("o", "caderno", "guardado"),
    ("o", "documento", "guardado"),
    ("o", "mapa", "guardado"),
    ("o", "relatorio", "guardado"),
    ("a", "chave", "guardada"),
    ("a", "caixa", "guardada"),
    ("a", "pasta", "guardada"),
    ("a", "camera", "guardada"),
    ("a", "maquete", "guardada"),
]

LOCAIS = [
    ("na", "mesa"),
    ("na", "gaveta"),
    ("na", "estante"),
    ("na", "cozinha"),
    ("na", "sala"),
    ("na", "oficina"),
    ("no", "armario"),
    ("no", "escritorio"),
    ("no", "deposito"),
    ("no", "corredor"),
]

PROBLEMAS = [
    ("a chuva forte", "a chuva atrasou a entrega dos materiais"),
    ("a falta de energia", "a falta de energia interrompeu parte do trabalho"),
    (
        "uma porta bloqueada",
        "uma porta bloqueada impediu o acesso ao deposito",
    ),
    (
        "um atraso no transporte",
        "o transporte chegou depois do horario previsto",
    ),
    (
        "uma lista incompleta",
        "a lista incompleta confundiu a separacao dos itens",
    ),
    (
        "uma janela quebrada",
        "a janela quebrada exigiu uma mudanca no plano",
    ),
]

ACOES = [
    "separar os materiais por categoria",
    "conferir cada item com cuidado",
    "limpar o espaco antes da montagem",
    "registrar as mudancas em uma lista",
    "mover as caixas para uma area segura",
    "revisar a ordem das tarefas",
]

TEMPOS = [
    "pela manha",
    "no inicio da tarde",
    "depois do almoco",
    "antes do anoitecer",
]


def gerar_registro_curto(
    semente: int,
    *,
    corrigir_gramatica: bool = True,
) -> dict[str, object]:
    """Gera um relato curto controlado para testes de composição."""

    aleatorio = random.Random(semente)
    (pessoa, pronome), (ajudante, pronome_ajudante) = aleatorio.sample(
        PESSOAS,
        2,
    )
    tarefa = aleatorio.choice(TAREFAS)
    objeto_principal, objeto_apoio = aleatorio.sample(OBJETOS, 2)
    local_principal, local_apoio = aleatorio.sample(LOCAIS, 2)
    problema_curto, problema_frase = aleatorio.choice(PROBLEMAS)
    acao_principal, acao_apoio = aleatorio.sample(ACOES, 2)
    tempo = aleatorio.choice(TEMPOS)
    artigo, objeto, _ = objeto_principal
    artigo_apoio, nome_apoio, _ = objeto_apoio
    preposicao, local = local_principal
    preposicao_apoio, lugar_apoio = local_apoio
    molde = aleatorio.randrange(3)
    colaborador = (
        "disposta"
        if corrigir_gramatica and pronome_ajudante == "ela"
        else "disposto"
    )
    artigo_contraido = (
        ("da" if artigo == "a" else "do")
        if corrigir_gramatica
        else f"de {artigo}"
    )

    pedido = (
        f"Pedido: escreva um relato sobre {pessoa}, com ajuda de "
        f"{ajudante}, para {tarefa}. Inclua {artigo} {objeto} "
        f"{preposicao} {local} e o problema {problema_curto}. Texto:"
    )
    if molde == 0:
        historia = (
            f"{tempo}, {pessoa} chegou ao local para {tarefa}. "
            f"Antes de comecar, {pronome} colocou {artigo} {objeto} "
            f"{preposicao} {local} e conferiu o plano. "
            f"Pouco depois, {ajudante} apareceu trazendo {artigo_apoio} "
            f"{nome_apoio} e ofereceu ajuda. "
            f"O trabalho ficou mais lento porque {problema_frase}. "
            f"Para evitar novos atrasos, {pessoa} decidiu {acao_principal}, "
            f"enquanto {ajudante} ficou responsavel por {acao_apoio}. "
            f"Durante a reorganizacao, {pronome_ajudante} levou "
            f"{artigo_apoio} {nome_apoio} {preposicao_apoio} "
            f"{lugar_apoio}, mas manteve {artigo} {objeto} "
            f"{preposicao} {local}. "
            f"No fim, os dois revisaram o espaco, concluiram a tarefa e "
            f"confirmaram que {artigo} {objeto} continuava "
            f"{preposicao} {local}."
        )
    elif molde == 1:
        historia = (
            f"{tempo}, {pessoa} iniciou o trabalho de {tarefa}. "
            f"{ajudante.capitalize()} chegou em seguida com "
            f"{artigo_apoio} {nome_apoio}, {colaborador} a colaborar. "
            f"Como {problema_frase}, a dupla precisou mudar a ordem prevista. "
            f"{pessoa.capitalize()} guardou {artigo} {objeto} "
            f"{preposicao} {local} antes de {acao_principal}. "
            f"Enquanto isso, {ajudante} levou {artigo_apoio} {nome_apoio} "
            f"{preposicao_apoio} {lugar_apoio} e comecou a {acao_apoio}. "
            f"Depois que o problema foi controlado, {pronome} verificou "
            f"novamente {artigo} {objeto} e percebeu que o objeto "
            f"permanecia {preposicao} {local}. "
            f"A atividade terminou com o espaco organizado, os materiais "
            f"registrados e a tarefa concluida sem novas perdas."
        )
    else:
        historia = (
            f"{tempo}, {pessoa} e {ajudante} se reuniram para {tarefa}. "
            f"O primeiro passo foi {acao_principal}, e depois eles decidiram "
            f"{acao_apoio}. "
            f"{pessoa.capitalize()} deixou {artigo} {objeto} "
            f"{preposicao} {local}, enquanto {ajudante} colocou "
            f"{artigo_apoio} {nome_apoio} {preposicao_apoio} {lugar_apoio}. "
            f"No meio do processo, {problema_frase}, obrigando a equipe a "
            f"interromper o cronograma. "
            f"{ajudante.capitalize()} resolveu a parte mais urgente e avisou "
            f"{pessoa} quando o caminho ficou livre. "
            f"Em seguida, os dois retomaram o trabalho sem alterar a posicao "
            f"{artigo_contraido} {objeto}. "
            f"Ao encerrar a atividade, {pronome} confirmou que {artigo} "
            f"{objeto} estava {preposicao} {local} e registrou o resultado."
        )

    texto = f"{pedido} {historia}"
    chave = (
        pessoa,
        ajudante,
        tarefa,
        objeto,
        local,
        problema_curto,
        molde,
    )
    return {
        "pedido": pedido,
        "historia": historia,
        "texto": texto,
        "perguntas": [],
        "palavras_chave": [
            pessoa,
            ajudante,
            objeto,
            local,
            problema_curto,
        ],
        "chave": chave,
    }
