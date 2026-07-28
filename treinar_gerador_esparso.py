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
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from src.decodificador_gerador import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from src.modelo_gerador_esparso import (
    ConfiguracaoGeradorEsparso,
    ModeloGeradorEsparso,
)
from src.tokenizador_palavras import TokenizadorPalavras
from src.corpus_gerador_esparso import (
    ACOES,
    LOCAIS as LOCAIS_LONGOS,
    OBJETOS,
    PESSOAS,
    PROBLEMAS,
    TAREFAS,
    TEMPOS,
    gerar_registro_curto,
)


RAIZ = Path(__file__).resolve().parent
RESULTADOS_PADRAO = RAIZ / "resultados" / "gerador_esparso_base_50k"
SEMENTE = 20260728
QUANTIDADES_PADRAO = (50_000, 1_000, 1_000)
INICIOS_PADRAO = (40_000_000, 50_000_000, 60_000_000)
PASSOS_POR_EPOCA_PADRAO = 500
MINIMO_CARACTERES_GERACAO = 2_000
FRASES_GERACAO = 24


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


@torch.inference_mode()
def avaliar_linguagem(
    modelo: ModeloGeradorEsparso,
    codificados: list[tuple[list[int], list[float]]],
    pad_id: int,
    tamanho_lote: int,
    dispositivo: torch.device,
) -> dict[str, float]:
    """Mede a perda sem os pesos extras usados apenas durante o treino."""

    modelo.eval()
    perda_total = 0.0
    tokens_total = 0
    corretos = 0
    for entradas, alvos, _ in criar_lotes(
        codificados,
        pad_id,
        tamanho_lote,
        dispositivo,
        embaralhar=False,
        semente=SEMENTE,
    ):
        logits, _ = modelo(entradas)
        mascara = alvos.ne(pad_id)
        perda_total += float(
            F.cross_entropy(
                logits[mascara],
                alvos[mascara],
                reduction="sum",
            )
        )
        tokens_total += int(mascara.sum())
        corretos += int(
            logits.argmax(dim=-1)[mascara].eq(alvos[mascara]).sum()
        )
    perda = perda_total / tokens_total
    return {
        "perda": perda,
        "ppl": math.exp(perda),
        "acuracia_token": corretos / tokens_total,
        "tokens": float(tokens_total),
    }


def extrair_continuacao(texto: str) -> str:
    """Remove o pedido devolvido pelo decodificador."""

    partes = texto.lower().split("texto:", 1)
    return (partes[-1] if partes else texto).strip()


def taxa_trigramas_repetidos(tokens: list[str]) -> float:
    """Calcula repeticao sem penalizar ocorrencias isoladas normais."""

    if len(tokens) < 3:
        return 0.0
    trigramas = [
        tuple(tokens[indice : indice + 3])
        for indice in range(len(tokens) - 2)
    ]
    return 1.0 - len(set(trigramas)) / len(trigramas)


def frequencia_maxima_trigrama(tokens: list[str]) -> int:
    """Distingue repetições naturais de ciclos autorregressivos."""

    if len(tokens) < 3:
        return 0
    contagens: dict[tuple[str, str, str], int] = {}
    for indice in range(len(tokens) - 2):
        trigrama = tuple(tokens[indice : indice + 3])
        contagens[trigrama] = contagens.get(trigrama, 0) + 1
    return max(contagens.values(), default=0)


def termo_recuperado(termo: str, continuacao: str) -> bool:
    """Compara conteudo ignorando apenas variacoes normais de artigo."""

    artigos = {"o", "a", "os", "as", "um", "uma", "uns", "umas"}
    tokens_termo = [
        token
        for token in TokenizadorPalavras.tokenizar(termo.lower())
        if token not in artigos
    ]
    tokens_texto = set(
        TokenizadorPalavras.tokenizar(continuacao.lower())
    )
    return bool(tokens_termo) and all(
        token in tokens_texto for token in tokens_termo
    )


def problema_recuperado(problema: str, continuacao: str) -> bool:
    """Aceita o nome curto ou sua realizacao causal definida no corpus."""

    if termo_recuperado(problema, continuacao):
        return True
    frase = next(
        (
            realizacao
            for nome, realizacao in PROBLEMAS
            if nome == problema
        ),
        "",
    )
    return bool(frase) and termo_recuperado(frase, continuacao)


def local_do_objeto_consistente(
    objeto: str,
    local_esperado: str,
    continuacao: str,
) -> bool:
    """Reprova quando o objeto pedido e associado a outro local."""

    tokens = TokenizadorPalavras.tokenizar(continuacao.lower())
    locais = {local for _, local in LOCAIS_LONGOS}
    encontrou_relacao = False
    for indice, token in enumerate(tokens):
        if token != objeto:
            continue
        proximos = tokens[indice + 1 : indice + 5]
        locais_proximos = locais.intersection(proximos)
        if locais_proximos - {local_esperado}:
            return False
        encontrou_relacao |= local_esperado in locais_proximos
    return encontrou_relacao


def acoes_consistentes(continuacao: str) -> bool:
    """Confere acoes em pontos de molde para detectar frases combinadas."""

    tokens = TokenizadorPalavras.tokenizar(continuacao.lower())
    acoes_tokens = [
        TokenizadorPalavras.tokenizar(acao)
        for acao in ACOES
    ]
    verbos = {acao[0] for acao in acoes_tokens}
    marcadores = {
        "decidiu",
        "decidiram",
        "responsavel",
        "comecou",
        "passo",
    }
    for indice, token in enumerate(tokens):
        if token not in verbos:
            continue
        contexto = tokens[max(0, indice - 4) : indice]
        em_ponto_de_acao = bool(marcadores.intersection(contexto)) or (
            contexto[-2:] == ["antes", "de"]
        )
        if not em_ponto_de_acao:
            continue
        if not any(
            tokens[indice : indice + len(acao)] == acao
            for acao in acoes_tokens
            if acao[0] == token
        ):
            return False
    return True


PADRAO_PEDIDO = re.compile(
    r"sobre\s+(?P<pessoa>[^,]+),\s*"
    r"com ajuda de\s+(?P<ajudante>[^,]+),\s*"
    r"para\s+.+?\.\s*"
    r"inclua\s+(?:o|a)\s+(?P<objeto>\w+)\s+"
    r"(?:na|no)\s+(?P<local>\w+)\s+"
    r"e o problema\s+(?P<problema>.+?)\.\s*texto:",
    flags=re.IGNORECASE,
)


def extrair_campos_pedido(prompt: str) -> list[str] | None:
    """Extrai os cinco campos verificaveis do formato publico do gerador."""

    correspondencia = PADRAO_PEDIDO.search(prompt)
    if correspondencia is None:
        return None
    return [
        correspondencia.group("pessoa").strip().lower(),
        correspondencia.group("ajudante").strip().lower(),
        correspondencia.group("objeto").strip().lower(),
        correspondencia.group("local").strip().lower(),
        correspondencia.group("problema").strip().lower(),
    ]


def _medir_continuacao(
    tokenizador: TokenizadorPalavras,
    continuacao: str,
    palavras_chave: list[str],
) -> dict[str, object]:
    tokens = tokenizador.tokenizar(continuacao)
    frases = sum(token in {".", "!", "?"} for token in tokens)
    recuperados = [
        termo_recuperado(valor, continuacao)
        for valor in palavras_chave[:-1]
    ]
    recuperados.append(
        problema_recuperado(palavras_chave[-1], continuacao)
    )
    cobertura = sum(recuperados) / len(recuperados)
    repeticao = taxa_trigramas_repetidos(tokens)
    frequencia_trigrama = frequencia_maxima_trigrama(tokens)
    consistencia_local = local_do_objeto_consistente(
        palavras_chave[2],
        palavras_chave[3],
        continuacao,
    )
    consistencia_acoes = acoes_consistentes(continuacao)
    vazou_bloco_pergunta = any(
        tokens[indice] in {"pergunta", "resposta"}
        and indice + 1 < len(tokens)
        and tokens[indice + 1] == ":"
        for indice in range(len(tokens))
    )
    caracteres = len(continuacao)
    completo = (
        caracteres >= MINIMO_CARACTERES_GERACAO
        and frases >= FRASES_GERACAO
        and not continuacao.endswith(":")
        and not vazou_bloco_pergunta
    )
    return {
        "tokens": len(tokens),
        "caracteres": caracteres,
        "frases": frases,
        "cobertura_palavras_chave": cobertura,
        "trigramas_repetidos": repeticao,
        "frequencia_maxima_trigrama": frequencia_trigrama,
        "local_do_objeto_consistente": consistencia_local,
        "acoes_consistentes": consistencia_acoes,
        "vazou_bloco_pergunta": vazou_bloco_pergunta,
        "completo": completo,
    }


@torch.inference_mode()
def gerar_relato_validado(
    modelo: ModeloGeradorEsparso,
    tokenizador: TokenizadorPalavras,
    prompt: str,
    palavras_chave: list[str],
    dispositivo: torch.device,
    *,
    semente: int = SEMENTE,
    maximo_retentativas: int = 3,
) -> tuple[str, dict[str, object]]:
    """Usa o greedy e recorre a baixa amostragem somente se houver conflito."""

    configuracao = ConfiguracaoDecodificacao(
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

    candidatos: list[tuple[str, dict[str, object], int]] = []
    for tentativa in range(maximo_retentativas + 1):
        texto = gerar_controlado(
            modelo,
            tokenizador,
            prompt,
            dispositivo,
            configuracao=configuracao,
            amostrar=tentativa > 0,
            semente=semente + tentativa * 10_007,
        )
        continuacao = extrair_continuacao(texto)
        medidas = _medir_continuacao(
            tokenizador,
            continuacao,
            palavras_chave,
        )
        candidatos.append((texto, medidas, tentativa))
        valido = (
            bool(medidas["completo"])
            and float(medidas["cobertura_palavras_chave"]) == 1.0
            and bool(medidas["local_do_objeto_consistente"])
            and bool(medidas["acoes_consistentes"])
        )
        # O greedy permanece preferido quando ja satisfaz os contratos.
        if valido:
            return texto, {**medidas, "retentativas": tentativa}

    # Se nenhuma tentativa cumprir tudo, devolve a menos arriscada e registra
    # a falha para que a avaliacao nao a esconda.
    melhor_texto, melhores_medidas, tentativa = max(
        candidatos,
        key=lambda item: (
            bool(item[1]["completo"]),
            float(item[1]["cobertura_palavras_chave"]),
            bool(item[1]["local_do_objeto_consistente"]),
            bool(item[1]["acoes_consistentes"]),
            -float(item[1]["trigramas_repetidos"]),
        ),
    )
    return melhor_texto, {
        **melhores_medidas,
        "retentativas": tentativa,
    }


@torch.inference_mode()
def avaliar_geracao_livre(
    modelo: ModeloGeradorEsparso,
    tokenizador: TokenizadorPalavras,
    registros: list[dict[str, object]],
    dispositivo: torch.device,
    quantidade: int = 24,
) -> dict[str, object]:
    """Mede completude, repeticao e recuperacao dos campos do pedido."""

    resultados: list[dict[str, object]] = []
    for indice, registro in enumerate(registros[:quantidade]):
        palavras_chave = [
            str(valor).lower()
            for valor in registro["palavras_chave"]
        ]
        texto, medidas = gerar_relato_validado(
            modelo,
            tokenizador,
            str(registro["pedido"]),
            palavras_chave,
            dispositivo,
            semente=SEMENTE + indice,
        )
        continuacao = extrair_continuacao(texto)
        cobertura = float(medidas["cobertura_palavras_chave"])
        repeticao = float(medidas["trigramas_repetidos"])
        consistencia_local = bool(
            medidas["local_do_objeto_consistente"]
        )
        consistencia_acoes = bool(medidas["acoes_consistentes"])
        vazou_bloco_pergunta = bool(
            medidas["vazou_bloco_pergunta"]
        )
        completo = bool(medidas["completo"])
        aprovado = (
            completo
            and cobertura == 1.0
            and repeticao <= 0.05
            and consistencia_local
            and consistencia_acoes
        )
        resultados.append(
            {
                "pedido": registro["pedido"],
                "texto": continuacao,
                "tokens": int(medidas["tokens"]),
                "caracteres": int(medidas["caracteres"]),
                "frases": int(medidas["frases"]),
                "cobertura_palavras_chave": cobertura,
                "trigramas_repetidos": repeticao,
                "frequencia_maxima_trigrama": int(
                    medidas["frequencia_maxima_trigrama"]
                ),
                "local_do_objeto_consistente": consistencia_local,
                "acoes_consistentes": consistencia_acoes,
                "vazou_bloco_pergunta": vazou_bloco_pergunta,
                "completo": completo,
                "aprovado": aprovado,
                "retentativas": int(medidas["retentativas"]),
            }
        )

    total = len(resultados)
    return {
        "amostras_avaliadas": total,
        "taxa_completude": (
            sum(bool(item["completo"]) for item in resultados) / total
        ),
        "taxa_aprovacao": (
            sum(bool(item["aprovado"]) for item in resultados) / total
        ),
        "taxa_vazamento_pergunta": (
            sum(
                bool(item["vazou_bloco_pergunta"])
                for item in resultados
            )
            / total
        ),
        "taxa_uso_retentativa": (
            sum(int(item["retentativas"]) > 0 for item in resultados)
            / total
        ),
        "consistencia_objeto_local": (
            sum(
                bool(item["local_do_objeto_consistente"])
                for item in resultados
            )
            / total
        ),
        "consistencia_acoes": (
            sum(
                bool(item["acoes_consistentes"])
                for item in resultados
            )
            / total
        ),
        "tokens_medios": (
            sum(int(item["tokens"]) for item in resultados) / total
        ),
        "caracteres_minimos": min(
            int(item["caracteres"]) for item in resultados
        ),
        "caracteres_medios": (
            sum(int(item["caracteres"]) for item in resultados) / total
        ),
        "frases_medias": (
            sum(int(item["frases"]) for item in resultados) / total
        ),
        "recuperacao_campos_pedido": (
            sum(
                float(item["cobertura_palavras_chave"])
                for item in resultados
            )
            / total
        ),
        "repeticao_trigramas_media": (
            sum(
                float(item["trigramas_repetidos"])
                for item in resultados
            )
            / total
        ),
        "frequencia_maxima_trigrama": max(
            int(item["frequencia_maxima_trigrama"])
            for item in resultados
        ),
        "exemplos": resultados[:6],
        "falhas": [
            item for item in resultados if not bool(item["aprovado"])
        ][:8],
    }


@torch.inference_mode()
def benchmark(
    modelo: ModeloGeradorEsparso,
    vocabulario: int,
    dispositivo: torch.device,
) -> dict[str, float]:
    """Mede o forward de forma sincronizada na GPU."""

    modelo.eval()
    entradas = torch.randint(
        4,
        vocabulario,
        (16, modelo.configuracao.maximo_contexto),
        device=dispositivo,
    )
    for _ in range(3):
        modelo(entradas)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    repeticoes = 12
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        modelo(entradas)
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    return {
        "tokens_por_segundo": entradas.numel() * repeticoes / duracao,
        "latencia_ms": duracao * 1000.0 / repeticoes,
        "vram_pico_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
    }


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
            "versao": "1.0.0",
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
    relatorio = {
        "experimento": "gerador_esparso_base_50k",
        "versao": "1.0.0",
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
        "desempenho": desempenho,
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
                "desempenho": desempenho,
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
