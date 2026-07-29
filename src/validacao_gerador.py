"""Validação textual e avaliação de geração do domínio procedural.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import re

import torch

from .corpus_gerador_esparso import (
    ACOES,
    LOCAIS as LOCAIS_LONGOS,
    OBJETOS,
    PESSOAS,
    PROBLEMAS,
    TAREFAS,
)
from .decodificador_gerador import (
    ConfiguracaoDecodificacao,
    gerar_controlado,
)
from .modelo_gerador_esparso import ModeloGeradorEsparso
from .tokenizador_palavras import TokenizadorPalavras


SEMENTE = 20260728
MINIMO_CARACTERES_GERACAO = 2_000
FRASES_GERACAO = 24

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
    verbos_relacao = {
        "colocou",
        "colocado",
        "colocada",
        "levou",
        "levado",
        "levada",
        "guardou",
        "guardado",
        "guardada",
        "deixou",
        "ficou",
        "estava",
        "permaneceu",
        "permanecia",
        "continuou",
        "continuava",
        "seguia",
    }
    delimitadores = {".", ";", ":", "!", "?", ",", "mas", "enquanto"}
    encontrou_relacao = False
    for indice, token in enumerate(tokens):
        if token != objeto:
            continue
        inicio = indice
        while inicio > 0 and tokens[inicio - 1] not in delimitadores:
            inicio -= 1
        fim = indice + 1
        while fim < len(tokens) and tokens[fim] not in delimitadores:
            fim += 1
        clausula = tokens[inicio:fim]
        depois_objeto = tokens[indice + 1 : fim]
        if not verbos_relacao.intersection(clausula):
            continue
        locais_relacionados = locais.intersection(depois_objeto)
        if locais_relacionados - {local_esperado}:
            return False
        encontrou_relacao |= local_esperado in locais_relacionados
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
    r"^\s*pedido:\s*escreva um relato sobre\s+"
    r"(?P<pessoa>[^,]+),\s*"
    r"com ajuda de\s+(?P<ajudante>[^,]+),\s*"
    r"para\s+(?P<tarefa>.+?)\.\s*"
    r"inclua\s+(?:o|a)\s+(?P<objeto>\w+)\s+"
    r"(?:na|no)\s+(?P<local>\w+)\s+"
    r"e o problema\s+(?P<problema>.+?)\.\s*texto:\s*$",
    flags=re.IGNORECASE,
)


def extrair_campos_pedido(prompt: str) -> list[str] | None:
    """Extrai os seis campos verificaveis do formato publico do gerador."""

    correspondencia = PADRAO_PEDIDO.fullmatch(prompt)
    if correspondencia is None:
        return None
    return [
        correspondencia.group("pessoa").strip().lower(),
        correspondencia.group("ajudante").strip().lower(),
        correspondencia.group("tarefa").strip().lower(),
        correspondencia.group("objeto").strip().lower(),
        correspondencia.group("local").strip().lower(),
        correspondencia.group("problema").strip().lower(),
    ]


def validar_prompt_publico(
    prompt: str,
    tokenizador: TokenizadorPalavras,
) -> list[str]:
    """Valida formato e vocabulário antes de iniciar geração custosa."""

    campos = extrair_campos_pedido(prompt)
    if campos is None:
        raise ValueError(
            "prompt fora do formato estruturado documentado"
        )
    # O erro de vocabulário deve listar os tokens em vez de mascará-los como
    # uma simples incompatibilidade de domínio.
    tokenizador.validar_texto_no_vocabulario(prompt)
    correspondencia = PADRAO_PEDIDO.fullmatch(prompt)
    if correspondencia is None:  # Coberto por extrair_campos_pedido.
        raise ValueError("prompt fora do formato estruturado documentado")
    pessoa, ajudante, tarefa, objeto, local, problema = campos
    pessoas_validas = {nome for nome, _ in PESSOAS}
    objetos_validos = {nome for _, nome, _ in OBJETOS}
    locais_validos = {nome for _, nome in LOCAIS_LONGOS}
    problemas_validos = {nome for nome, _ in PROBLEMAS}
    if (
        pessoa not in pessoas_validas
        or ajudante not in pessoas_validas
        or pessoa == ajudante
        or tarefa not in TAREFAS
        or objeto not in objetos_validos
        or local not in locais_validos
        or problema not in problemas_validos
    ):
        raise ValueError(
            "prompt fora do formato estruturado: campo em categoria "
            "invalida para o dominio treinado"
        )
    return campos


def concordancia_de_genero_consistente(texto: str) -> bool:
    """Valida pares morfológicos simples usados pelo domínio procedural."""

    palavras = re.findall(r"[a-záàâãéêíóôõúç]+", texto.lower())
    masculinas = {
        "levado",
        "guardado",
        "reservado",
        "protegido",
        "organizado",
    }
    femininas = {
        "levada",
        "guardada",
        "reservada",
        "protegida",
        "organizada",
    }
    verbos = {"foi", "estava", "ficou", "seguia"}
    for indice, token in enumerate(palavras):
        if token in {"o", "um"} and indice + 3 < len(palavras):
            if (
                palavras[indice + 2] in verbos
                and palavras[indice + 3] in femininas
            ):
                return False
        if token in {"a", "uma"} and indice + 3 < len(palavras):
            if (
                palavras[indice + 2] in verbos
                and palavras[indice + 3] in masculinas
            ):
                return False
        if token == "ele" and indice + 2 < len(palavras):
            if (
                palavras[indice + 1] in verbos
                and palavras[indice + 2] in femininas
            ):
                return False
        if token == "ela" and indice + 2 < len(palavras):
            if (
                palavras[indice + 1] in verbos
                and palavras[indice + 2] in masculinas
            ):
                return False
    return True


def _medir_continuacao(
    tokenizador: TokenizadorPalavras,
    continuacao: str,
    palavras_chave: list[str],
) -> dict[str, object]:
    if len(palavras_chave) != 6:
        raise ValueError(
            "a avaliacao exige pessoa, ajudante, tarefa, objeto, local "
            "e problema"
        )
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
        palavras_chave[3],
        palavras_chave[4],
        continuacao,
    )
    consistencia_acoes = acoes_consistentes(continuacao)
    concordancia_genero = concordancia_de_genero_consistente(continuacao)
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
        "concordancia_genero": concordancia_genero,
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

    tokenizador.validar_texto_no_vocabulario(prompt)
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
            and bool(medidas["concordancia_genero"])
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
            bool(item[1]["concordancia_genero"]),
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
        concordancia_genero = bool(medidas["concordancia_genero"])
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
            and concordancia_genero
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
                "concordancia_genero": concordancia_genero,
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
        "concordancia_genero": (
            sum(
                bool(item["concordancia_genero"])
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
        "exemplos": resultados,
        "falhas": [
            item for item in resultados if not bool(item["aprovado"])
        ],
    }
