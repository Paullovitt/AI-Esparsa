"""Audita pipeline, geracao livre e roteamento aprendido da V6.

O checkpoint-base permanece congelado. O experimento treina somente um
codebook esparso sem mapas manuais de entidade, objeto ou alias.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from executar_v6 import CHECKPOINT_PADRAO, carregar_v6
from src.atencao_causal_texto import (
    CodigoEsparsoCausal,
    ConfiguracaoTextoCausal,
    DadosTextoCausal,
    VocabularioTexto,
    criar_vocabulario,
    gerar_corpus_texto_causal,
    selecionar_candidatos_causais,
)
from src.modelo_v6 import ModeloV6
from treinar_v6 import avaliar, expandir_tokens


RAIZ = Path(__file__).resolve().parent
RESULTADOS = RAIZ / "resultados"


@dataclass(frozen=True)
class ConfiguracaoRoteadorAprendido:
    dimensao_features: int = 128
    features_ativas: int = 12
    janela_local: int = 8
    temperatura: float = 0.08
    tamanho_lote: int = 128
    taxa_aprendizado: float = 0.04


CONFIGURACAO_ROTEADOR = ConfiguracaoRoteadorAprendido()


class RoteadorEsparsoAprendido(nn.Module):
    """Codebook treinavel que produz descritores Top-K a partir dos tokens."""

    def __init__(
        self,
        tamanho_vocabulario: int,
        configuracao: ConfiguracaoRoteadorAprendido = CONFIGURACAO_ROTEADOR,
        semente: int = 20260729,
    ) -> None:
        super().__init__()
        self.configuracao = configuracao
        gerador = torch.Generator().manual_seed(semente)
        iniciais = torch.randn(
            tamanho_vocabulario,
            configuracao.dimensao_features,
            generator=gerador,
        ) * 0.08
        self.codigo_token = nn.Embedding(
            tamanho_vocabulario,
            configuracao.dimensao_features,
            sparse=True,
        )
        with torch.no_grad():
            self.codigo_token.weight.copy_(iniciais)

    def representacoes(self, tokens: Tensor) -> Tensor:
        """Resume a janela local sem consultar classes semanticas manuais."""

        codigos = self.codigo_token(tokens)
        soma = torch.zeros_like(codigos)
        contagem = torch.zeros(
            *tokens.shape,
            1,
            dtype=codigos.dtype,
            device=tokens.device,
        )
        for deslocamento in range(self.configuracao.janela_local):
            if deslocamento == 0:
                soma.add_(codigos)
                contagem.add_(1)
            elif deslocamento < tokens.shape[1]:
                soma[:, deslocamento:].add_(codigos[:, :-deslocamento])
                contagem[:, deslocamento:].add_(1)
        medias = soma / contagem.clamp_min(1)
        return F.normalize(medias, dim=-1)

    @torch.no_grad()
    def descritores(self, tokens: Tensor) -> Tensor:
        representacoes = self.representacoes(tokens)
        return (
            representacoes.topk(
                self.configuracao.features_ativas, dim=-1
            ).indices
            + 1
        )

    def auditoria(self) -> dict[str, int | bool | str]:
        return {
            "tipo": "codebook_topk_sem_mapas_manuais",
            "parametros": self.codigo_token.weight.numel(),
            "features_totais": self.configuracao.dimensao_features,
            "features_ativas": self.configuracao.features_ativas,
            "usa_rede_densa": any(
                isinstance(modulo, nn.Linear) for modulo in self.modules()
            ),
        }


def posicoes_consulta_e_origem(
    dados: DadosTextoCausal,
) -> tuple[Tensor, Tensor]:
    lote, comprimento = dados.mascara_resposta_local.shape
    consultas = (
        dados.mascara_resposta_local.nonzero(as_tuple=False)[:, 1]
        .view(lote, -1)
        .contiguous()
    )
    origens = dados.posicao_fato_origem.gather(1, consultas)
    if consultas.shape[1] != 3:
        raise RuntimeError("o corpus deveria conter tres consultas")
    if torch.any(origens < 0) or torch.any(origens >= comprimento):
        raise RuntimeError("origens causais invalidas")
    return consultas, origens


def perda_contrastiva(
    roteador: RoteadorEsparsoAprendido,
    dados: DadosTextoCausal,
) -> tuple[Tensor, float]:
    representacoes = roteador.representacoes(dados.entradas)
    consultas, origens = posicoes_consulta_e_origem(dados)
    lote = len(dados.tokens)
    indice_lote = torch.arange(
        lote, device=dados.tokens.device
    ).view(-1, 1)
    representacoes_consulta = representacoes[indice_lote, consultas]
    logits = torch.einsum(
        "bqd,btd->bqt",
        representacoes_consulta,
        representacoes,
    ) / roteador.configuracao.temperatura
    posicoes = torch.arange(
        representacoes.shape[1], device=logits.device
    ).view(1, 1, -1)
    causal = posicoes < consultas.unsqueeze(-1)
    logits = logits.masked_fill(~causal, -torch.inf)
    alvos = origens
    perda = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), alvos.reshape(-1)
    )
    acuracia = float(logits.argmax(dim=-1).eq(alvos).float().mean())
    return perda, acuracia


@torch.no_grad()
def preparar_com_roteador(
    dados: DadosTextoCausal,
    roteador: RoteadorEsparsoAprendido,
    top_k: int,
) -> tuple[DadosTextoCausal, Tensor]:
    descritores = roteador.descritores(dados.entradas)
    candidatos, validos = selecionar_candidatos_causais(
        descritores, top_k
    )
    return dados.com_candidatos(candidatos, validos), descritores


@torch.no_grad()
def recall_candidatos(
    dados: DadosTextoCausal,
) -> dict[str, float]:
    if dados.candidatos is None or dados.candidatos_validos is None:
        raise ValueError("dados ainda nao possuem candidatos")
    mascara = dados.mascara_resposta_local
    origens = dados.posicao_fato_origem[mascara]
    candidatos = dados.candidatos[mascara]
    validos = dados.candidatos_validos[mascara]
    presente = ((candidatos == origens.unsqueeze(-1)) & validos).any(
        dim=-1
    )
    primeiro = candidatos[:, 0].eq(origens) & validos[:, 0]
    return {
        "consultas": float(len(origens)),
        "recall_top24": float(presente.float().mean()),
        "acuracia_top1": float(primeiro.float().mean()),
    }


def treinar_roteador_epoca(
    roteador: RoteadorEsparsoAprendido,
    dados: DadosTextoCausal,
    otimizador: torch.optim.Optimizer,
) -> dict[str, float]:
    roteador.train()
    perdas = []
    acuracias = []
    permutacao = torch.randperm(
        len(dados.tokens), device=dados.tokens.device
    )
    for indices in permutacao.split(
        roteador.configuracao.tamanho_lote
    ):
        lote = dados.selecionar(indices)
        perda, acuracia = perda_contrastiva(roteador, lote)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        otimizador.step()
        perdas.append(float(perda))
        acuracias.append(acuracia)
    return {
        "perda": statistics.fmean(perdas),
        "acuracia_contrastiva": statistics.fmean(acuracias),
    }


def perda_leitor_qk(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritores: Tensor,
) -> tuple[Tensor, float]:
    """Ensina Q/K a escolher a origem sem alterar o restante da V6."""

    if dados.candidatos is None or dados.candidatos_validos is None:
        raise ValueError("dados ainda nao possuem candidatos")
    mascara = dados.mascara_resposta_local
    lote = len(dados.tokens)
    indice_lote = torch.arange(
        lote, device=dados.tokens.device
    ).view(lote, 1, 1)
    descritores_candidatos = descritores[
        indice_lote, dados.candidatos
    ]
    consulta = descritores[mascara]
    candidatos = descritores_candidatos[mascara]
    validos = dados.candidatos_validos[mascara]
    origens = dados.posicao_fato_origem[mascara]
    indices_candidatos = dados.candidatos[mascara]
    alvos_booleanos = (
        indices_candidatos.eq(origens.unsqueeze(-1)) & validos
    )
    presentes = alvos_booleanos.any(dim=-1)
    if not torch.any(presentes):
        raise RuntimeError("nenhuma origem correta entre os candidatos")

    consulta = consulta[presentes]
    candidatos = candidatos[presentes]
    validos = validos[presentes]
    alvos = alvos_booleanos[presentes].to(torch.long).argmax(dim=-1)
    q = modelo.atencao.q_feature(consulta).squeeze(-1)
    k = modelo.atencao.k_feature(candidatos).squeeze(-1)
    igualdade = consulta.unsqueeze(1).unsqueeze(-1).eq(
        candidatos.unsqueeze(-2)
    )
    nao_zero = consulta.unsqueeze(1).unsqueeze(-1).ne(0)
    pontuacao = (
        (igualdade & nao_zero).to(q.dtype)
        * q.unsqueeze(1).unsqueeze(-1)
        * k.unsqueeze(-2)
    ).sum(dim=(-1, -2))
    pontuacao = pontuacao / math.sqrt(float(consulta.shape[-1]))
    pontuacao = pontuacao.masked_fill(~validos, -torch.inf)
    perda = F.cross_entropy(pontuacao, alvos)
    acuracia = float(
        pontuacao.argmax(dim=-1).eq(alvos).float().mean()
    )
    return perda, acuracia


def treinar_leitor_qk_epoca(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritores: Tensor,
    otimizador: torch.optim.Optimizer,
    tamanho_lote: int = 128,
) -> dict[str, float]:
    modelo.train()
    perdas = []
    acuracias = []
    permutacao = torch.randperm(
        len(dados.tokens), device=dados.tokens.device
    )
    for indices in permutacao.split(tamanho_lote):
        lote = dados.selecionar(indices)
        perda, acuracia = perda_leitor_qk(
            modelo, lote, descritores[indices]
        )
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        otimizador.step()
        perdas.append(float(perda))
        acuracias.append(acuracia)
    return {
        "perda_qk": statistics.fmean(perdas),
        "acuracia_qk": statistics.fmean(acuracias),
    }


def _medir_cuda(
    funcao: Callable[[], object],
    tokens: int,
    repeticoes: int,
    aquecimentos: int,
) -> dict[str, float]:
    for _ in range(aquecimentos):
        funcao()
    torch.cuda.synchronize()
    memoria_base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        funcao()
    torch.cuda.synchronize()
    duracao = time.perf_counter() - inicio
    return {
        "latencia_ms": duracao * 1_000 / repeticoes,
        "tokens_por_segundo": tokens * repeticoes / duracao,
        "vram_temporaria_mib": max(
            0,
            torch.cuda.max_memory_allocated() - memoria_base,
        )
        / (1024**2),
    }


@torch.inference_mode()
def benchmark_pipeline(
    modelo: ModeloV6,
    tokens_base: Tensor,
    descritor: Callable[[Tensor], Tensor],
    configuracao_texto: ConfiguracaoTextoCausal,
    comprimento: int,
    lote: int,
) -> dict[str, dict[str, float]]:
    tokens = expandir_tokens(tokens_base, comprimento, lote)
    descritores = descritor(tokens)
    candidatos, validos = selecionar_candidatos_causais(
        descritores, configuracao_texto.top_k
    )
    repeticoes = 20 if comprimento == 73 else 4
    aquecimentos = 5 if comprimento == 73 else 2

    def somente_descritor() -> Tensor:
        return descritor(tokens)

    def descritor_e_indice() -> tuple[Tensor, Tensor]:
        atuais = descritor(tokens)
        return selecionar_candidatos_causais(
            atuais, configuracao_texto.top_k
        )

    def pipeline_completo() -> tuple[Tensor, Tensor]:
        atuais = descritor(tokens)
        candidatos_atuais, validos_atuais = (
            selecionar_candidatos_causais(
                atuais, configuracao_texto.top_k
            )
        )
        return modelo(
            tokens,
            atuais,
            candidatos_atuais,
            validos_atuais,
        )

    return {
        "forward_isolado": _medir_cuda(
            lambda: modelo(
                tokens, descritores, candidatos, validos
            ),
            tokens.numel(),
            repeticoes,
            aquecimentos,
        ),
        "descritor": _medir_cuda(
            somente_descritor,
            tokens.numel(),
            repeticoes,
            aquecimentos,
        ),
        "descritor_indice": _medir_cuda(
            descritor_e_indice,
            tokens.numel(),
            repeticoes,
            aquecimentos,
        ),
        "pipeline_completo": _medir_cuda(
            pipeline_completo,
            tokens.numel(),
            repeticoes,
            aquecimentos,
        ),
    }


@torch.inference_mode()
def gerar_livre(
    modelo: ModeloV6,
    prefixo: Tensor,
    descritor: Callable[[Tensor], Tensor],
    configuracao_texto: ConfiguracaoTextoCausal,
    eos_id: int,
    maximo_novos_tokens: int,
) -> Tensor:
    gerados: list[int] = []
    tokens = prefixo
    for _ in range(maximo_novos_tokens):
        descritores = descritor(tokens)
        candidatos, validos = selecionar_candidatos_causais(
            descritores, configuracao_texto.top_k
        )
        logits, _ = modelo(
            tokens, descritores, candidatos, validos
        )
        proximo = int(logits[:, -1].argmax(dim=-1).item())
        gerados.append(proximo)
        tokens = torch.cat(
            (
                tokens,
                torch.tensor(
                    [[proximo]],
                    device=tokens.device,
                    dtype=tokens.dtype,
                ),
            ),
            dim=1,
        )
        if proximo == eos_id:
            break
    return torch.tensor(
        gerados, dtype=prefixo.dtype, device=prefixo.device
    )


@torch.inference_mode()
def avaliar_geracao_livre(
    modelo: ModeloV6,
    dados: DadosTextoCausal,
    descritor: Callable[[Tensor], Tensor],
    configuracao_texto: ConfiguracaoTextoCausal,
    vocabulario: VocabularioTexto,
    quantidade: int = 72,
) -> dict[str, float | int]:
    eos_id = vocabulario.id("<eos>")
    exatas = 0
    eos_corretos = 0
    tokens_corretos = 0
    tokens_esperados = 0
    locais_corretos = 0
    locais_total = 0
    for linha in range(quantidade):
        posicoes = (
            dados.mascara_resposta_local[linha]
            .nonzero(as_tuple=False)
            .flatten()
        )
        inicio = int(posicoes[0])
        prefixo = dados.tokens[
            linha : linha + 1, : inicio + 1
        ].contiguous()
        esperado = dados.tokens[linha, inicio + 1 :]
        gerado = gerar_livre(
            modelo,
            prefixo,
            descritor,
            configuracao_texto,
            eos_id,
            len(esperado) + 8,
        )
        exatas += int(torch.equal(gerado, esperado))
        eos_corretos += int(
            len(gerado) > 0 and int(gerado[-1]) == eos_id
        )
        limite = min(len(gerado), len(esperado))
        tokens_corretos += int(
            gerado[:limite].eq(esperado[:limite]).sum()
        )
        tokens_esperados += len(esperado)
        for posicao in posicoes:
            deslocamento = int(posicao) - inicio
            if deslocamento < len(gerado):
                locais_corretos += int(
                    gerado[deslocamento]
                    == dados.alvos[linha, int(posicao)]
                )
            locais_total += 1
    return {
        "prompts": quantidade,
        "sequencias_exatas": exatas / quantidade,
        "taxa_eos": eos_corretos / quantidade,
        "acuracia_tokens_alinhados": tokens_corretos
        / tokens_esperados,
        "acuracia_locais_alinhados": locais_corretos
        / locais_total,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA e obrigatoria para o experimento")
    dispositivo = torch.device("cuda")
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    vocabulario = criar_vocabulario()
    modelo, codigo_manual, checkpoint = carregar_v6(
        CHECKPOINT_PADRAO, dispositivo
    )
    for parametro in modelo.parameters():
        parametro.requires_grad_(False)

    configuracao_texto = modelo.configuracao_texto
    treino = gerar_corpus_texto_causal(
        4000,
        101,
        vocabulario,
        configuracao_texto,
        usar_combinacoes_ineditas=False,
    ).para(dispositivo)
    validacao = gerar_corpus_texto_causal(
        1000,
        202,
        vocabulario,
        configuracao_texto,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)
    teste = gerar_corpus_texto_causal(
        2000,
        303,
        vocabulario,
        configuracao_texto,
        usar_combinacoes_ineditas=True,
    ).para(dispositivo)

    roteador = RoteadorEsparsoAprendido(
        vocabulario.tamanho
    ).to(dispositivo)
    otimizador = torch.optim.SparseAdam(
        roteador.parameters(),
        lr=CONFIGURACAO_ROTEADOR.taxa_aprendizado,
    )
    diretorio = RESULTADOS / (
        "teste_isolado_v6_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    diretorio.mkdir(parents=True, exist_ok=False)
    historico = []
    for epoca in range(1, 6):
        treino_epoca = treinar_roteador_epoca(
            roteador, treino, otimizador
        )
        validacao_roteada, _ = preparar_com_roteador(
            validacao, roteador, configuracao_texto.top_k
        )
        recall = recall_candidatos(validacao_roteada)
        registro = {
            "epoca": epoca,
            **treino_epoca,
            **recall,
        }
        historico.append(registro)
        torch.save(
            {
                "estado_roteador": roteador.state_dict(),
                "epoca": epoca,
                "configuracao": asdict(CONFIGURACAO_ROTEADOR),
                "metricas_validacao": registro,
            },
            diretorio / f"roteador_epoca{epoca}.pt",
        )
        print(
            f"[roteador] epoca {epoca}/5 "
            f"loss={registro['perda']:.4f} "
            f"contraste={registro['acuracia_contrastiva']:.2%} "
            f"recall={registro['recall_top24']:.2%} "
            f"top1={registro['acuracia_top1']:.2%}",
            flush=True,
        )

    roteador.eval()
    treino_aprendido, descritores_treino_aprendido = (
        preparar_com_roteador(
            treino, roteador, configuracao_texto.top_k
        )
    )
    validacao_aprendida, descritores_validacao_aprendida = (
        preparar_com_roteador(
            validacao, roteador, configuracao_texto.top_k
        )
    )
    modelo_adaptado, _, _ = carregar_v6(
        CHECKPOINT_PADRAO, dispositivo
    )
    for parametro in modelo_adaptado.parameters():
        parametro.requires_grad_(False)
    for parametro in modelo_adaptado.parametros_qk_esparsos():
        parametro.requires_grad_(True)
    otimizador_qk = torch.optim.SparseAdam(
        modelo_adaptado.parametros_qk_esparsos(), lr=0.04
    )
    historico_qk = []
    for epoca in range(1, 6):
        treino_qk = treinar_leitor_qk_epoca(
            modelo_adaptado,
            treino_aprendido,
            descritores_treino_aprendido,
            otimizador_qk,
        )
        perda_qk_validacao, acuracia_qk_validacao = perda_leitor_qk(
            modelo_adaptado,
            validacao_aprendida,
            descritores_validacao_aprendida,
        )
        qualidade_validacao = avaliar(
            modelo_adaptado,
            validacao_aprendida,
            descritores_validacao_aprendida,
        )
        registro_qk = {
            "epoca": epoca,
            **treino_qk,
            "perda_qk_validacao": float(perda_qk_validacao),
            "acuracia_qk_validacao": acuracia_qk_validacao,
            "acuracia_local_validacao": qualidade_validacao[
                "acuracia_local_resposta"
            ],
            "recuperacao_fato_validacao": qualidade_validacao[
                "recuperacao_fato"
            ],
        }
        historico_qk.append(registro_qk)
        torch.save(
            {
                "estado_qk": {
                    "q_feature.weight": (
                        modelo_adaptado.atencao.q_feature.weight.detach()
                    ),
                    "k_feature.weight": (
                        modelo_adaptado.atencao.k_feature.weight.detach()
                    ),
                },
                "epoca": epoca,
                "metricas_validacao": registro_qk,
                "checkpoint_base": str(CHECKPOINT_PADRAO),
            },
            diretorio / f"leitor_qk_epoca{epoca}.pt",
        )
        print(
            f"[leitor Q/K] epoca {epoca}/5 "
            f"acc={registro_qk['acuracia_qk_validacao']:.2%} "
            f"local={registro_qk['acuracia_local_validacao']:.2%} "
            f"recall={registro_qk['recuperacao_fato_validacao']:.2%}",
            flush=True,
        )

    manual_descritores = codigo_manual.descritores(teste.entradas)
    manual_candidatos, manual_validos = selecionar_candidatos_causais(
        manual_descritores, configuracao_texto.top_k
    )
    teste_manual = teste.com_candidatos(
        manual_candidatos, manual_validos
    )
    teste_aprendido, descritores_aprendidos = preparar_com_roteador(
        teste, roteador, configuracao_texto.top_k
    )
    qualidade_aprendida_sem_leitor = avaliar(
        modelo, teste_aprendido, descritores_aprendidos
    )

    benchmark = {}
    for nome, modelo_benchmark, funcao_descritor in (
        ("manual", modelo, codigo_manual.descritores),
        ("aprendido", modelo_adaptado, roteador.descritores),
    ):
        benchmark[nome] = {}
        for comprimento, lote in ((73, 64), (512, 16)):
            benchmark[nome][str(comprimento)] = benchmark_pipeline(
                modelo_benchmark,
                teste.entradas,
                funcao_descritor,
                configuracao_texto,
                comprimento,
                lote,
            )

    geracao_manual = avaliar_geracao_livre(
        modelo,
        teste,
        codigo_manual.descritores,
        configuracao_texto,
        vocabulario,
    )
    geracao_aprendida = avaliar_geracao_livre(
        modelo_adaptado,
        teste,
        roteador.descritores,
        configuracao_texto,
        vocabulario,
    )
    relatorio = {
        "experimento": "auditoria_isolada_v6",
        "autor": "Paulo Augusto",
        "ano": 2026,
        "checkpoint_v6": str(CHECKPOINT_PADRAO),
        "checkpoint_v6_semente": checkpoint["semente"],
        "checkpoint_v6_epoca": checkpoint["epoca"],
        "v6_modificada": False,
        "epocas_roteador": 5,
        "epocas_leitor_qk": 5,
        "configuracao_roteador": asdict(CONFIGURACAO_ROTEADOR),
        "auditoria_roteador": roteador.auditoria(),
        "historico": historico,
        "historico_leitor_qk": historico_qk,
        "manual": {
            "recall_candidatos": recall_candidatos(teste_manual),
            "qualidade": avaliar(
                modelo, teste_manual, manual_descritores
            ),
            "geracao_livre": geracao_manual,
        },
        "aprendido_sem_leitor_qk": {
            "qualidade": qualidade_aprendida_sem_leitor,
        },
        "aprendido_com_leitor_qk": {
            "recall_candidatos": recall_candidatos(
                teste_aprendido
            ),
            "qualidade": avaliar(
                modelo_adaptado,
                teste_aprendido,
                descritores_aprendidos,
            ),
            "geracao_livre": geracao_aprendida,
        },
        "benchmark": benchmark,
        "interpretacao": (
            "O roteador aprendido remove mapas semanticos manuais, mas usa "
            "supervisao contrastiva da posicao correta do fato. O leitor "
            "Q/K experimental tambem recebe a origem correta como alvo."
        ),
    }
    criterios = {
        "roteamento_sem_mapas_top1_minimo_99": (
            relatorio["aprendido_com_leitor_qk"]["recall_candidatos"][
                "acuracia_top1"
            ]
            >= 0.99
        ),
        "ppl_sem_leitor_ate_5_porcento_da_base": (
            qualidade_aprendida_sem_leitor["ppl"]
            <= relatorio["manual"]["qualidade"]["ppl"] * 1.05
        ),
        "geracao_livre_base_exata_minimo_90": (
            geracao_manual["sequencias_exatas"] >= 0.90
        ),
        "pipeline_completo_medido": True,
    }
    relatorio["criterios"] = criterios
    relatorio["decisao"] = (
        "promover"
        if all(criterios.values())
        else "nao_promover_roteador_aprendido_manter_v6_base"
    )
    serializado = json.dumps(
        relatorio, ensure_ascii=False, indent=2
    ) + "\n"
    (diretorio / "relatorio.json").write_text(
        serializado, encoding="utf-8"
    )
    (RESULTADOS / "teste_isolado_v6_ultimo.json").write_text(
        serializado, encoding="utf-8"
    )
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    print(f"RELATORIO={diretorio / 'relatorio.json'}")


if __name__ == "__main__":
    main()
