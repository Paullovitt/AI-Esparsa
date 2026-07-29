"""Testes do corpus, condicionamento e checkpoint do gerador esparso."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import torch

from executar_gerador_esparso import carregar_gerador
from promover_gerador_esparso import validar_promocao
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_gerador_esparso import (
    _medir_continuacao,
    acoes_consistentes,
    codificar_registros,
    concordancia_de_genero_consistente,
    extrair_campos_pedido,
    gerar_divisoes_gerador,
    gerar_registro_extenso,
    local_do_objeto_consistente,
    problema_recuperado,
    validar_prompt_publico,
)
from src.corpus_gerador_esparso import PESSOAS, TAREFAS


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    RAIZ
    / "modelos"
    / "gerador_esparso_base.pt"
)
RESULTADOS = RAIZ / "resultados" / "gerador_esparso_base_50k"


class TestePipelineGeradorEsparso(unittest.TestCase):
    def test_extrai_campos_do_formato_publico_do_prompt(self) -> None:
        campos = extrair_campos_pedido(
            "Pedido: escreva um relato sobre bruno, com ajuda de tiago, "
            "para preparar uma feira de ciencias. Inclua a caixa na "
            "estante e o problema uma janela quebrada. Texto:"
        )
        self.assertEqual(
            campos,
            [
                "bruno",
                "tiago",
                "preparar uma feira de ciencias",
                "caixa",
                "estante",
                "uma janela quebrada",
            ],
        )
        self.assertIsNone(extrair_campos_pedido("prompt livre"))

    def test_divisoes_sao_disjuntas_e_historia_recebe_reforco(self) -> None:
        treino, validacao, teste = gerar_divisoes_gerador(
            (30, 10, 10),
            (101_000, 202_000, 303_000),
        )
        chaves = {
            tuple(registro["chave"])
            for registro in treino + validacao + teste
        }
        self.assertEqual(len(chaves), 50)
        tokenizador = TokenizadorPalavras(
            str(registro["texto"])
            for registro in treino + validacao + teste
        )
        codificados = codificar_registros(treino, tokenizador)
        self.assertTrue(
            all(max(pesos) == 3.0 for _, pesos in codificados)
        )

    def test_parafrase_de_problema_e_consistencia_de_local(self) -> None:
        texto = (
            "lucas deixou a chave na oficina. "
            "a chuva atrasou a entrega dos materiais. "
            "a chave continuava na oficina."
        )
        self.assertTrue(problema_recuperado("a chuva forte", texto))
        self.assertTrue(
            local_do_objeto_consistente("chave", "oficina", texto)
        )
        contraditorio = texto + " depois a chave ficou na cozinha."
        self.assertFalse(
            local_do_objeto_consistente(
                "chave",
                "oficina",
                contraditorio,
            )
        )
        self.assertFalse(
            local_do_objeto_consistente(
                "livro",
                "sala",
                (
                    "o livro foi levado para o deposito, mas o livro "
                    "continuava na sala."
                ),
            )
        )
        self.assertTrue(
            local_do_objeto_consistente(
                "relatorio",
                "cozinha",
                (
                    "o documento foi levado para o escritorio, sem "
                    "interferir no local reservado para o relatorio. "
                    "o relatorio ficou na cozinha."
                ),
            )
        )
        self.assertTrue(
            acoes_consistentes(
                "o primeiro passo foi mover as caixas para uma area segura."
            )
        )
        self.assertFalse(
            acoes_consistentes(
                "o primeiro passo foi mover as mudancas em uma area segura."
            )
        )

    def test_novo_corpus_corrige_contracoes_e_genero(self) -> None:
        treino, _, _ = gerar_divisoes_gerador(
            (200, 10, 10),
            (4_101_000, 4_202_000, 4_303_000),
        )
        textos = " ".join(str(registro["texto"]) for registro in treino)
        self.assertNotIn("posicao de o ", textos)
        self.assertNotIn("posicao de a ", textos)
        self.assertIn("disposta a colaborar", textos)
        self.assertIn("disposto a colaborar", textos)
        for registro in treino:
            principal = str(registro["palavras_chave"][3])
            correspondencia = re.search(
                r"Em outra parte do ambiente, (?:o|a) (\w+) foi",
                str(registro["historia"]),
            )
            self.assertIsNotNone(correspondencia)
            self.assertNotEqual(principal, correspondencia.group(1))

    def test_avaliador_rejeita_concordancia_incorreta(self) -> None:
        self.assertFalse(
            concordancia_de_genero_consistente(
                "o relatorio foi levada para o deposito."
            )
        )
        self.assertFalse(
            concordancia_de_genero_consistente(
                "a camera foi levado para a sala."
            )
        )
        self.assertTrue(
            concordancia_de_genero_consistente(
                "o relatorio foi levado e a camera foi levada."
            )
        )

    def test_prompt_oov_ou_fora_do_formato_e_rejeitado(self) -> None:
        treino, _, _ = gerar_divisoes_gerador(
            (10, 1, 1),
            (71_000, 72_000, 73_000),
        )
        tokenizador = TokenizadorPalavras(
            str(registro["texto"]) for registro in treino
        )
        with self.assertRaisesRegex(ValueError, "formato estruturado"):
            validar_prompt_publico("escreva qualquer coisa", tokenizador)
        desconhecido = (
            "Pedido: escreva um relato sobre joão, com ajuda de pedro, "
            "para lançar um foguete. Inclua o telescópio no laboratório "
            "e o problema um vazamento de gás. Texto:"
        )
        with self.assertRaisesRegex(ValueError, "joão.*pedro.*foguete"):
            validar_prompt_publico(desconhecido, tokenizador)
        pessoa, ajudante, tarefa, objeto, local, problema = (
            treino[0]["chave"][:6]
        )
        ordem_invalida = (
            f"Pedido: escreva um relato sobre {pessoa}, com ajuda de "
            f"{ajudante}, para {tarefa}. Inclua o {local} no {objeto} "
            f"e o problema {problema}. Texto:"
        )
        with self.assertRaisesRegex(ValueError, "formato estruturado"):
            validar_prompt_publico(ordem_invalida, tokenizador)

    def test_relato_extenso_supera_dois_mil_caracteres(self) -> None:
        registro = gerar_registro_extenso(40_000_000)
        historia = str(registro["historia"])
        palavras = [str(item) for item in registro["palavras_chave"]]
        self.assertGreaterEqual(len(historia), 2_000)
        self.assertEqual(registro["quantidade_frases"], 24)
        self.assertTrue(acoes_consistentes(historia))
        self.assertTrue(
            local_do_objeto_consistente(
                palavras[3],
                palavras[4],
                historia,
            )
        )
        self.assertTrue(problema_recuperado(palavras[5], historia))
        self.assertNotIn(" a camera foi levado ", historia.lower())
        self.assertNotIn(" a chave foi levado ", historia.lower())
        self.assertNotIn(" a caixa foi levado ", historia.lower())
        self.assertNotIn(" a pasta foi levado ", historia.lower())
        self.assertNotIn(" a maquete foi levado ", historia.lower())
        self.assertNotIn(" levado na ", historia.lower())
        self.assertNotIn(" levado no ", historia.lower())

    def test_avaliador_rejeita_tarefa_trocada(self) -> None:
        registro = gerar_registro_extenso(40_000_000)
        palavras = [str(item) for item in registro["palavras_chave"]]
        tarefa_errada = next(
            tarefa for tarefa in TAREFAS if tarefa != palavras[2]
        )
        historia_errada = str(registro["historia"]).replace(
            palavras[2],
            tarefa_errada,
        )
        tokenizador = TokenizadorPalavras([str(registro["texto"])])

        medidas = _medir_continuacao(
            tokenizador,
            historia_errada,
            palavras,
        )

        self.assertLess(medidas["cobertura_palavras_chave"], 1.0)

    def test_resposta_em_prosa_nao_e_vazamento_de_qa(self) -> None:
        treino, _, _ = gerar_divisoes_gerador(
            (2, 1, 1),
            (91_000, 92_000, 93_000),
        )
        tokenizador = TokenizadorPalavras(
            str(registro["texto"]) for registro in treino
        )
        palavras = [
            str(valor).lower()
            for valor in treino[0]["palavras_chave"]
        ]
        medidas = _medir_continuacao(
            tokenizador,
            (
                str(treino[0]["historia"])
                + " Uma resposta cuidadosa foi registrada."
            ),
            palavras,
        )
        self.assertFalse(medidas["vazou_bloco_pergunta"])
        medidas_qa = _medir_continuacao(
            tokenizador,
            str(treino[0]["historia"]) + " Resposta: exemplo.",
            palavras,
        )
        self.assertTrue(medidas_qa["vazou_bloco_pergunta"])

    def test_concordancia_do_grupo_em_todas_as_combinacoes(self) -> None:
        genero = dict(PESSOAS)
        encontrou_duas_mulheres = False
        for indice in range(500):
            registro = gerar_registro_extenso(
                80_000_000 + indice * 7_919
            )
            pessoa, ajudante = registro["chave"][:2]
            historia = str(registro["historia"]).lower()
            duas_mulheres = (
                genero[str(pessoa)] == "ela"
                and genero[str(ajudante)] == "ela"
            )
            if duas_mulheres:
                encontrou_duas_mulheres = True
                self.assertIn("as duas revisaram", historia)
                self.assertIn("as duas reorganizaram", historia)
                self.assertIn("na retomada, elas decidiram", historia)
                self.assertNotIn("os dois", historia)
                self.assertNotIn("eles decidiram", historia)
            else:
                self.assertIn("os dois revisaram", historia)
                self.assertIn("os dois reorganizaram", historia)
                self.assertIn("na retomada, eles decidiram", historia)
        self.assertTrue(encontrou_duas_mulheres)

    def test_promocao_rejeita_relatorio_sem_criterios(self) -> None:
        checkpoint = {
            "modelo": "gerador-esparso-coerente-base",
            "epoca": 5,
            "epocas_planejadas": 5,
            "dados_treino": 50_000,
            "passos_por_epoca": 500,
        }
        relatorio = {
            "revalidacao": {"aprovado": False, "criterios": {}},
            "geracao_livre": {"caracteres_minimos": 2_000},
        }
        with self.assertRaisesRegex(RuntimeError, "revalidacao_aprovada"):
            validar_promocao(checkpoint, relatorio)

    def test_checkpoint_final_recarrega_estritamente(self) -> None:
        self.assertTrue(CHECKPOINT.exists())
        modelo, tokenizador, checkpoint = carregar_gerador(
            CHECKPOINT,
            torch.device("cpu"),
        )
        self.assertEqual(checkpoint["epoca"], 5)
        self.assertEqual(checkpoint["epocas_planejadas"], 5)
        self.assertEqual(checkpoint["dados_treino"], 50_000)
        self.assertEqual(checkpoint["passos_por_epoca"], 500)
        self.assertTrue(
            checkpoint.get(
                "checkpoint_oficial_preservado",
                checkpoint.get("modelo_oficial_preservado"),
            )
        )
        tokens = torch.tensor(
            [tokenizador.codificar("pedido: texto:", eos=False)]
        )
        with torch.inference_mode():
            logits, _ = modelo(tokens)
        self.assertEqual(
            logits.shape,
            (1, tokens.shape[1], tokenizador.tamanho),
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_cinco_checkpoints_e_relatorio_aprovado(self) -> None:
        pasta = RESULTADOS
        for epoca in range(1, 6):
            caminho = pasta / f"epoca_{epoca:02d}.pt"
            self.assertTrue(caminho.exists())
            checkpoint = torch.load(
                caminho,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["epoca"], epoca)
            self.assertEqual(checkpoint["epocas_planejadas"], 5)
        relatorio = json.loads(
            (pasta / "relatorio.json").read_text(encoding="utf-8")
        )
        self.assertTrue(relatorio["revalidacao"]["aprovado"])
        self.assertTrue(
            all(
                relatorio["revalidacao"]["criterios"].values()
            )
        )
        self.assertEqual(
            relatorio["geracao_livre"]["taxa_aprovacao"],
            1.0,
        )
        self.assertEqual(relatorio["dados"]["treino"], 50_000)
        self.assertEqual(relatorio["passos_por_epoca"], 500)
        self.assertEqual(relatorio["passos_totais"], 2_500)
        self.assertGreaterEqual(
            relatorio["geracao_livre"]["caracteres_minimos"],
            2_000,
        )
        self.assertEqual(
            len(relatorio["geracao_livre"]["exemplos"]),
            24,
        )
        for exemplo in relatorio["geracao_livre"]["exemplos"]:
            campos = extrair_campos_pedido(str(exemplo["pedido"]))
            self.assertIsNotNone(campos)
            self.assertTrue(
                local_do_objeto_consistente(
                    campos[3],
                    campos[4],
                    str(exemplo["texto"]),
                )
            )
        self.assertGreater(
            relatorio["desempenho_autorregressivo"][
                "tokens_por_segundo"
            ],
            0.0,
        )
        self.assertEqual(
            relatorio["revalidacao"]["versao_validador"],
            "3.0.0",
        )
        validar_promocao(
            torch.load(
                CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            ),
            relatorio,
        )

    def test_repositorio_mantem_somente_v62_v73_e_denso(
        self,
    ) -> None:
        """Impede a reintrodução das versões experimentais descartadas."""

        implementacoes = sorted(
            caminho.name
            for caminho in (RAIZ / "src").glob("modelo_*.py")
        )
        checkpoints = sorted(
            caminho.name
            for caminho in (RAIZ / "modelos").glob("*.pt")
        )
        resultados = sorted(
            caminho.name
            for caminho in (RAIZ / "resultados").iterdir()
            if caminho.is_dir()
        )
        self.assertEqual(
            implementacoes,
            [
                "modelo_gerador_denso.py",
                "modelo_gerador_esparso.py",
                "modelo_gerador_esparso_v62.py",
                "modelo_gerador_esparso_v73.py",
            ],
        )
        self.assertEqual(
            checkpoints,
            [
                "gerador_esparso_base.pt",
                "gerador_esparso_v73_base.pt",
            ],
        )
        self.assertIn("gerador_esparso_base_50k", resultados)
        self.assertTrue(
            set(resultados).issubset(
                {
                    "gerador_esparso_base_50k",
                    "comparacao_esparso_denso_50k",
                    "v62_base_runtime",
                    "v73_base",
                }
            )
        )
        self.assertFalse(
            list(
                (
                    RAIZ / "resultados" / "v62_base_runtime"
                ).glob("*.pt")
            )
        )
        checkpoint_v73 = torch.load(
            RAIZ / "modelos" / "gerador_esparso_v73_base.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(
            checkpoint_v73["modelo"],
            "gerador-esparso-v73-base",
        )
        self.assertFalse(checkpoint_v73["treinado_em_bf16"])


if __name__ == "__main__":
    unittest.main()
