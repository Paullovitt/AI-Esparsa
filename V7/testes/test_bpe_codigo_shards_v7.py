"""Testes do BPE de codigo e dos shards uint16 da V7.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from executar_gerador_esparso_v73 import carregar_v73
from src.dados_binarios_v7 import (
    CorpusShardsV7,
    preparar_dataset_shards,
)
from src.tokenizador_palavras import (
    TokenizadorBPECodigo,
    tokenizador_de_checkpoint,
)
from src.topologia_checkpoint_v73 import metadados_topologia
from treinar_textual_v7 import (
    SEMENTE_TOPOLOGIA_V73,
    criar_modelos,
)


def escrever_jsonl(caminho: Path, registros: list[dict[str, str]]) -> None:
    caminho.write_text(
        "".join(
            json.dumps(registro, ensure_ascii=False) + "\n"
            for registro in registros
        ),
        encoding="utf-8",
    )


class TesteBPECodigoShardsV7(unittest.TestCase):
    def test_bpe_preserva_codigo_byte_a_byte(self) -> None:
        codigo = (
            "  class HTTPServer:\r\n"
            "\tdef executar(self, valor='Árvore'):\n"
            "\t\t# Caixa, tabs, CRLF e operadores.\n"
            "\t\treturn valor != None and valor == f\"{self!r}\"\n"
        )
        tokenizador = TokenizadorBPECodigo(
            [codigo] * 20 + ["x += 1\ny >>= 2\n"] * 20,
            tamanho_vocabulario=320,
        )
        ids = tokenizador.codificar(codigo, bos=True, eos=True)
        self.assertEqual(tokenizador.decodificar(ids), codigo)
        self.assertEqual((tokenizador.pad_id, tokenizador.bos_id), (0, 1))
        self.assertEqual((tokenizador.eos_id, tokenizador.unk_id), (2, 3))
        self.assertNotIn(tokenizador.unk_id, ids)
        self.assertEqual(
            sum(tokenizador.comprimentos_bytes[indice] for indice in ids),
            len(codigo.encode("utf-8")),
        )

    def test_tokenizador_e_reconstruido_do_checkpoint(self) -> None:
        codigo = "def Soma(A, B):\n    return A + B\n"
        tokenizador = TokenizadorBPECodigo(
            [codigo] * 30 + ["class Outro:\n\tpass\n"] * 30,
            tamanho_vocabulario=280,
        )
        checkpoint = {
            "tokenizador": {
                "tipo": tokenizador.TIPO,
                "serializado": tokenizador.serializado,
            },
            "vocabulario": tokenizador.id_para_token,
        }
        carregado = tokenizador_de_checkpoint(checkpoint)
        self.assertEqual(
            carregado.decodificar(carregado.codificar(codigo)),
            codigo,
        )
        self.assertEqual(carregado.id_para_token, tokenizador.id_para_token)

    def test_shards_uint16_sao_reversiveis_e_nao_vazam_validacao(self) -> None:
        treino = [
            {
                "prompt": "Complete sem alterar indentação:\n",
                "resposta": f"def funcao_{indice}(X):\n\treturn X ** 2\n",
            }
            for indice in range(24)
        ]
        validacao_a = [
            {
                "prompt": "Código exclusivo",
                "resposta": "VALIDACAO_EXCLUSIVA_ABCDEF = True\n",
            }
        ]
        validacao_b = [
            {
                "prompt": "Outro código",
                "resposta": "OUTRA_VALIDACAO_987654 = False\n",
            }
        ]
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            dados_a, dados_b = raiz / "dados_a", raiz / "dados_b"
            dados_a.mkdir()
            dados_b.mkdir()
            for dados, validacao in (
                (dados_a, validacao_a),
                (dados_b, validacao_b),
            ):
                escrever_jsonl(dados / "treino.jsonl", treino)
                escrever_jsonl(dados / "validacao.jsonl", validacao)
            saida_a, saida_b = raiz / "saida_a", raiz / "saida_b"
            preparar_dataset_shards(
                dados_a,
                saida_a,
                tamanho_vocabulario=320,
                tokens_por_shard=300,
            )
            preparar_dataset_shards(
                dados_b,
                saida_b,
                tamanho_vocabulario=320,
                tokens_por_shard=300,
            )
            artefato_a = json.loads(
                (saida_a / "tokenizador.json").read_text(encoding="utf-8")
            )
            artefato_b = json.loads(
                (saida_b / "tokenizador.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                artefato_a["serializado"],
                artefato_b["serializado"],
            )
            corpus = CorpusShardsV7(
                saida_a,
                "validacao",
                maximo_contexto=64,
            )
            self.assertEqual(
                corpus.decodificar_resposta(0),
                validacao_a[0]["resposta"],
            )
            self.assertEqual(
                corpus.decodificar_registro(0),
                "usuario: " + validacao_a[0]["prompt"]
                + "\nassistente: " + validacao_a[0]["resposta"],
            )
            tokens, labels = corpus.lote(
                list(range(len(corpus.exemplos))),
                torch.device("cpu"),
            )
            self.assertEqual(tokens.dtype, torch.long)
            self.assertEqual(
                int(labels.ne(-100).sum()),
                corpus.estatisticas["tokens_alvo"],
            )
            manifesto = corpus.manifesto
            self.assertEqual(manifesto["dtype"], "uint16_le")
            self.assertEqual(
                manifesto["treinado_apenas_no_split"],
                "treino",
            )
            for shard in manifesto["splits"]["treino"]["shards"]:
                caminho = saida_a / shard["tokens"]
                self.assertEqual(caminho.stat().st_size % 2, 0)
            tokenizador_path = saida_a / "tokenizador.json"
            tokenizador_path.write_text(
                tokenizador_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash do tokenizador"):
                CorpusShardsV7(
                    saida_a,
                    "treino",
                    maximo_contexto=64,
                )

    def test_modelo_e_executor_aceitam_vocabulario_8192(self) -> None:
        textos = (
            f"def funcao_{indice:05d}(valor_{indice:05d}):\n"
            f"    return valor_{indice:05d} + {indice}\n"
            for indice in range(12_000)
        )
        tokenizador = TokenizadorBPECodigo(
            textos,
            tamanho_vocabulario=8192,
        )
        modelo, arquitetura, roteamento, _ = criar_modelos(tokenizador)
        self.assertEqual(modelo.embedding.weight.shape, (8192, 128))
        self.assertEqual(modelo.saida_bias.shape, (8192,))
        checkpoint = {
            "modelo": "gerador-esparso-v7-textual-experimental",
            "semente": SEMENTE_TOPOLOGIA_V73,
            "configuracao": asdict(arquitetura),
            "configuracao_roteamento": asdict(roteamento),
            "tokenizador": {
                "tipo": tokenizador.TIPO,
                "serializado": tokenizador.serializado,
            },
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
            "topologia": metadados_topologia(modelo),
        }
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "checkpoint.pt"
            torch.save(checkpoint, caminho)
            carregado, restaurado, _ = carregar_v73(
                caminho,
                torch.device("cpu"),
                exigir_kernel_cuda=False,
            )
        self.assertEqual(restaurado.tamanho, 8192)
        self.assertEqual(carregado.embedding.weight.shape, (8192, 128))
        self.assertEqual(carregado.saida_bias.shape, (8192,))


if __name__ == "__main__":
    unittest.main()
