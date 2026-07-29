"""Testes do manifesto seguro e do lote multimodal V8.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from src.dados_multimodais_v8 import (
    DatasetMultimodalV8,
    agrupar_lote_multimodal,
    carregar_manifesto_multimodal,
    criar_tokenizador_multimodal,
)


class TesteDadosMultimodaisV8(unittest.TestCase):
    def _preparar(self, pasta: Path) -> tuple[Path, Path]:
        raiz = pasta / "imagens"
        raiz.mkdir()
        Image.new("RGB", (12, 10), color=(10, 20, 30)).save(
            raiz / "exemplo.png"
        )
        manifesto = pasta / "treino.jsonl"
        manifesto.write_text(
            json.dumps(
                {
                    "id": "amostra-1",
                    "imagem": "exemplo.png",
                    "prompt": "qual e a cor",
                    "resposta": "a imagem e escura",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return raiz, manifesto

    def test_manifesto_dataset_e_lote(self) -> None:
        with tempfile.TemporaryDirectory() as temporaria:
            raiz, manifesto = self._preparar(Path(temporaria))
            registros = carregar_manifesto_multimodal(manifesto, raiz)
            tokenizador = criar_tokenizador_multimodal(registros)
            dataset = DatasetMultimodalV8(
                registros,
                tokenizador,
                altura_imagem=8,
                largura_imagem=8,
            )
            entradas, alvos, imagem = dataset[0]
            self.assertEqual(tuple(imagem.shape), (3, 8, 8))
            self.assertGreaterEqual(float(imagem.min()), -1.0)
            self.assertLessEqual(float(imagem.max()), 1.0)
            lote = agrupar_lote_multimodal(
                [(entradas, alvos, imagem)],
                tokenizador.pad_id,
            )
            self.assertEqual(tuple(lote[0].shape), (1, entradas.numel()))
            self.assertEqual(tuple(lote[2].shape), (1, 3, 8, 8))
            self.assertTrue(torch.isfinite(lote[2]).all())
            self.assertGreater(int(alvos.eq(tokenizador.pad_id).sum()), 0)

    def test_caminho_absoluto_e_travessia_sao_rejeitados(self) -> None:
        with tempfile.TemporaryDirectory() as temporaria:
            pasta = Path(temporaria)
            raiz, manifesto = self._preparar(pasta)
            bruto = {
                "id": "x",
                "imagem": str((raiz / "exemplo.png").resolve()),
                "prompt": "p",
                "resposta": "r",
            }
            manifesto.write_text(
                json.dumps(bruto) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "relativo"):
                carregar_manifesto_multimodal(manifesto, raiz)
            bruto["imagem"] = "../fora.png"
            manifesto.write_text(
                json.dumps(bruto) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapou"):
                carregar_manifesto_multimodal(manifesto, raiz)

    def test_json_invalido_campo_ausente_e_id_duplicado(self) -> None:
        with tempfile.TemporaryDirectory() as temporaria:
            pasta = Path(temporaria)
            raiz, manifesto = self._preparar(pasta)
            manifesto.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON invalido"):
                carregar_manifesto_multimodal(manifesto, raiz)
            manifesto.write_text(
                json.dumps({"id": "x"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sem campos"):
                carregar_manifesto_multimodal(manifesto, raiz)
            linha = json.dumps(
                {
                    "id": "x",
                    "imagem": "exemplo.png",
                    "prompt": "p",
                    "resposta": "r",
                }
            )
            manifesto.write_text(
                linha + "\n" + linha + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicado"):
                carregar_manifesto_multimodal(manifesto, raiz)


if __name__ == "__main__":
    unittest.main()
