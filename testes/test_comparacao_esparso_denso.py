"""Testes do protocolo comparativo esparso × denso.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

import torch

from comparar_esparso_denso import _protocolo, construir_comparacao
from src.modelo_gerador_denso import (
    ConfiguracaoGeradorDenso,
    ModeloGeradorDenso,
)
from src.tokenizador_palavras import TokenizadorPalavras
from src.versao import VERSAO_PROJETO


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados" / "comparacao_esparso_denso_50k"


def _relatorio_minimo(
    *,
    parametros: int,
    ppl: float,
    tempo: float,
    vram: float,
    velocidade: float,
) -> dict[str, object]:
    return {
        "arquitetura": {"parametros": parametros},
        "avaliacao_final": {
            "ppl": ppl,
            "acuracia_token": 0.95,
        },
        "geracao_livre": {
            "taxa_aprovacao": 1.0,
            "recuperacao_campos_pedido": 1.0,
            "repeticao_trigramas_media": 0.01,
            "taxa_uso_retentativa": 0.1,
        },
        "desempenho_forward": {
            "tokens_por_segundo": velocidade,
            "vram_pico_mib": vram / 10.0,
        },
        "desempenho_autorregressivo": {
            "tokens_por_segundo": velocidade / 100.0,
            "latencia_primeiro_token_ms": 10.0,
        },
        "tempo_treino_total_segundos": tempo,
        "vram_treino_pico_mib": vram,
        "checkpoint_final_bytes": parametros * 4,
        "ambiente": {"gpu": "teste"},
    }


class TesteComparacaoEsparsoDenso(unittest.TestCase):
    def test_protocolo_fixa_o_mesmo_treino_oficial(self) -> None:
        protocolo = _protocolo()
        self.assertEqual(protocolo["dados"]["treino"], 50_000)
        self.assertEqual(protocolo["epocas"], 5)
        self.assertEqual(protocolo["lote"], 100)
        self.assertEqual(protocolo["passos_por_epoca"], 500)
        self.assertEqual(protocolo["passos_totais"], 2_500)

    def test_comparacao_respeita_direcao_das_metricas(self) -> None:
        esparso = _relatorio_minimo(
            parametros=163_667,
            ppl=1.05,
            tempo=100.0,
            vram=500.0,
            velocidade=1_000.0,
        )
        denso = _relatorio_minimo(
            parametros=163_003,
            ppl=1.10,
            tempo=120.0,
            vram=600.0,
            velocidade=800.0,
        )
        comparacao = construir_comparacao(esparso, denso)
        self.assertEqual(
            comparacao["metricas"]["ppl"]["vencedor"],
            "esparso",
        )
        self.assertEqual(
            comparacao["metricas"]["tempo_treino_segundos"][
                "vencedor"
            ],
            "esparso",
        )
        self.assertEqual(
            comparacao["metricas"]["parametros"]["vencedor"],
            "denso",
        )

    def test_versao_python_corresponde_ao_pyproject(self) -> None:
        with (RAIZ / "pyproject.toml").open("rb") as arquivo:
            pyproject = tomllib.load(arquivo)
        self.assertEqual(
            VERSAO_PROJETO,
            pyproject["project"]["version"],
        )

    def test_cinco_checkpoints_densos_recarregam_estritamente(
        self,
    ) -> None:
        for epoca in range(1, 6):
            checkpoint = torch.load(
                RESULTADOS / "denso" / f"epoca_{epoca:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["epoca"], epoca)
            self.assertEqual(checkpoint["epocas_planejadas"], 5)
            self.assertEqual(checkpoint["passos_por_epoca"], 500)
        tokenizador = TokenizadorPalavras.de_vocabulario(
            checkpoint["vocabulario"]
        )
        modelo = ModeloGeradorDenso(
            tokenizador.tamanho,
            tokenizador.pad_id,
            ConfiguracaoGeradorDenso(**checkpoint["configuracao"]),
            int(checkpoint["semente"]),
        )
        modelo.load_state_dict(
            checkpoint["estado_modelo"],
            strict=True,
        )
        with torch.inference_mode():
            logits, _ = modelo(
                torch.tensor(
                    [[tokenizador.bos_id, tokenizador.eos_id]]
                )
            )
        self.assertEqual(
            logits.shape,
            (1, 2, tokenizador.tamanho),
        )

    def test_relatorio_comparativo_registra_resultado_honesto(
        self,
    ) -> None:
        comparacao = json.loads(
            (RESULTADOS / "comparacao.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            comparacao["vantagem_pratica_esparsa_sustentada"]
        )
        self.assertEqual(
            comparacao["metricas"]["aprovacao_geracao"]["vencedor"],
            "empate",
        )
        self.assertEqual(
            comparacao["metricas"]["ppl"]["vencedor"],
            "denso",
        )
        self.assertTrue(
            comparacao["revalidacao_independente"]["aprovado"]
        )
        self.assertEqual(
            comparacao["modelo_superior_neste_protocolo"],
            "denso",
        )


if __name__ == "__main__":
    unittest.main()
