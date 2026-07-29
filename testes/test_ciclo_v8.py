"""Testes de transferência e carregamento seguro da V8.

Autor: Paulo Augusto
Ano: 2026
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from executar_multimodal_v8 import carregar_v8
from src.modelo_gerador_esparso import ConfiguracaoGeradorEsparso
from src.modelo_multimodal_v8 import (
    ConfiguracaoMultimodalV8,
    ModeloMultimodalEsparsoV8,
    ModeloMultimodalEsparsoTreinoV8,
)
from src.roteamento_combinatorio_v73 import ConfiguracaoRoteamentoV73
from src.tokenizador_palavras import TokenizadorPalavras
from treinar_multimodal_v8 import inicializar_backbone_v73


RAIZ = Path(__file__).resolve().parents[1]
CHECKPOINT_V73 = RAIZ / "modelos" / "gerador_esparso_v73_base.pt"


class TesteCicloV8(unittest.TestCase):
    def test_transferencia_v73_preserva_orcamento_documentado(self) -> None:
        checkpoint = torch.load(
            CHECKPOINT_V73,
            map_location="cpu",
            weights_only=True,
        )
        tokenizador = TokenizadorPalavras.de_vocabulario(
            checkpoint["vocabulario"]
        )
        modelo = ModeloMultimodalEsparsoTreinoV8(
            tokenizador.tamanho,
            tokenizador.pad_id,
            ConfiguracaoGeradorEsparso(**checkpoint["configuracao"]),
            int(checkpoint["semente"]),
            ConfiguracaoRoteamentoV73(
                **checkpoint["configuracao_roteamento"]
            ),
        )
        transferencia = inicializar_backbone_v73(
            modelo,
            CHECKPOINT_V73,
        )
        self.assertEqual(transferencia["tensores_transferidos"], 56)
        self.assertEqual(transferencia["tensores_inicializados_v8"], 7)
        self.assertEqual(modelo.auditoria()["parametros"], 172_228)

    def test_checkpoint_v8_round_trip_com_weights_only(self) -> None:
        tokenizador = TokenizadorPalavras(
            ["pergunta: cor resposta: azul"]
        )
        base = ConfiguracaoGeradorEsparso(
            dimensao=16,
            camadas=1,
            top_k_atencao=4,
            fan_in_qk=4,
            especialistas=16,
            fan_in_ffn=4,
            fan_out_ffn=4,
            top_k_ffn=4,
            maximo_contexto=24,
            bloco_consultas_atencao=4,
        )
        roteamento = ConfiguracaoRoteamentoV73(
            grupos=4,
            grupos_ativos=2,
            dimensao_latente=2,
        )
        multimodal = ConfiguracaoMultimodalV8(
            canais_imagem=3,
            tamanho_patch=4,
            altura_imagem=8,
            largura_imagem=8,
            fan_in_projecao_visual=8,
            semente_visual=321,
        )
        modelo = ModeloMultimodalEsparsoTreinoV8(
            tokenizador.tamanho,
            tokenizador.pad_id,
            base,
            123,
            roteamento,
            multimodal,
        ).eval()
        checkpoint = {
            "modelo": "gerador-esparso-v8-multimodal",
            "epoca": 1,
            "semente": 123,
            "configuracao": asdict(base),
            "configuracao_roteamento": asdict(roteamento),
            "configuracao_multimodal": asdict(multimodal),
            "configuracao_treino": asdict(modelo.configuracao_treino),
            "vocabulario": tokenizador.id_para_token,
            "estado_modelo": modelo.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporaria:
            caminho = Path(temporaria) / "v8.pt"
            torch.save(checkpoint, caminho)
            carregado, _, bruto = carregar_v8(
                caminho,
                torch.device("cpu"),
            )
        self.assertEqual(bruto["modelo"], "gerador-esparso-v8-multimodal")
        self.assertEqual(
            set(carregado.state_dict()),
            set(modelo.state_dict()),
        )


if __name__ == "__main__":
    unittest.main()
