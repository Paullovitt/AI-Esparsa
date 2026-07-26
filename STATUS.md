# Status da V6

Autor: Paulo Augusto  
Ano: 2026

- Modelo-base oficial: V6.
- Checkpoint canônico: `modelos/v6_base.pt`.
- Semente canônica: 20260727.
- Época canônica: 5.
- Parâmetros: 9.621.
- Atenção causal: Q/K esparsos, Top-24.
- FFN: COO 48-144-48, Top-12 e 25% de densidade.
- `nn.Linear` dentro da FFN: zero.
- Treino validado: três sementes e cinco épocas.
- Checkpoints V6 preservados: 15.
- PPL média: 1,9196.
- Melhor PPL de teste: 1,9077.
- Acurácia de token média: 77,41%.
- Localização da resposta: 100%.
- Recuperação causal: 100%.
- Geração controlada: 100%.
- PPL sem FFN na ablação: 4,9902.
- Qualidade e compactação: aprovadas.
- Backend PyTorch integrado: reprovado em velocidade.
- Próxima otimização: kernel CUDA da FFN textual contínua.
- Relatório: `resultados/v6_ultimo.json`.
- Documento técnico: `DOCUMENTO_MODELO_V6.md`.
