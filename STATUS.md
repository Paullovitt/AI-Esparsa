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
- Primeiro token do local na avaliação controlada: 100%.
- Geração livre exata da V6 em 72 prompts: 0,00%.
- Geração livre encerrada em `EOS`: 61,11%.
- Acurácia de tokens alinhados na geração livre: 41,07%.
- Acurácia de locais alinhados na geração livre: 42,13%.
- PPL sem FFN na ablação: 4,9899.
- Qualidade e compactação: aprovadas.
- Backend PyTorch integrado: reprovado em velocidade.
- Pipeline V6 completo, 73 tokens/lote 64: 0,555 M tokens/s.
- Pipeline V6 completo, 512 tokens/lote 16: 0,296 M tokens/s.
- Roteador aprendido sem mapas: Top-1 de 100%, PPL 2,6347, local 87,87% e
  recuperação 99,95%.
- Leitor Q/K adaptado: recuperação de 100%, mas PPL 3,6330.
- Decisão do teste isolado: não promover; manter a V6 canônica intacta.
- Próxima otimização: kernel CUDA da FFN textual contínua.
- Relatório: `resultados/v6_ultimo.json`.
- Relatório isolado: `resultados/teste_isolado_v6_ultimo.json`.
- Documento técnico: `DOCUMENTO_MODELO_V6.md`.
- Suíte automatizada: 15 testes aprovados.
