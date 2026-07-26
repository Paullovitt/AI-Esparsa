# Status da V6

Autor: Paulo Augusto  
Ano: 2026

- Modelo-base oficial: V6.
- Checkpoint canônico: `modelos/v6_base.pt`.
- Evolução aprovada como candidata: V6.1 posicional.
- Checkpoint candidato: `modelos/v61_candidata.pt`.
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
- V6.1: posição senoidal fixa, três slots e dois papéis esparsos.
- Parâmetros V6.1: 9.632, somente 11 a mais que a V6.
- PPL V6.1 em quatro sementes inéditas: 1,5317 média.
- Acurácia de token V6.1: 88,06% média.
- Geração livre exata V6.1: 93,06% média.
- Tokens livres alinhados V6.1: 99,17% média.
- Locais livres V6.1: 99,54% média e 99,07% no pior caso.
- Pipeline V6.1, 73 tokens/lote 64: 0,536 M tokens/s, 104,05% da V6.
- Pipeline V6.1, 512 tokens/lote 16: 0,279 M tokens/s, 99,72% da V6.
- VRAM temporária da V6.1: igual à V6 nas médias pareadas.
- Códigos temporais fixos: armazenados em cache, sem mudar as previsões.
- Decisão V6.1: candidata aprovada; V6 preservada como rollback.
- Limite V6.1: slots e papéis ainda dependem da estrutura fixa do corpus.
- Próximo experimento: aprender slots e papéis em sequências variáveis.
- Próxima otimização: índice causal invertido para seleção de candidatos.
- Relatório: `resultados/v6_ultimo.json`.
- Relatório isolado: `resultados/teste_isolado_v6_ultimo.json`.
- Relatório V6.1: `resultados/v61_candidata_validacao.json`.
- Documento técnico: `DOCUMENTO_MODELO_V6.md`.
- Suíte automatizada: 23 testes aprovados.
