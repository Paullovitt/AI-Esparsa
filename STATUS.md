# Status da V6.1 Base

Autor: Paulo Augusto  
Ano: 2026

- Modelo-base oficial: V6.1 posicional.
- Checkpoint canônico: `modelos/v61_base.pt`.
- Modelo anterior: V6 preservada como rollback.
- Checkpoint de rollback: `modelos/v6_rollback.pt`.
- Semente canônica: 20260727.
- Época canônica: 5.
- Parâmetros: 9.632.
- Atenção causal: Q/K esparsos, Top-24.
- FFN: COO 48-144-48, Top-12 e 25% de densidade.
- `nn.Linear` dentro da FFN: zero.
- Treino V6.1: cinco épocas e cinco checkpoints preservados.
- Validação V6.1: quatro sementes inéditas e 72 gerações por semente.
- PPL média V6.1: 1,5317.
- PPL de teste V6.1: 1,5308.
- Acurácia de token média V6.1: 88,06%.
- Localização da resposta: 100%.
- Recuperação causal: 100%.
- Geração livre exata média V6.1: 93,06%.
- Tokens livres alinhados V6.1: 99,17%.
- Locais livres V6.1: 99,54% média e 99,07% no pior caso.
- Rollback V6, PPL média: 1,9196.
- Rollback V6, melhor PPL: 1,9077.
- Rollback V6, acurácia de token: 77,41%.
- Rollback V6, geração livre exata: 0,00%.
- Rollback V6, término em `EOS`: 61,11%.
- Rollback V6, tokens alinhados: 41,07%.
- Rollback V6, locais alinhados: 42,13%.
- Rollback V6, PPL sem FFN: 4,9899.
- Qualidade e compactação: aprovadas.
- Backend PyTorch integrado: reprovado em velocidade.
- Pipeline V6 completo, 73 tokens/lote 64: 0,555 M tokens/s.
- Pipeline V6 completo, 512 tokens/lote 16: 0,296 M tokens/s.
- Roteador aprendido sem mapas: Top-1 de 100%, PPL 2,6347, local 87,87% e
  recuperação 99,95%.
- Leitor Q/K adaptado: recuperação de 100%, mas PPL 3,6330.
- Decisão do teste isolado antigo: não promover o roteador aprendido.
- V6.1: posição senoidal fixa, três slots e dois papéis esparsos.
- Pipeline V6.1, 73 tokens/lote 64: 0,536 M tokens/s, 104,05% da V6.
- Pipeline V6.1, 512 tokens/lote 16: 0,279 M tokens/s, 99,72% da V6.
- VRAM temporária da V6.1: igual à V6 nas médias pareadas.
- Códigos temporais fixos: armazenados em cache, sem mudar as previsões.
- Decisão V6.1: promovida oficialmente; V6 preservada como rollback.
- Limite V6.1: slots e papéis ainda dependem da estrutura fixa do corpus.
- Próximo experimento: aprender slots e papéis em sequências variáveis.
- Próxima otimização: índice causal invertido para seleção de candidatos.
- Relatório: `resultados/v6_ultimo.json`.
- Relatório isolado: `resultados/teste_isolado_v6_ultimo.json`.
- Relatório V6.1: `resultados/v61_base_validacao.json`.
- Documento técnico: `DOCUMENTO_MODELO_V61.md`.
- Suíte automatizada: 23 testes aprovados.
