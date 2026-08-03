# Documento técnico — AI-Esparsa V7.3

Autor: Paulo Augusto

Versão do código: 2.8.0

Ano: 2026

## 1. Escopo

Esta pasta mantém exclusivamente a base textual V7.3 com tokenizador BPE de código e vocabulário 8.192. Não contém a arquitetura multimodal V8 nem a baseline densa.

## 2. Arquitetura canônica

O checkpoint principal usa dimensão 128, três blocos e contexto máximo 640. Cada bloco contém normalização pré-atenção, Q/K COO 128 × 128 com fan-in 32, atenção causal Top-32, residual com gate, normalização pré-FFN, banco FFN COO com 384 unidades, 16 microgrupos, quatro grupos ativos por token e residual final com gate.

Embedding e projeção de saída compartilham pesos. O roteador baixo-rank seleciona grupos por token. Os grupos reutilizam um banco de arestas COO e não são especialistas densos independentes.

## 3. Contrato textual

Treino, avaliação e execução usam exatamente:

```text
usuario: <prompt>
assistente: <resposta>
```

O executor aplica o prefixo automaticamente. O modo cru existe somente para sequências já preparadas. A saída interativa remove o prefixo e imprime apenas os tokens novos.

## 4. Topologia e checkpoint

Os índices COO não fazem parte do `state_dict`, pois são buffers não persistentes. Para impedir a associação silenciosa de pesos a conexões diferentes, o formato `coo-v1` calcula SHA-256 sobre:

- nome de cada camada COO;
- dimensões de entrada e saída;
- quantidade de conexões;
- índices COO em int64 e ordem canônica.

O checkpoint guarda o hash e o resumo das camadas. O loader reconstrói a topologia, recalcula a identidade e falha antes de carregar os pesos quando houver divergência. Checkpoints legados exigem migração ou autorização explícita.

## 5. Runtime e memória

O caminho de referência calcula atenção Top-K em blocos e não materializa a matriz completa de escores, embora a computação total continue quadrática no tempo. O caminho vetorizado de lotes CUDA materializa tensores `T × T`; sua auditoria declara memória quadrática.

O FFN CUDA fundido pode ser usado no checkpoint BPE-8192. O caminho incremental totalmente fundido inclui projeção vocabular limitada a 512 tokens e, portanto, não é elegível para o vocabulário 8.192. Nesse modelo, a auditoria registra `caminho_incremental_fundido=false` e zero lançamentos desse caminho.

## 6. Treinamento configurável

As opções arquiteturais do CLI são convertidas diretamente em `ConfiguracaoGeradorEsparso`. A configuração padrão coincide com o checkpoint: 128 dimensões, três camadas, Top-32, fan-in Q/K 32, 384 unidades FFN, fan-in 32, fan-out 48 e Top-64 FFN.

A arquitetura não usa cabeças de atenção; esse parâmetro foi removido. O backend de treinamento pode materializar pesos densos temporários por desempenho, mas os únicos pesos treináveis internos permanecem COO.

## 7. Dados e portabilidade

Datasets não fazem parte do repositório. Caminhos são fornecidos por CLI ou pelas variáveis `AI_ESPARSA_DADOS_BRUTOS`, `AI_ESPARSA_DATASET` e `AI_ESPARSA_SAIDA`. Os padrões relativos ficam dentro de `dados/` e `resultados/`, ambos ignorados pelo Git.

O BPE é treinado somente no split de treino. Shards e tokenizador são validados por hash. A tabela de compatibilidade GPT-2 usa as faixas de bytes 33–126, 161–172 e 174–255.

## 8. Métricas e promoção

Loss e PPL medem previsão token a token, não coerência de resposta. O avaliador gerativo registra respostas sem o prompt, diversidade, repetição, EOS, latência e throughput. A amostragem só é aprovada quando passa em todos os limites explícitos do relatório.

O BPB histórico foi publicado como 1,3635424677. Corrigindo a tabela GPT-2, o mesmo total supervisionado resulta em aproximadamente 1,3635468168. A diferença é apenas métrica; checkpoint e tokenização permanecem idênticos.

## 9. Limitações

O checkpoint atual ainda gera texto repetitivo e por vezes incoerente. Penalidades e bloqueios de n-grama são controles de decodificação, não correções do aprendizado. Uma promoção de qualidade exige novo treino, conjunto fixo de prompts, critérios automáticos aprovados e avaliação humana.
