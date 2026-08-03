# Relatório do modelo V7.3 BPE-8192 — 5 × 3.000

## Configuração

- Dataset: `D:\AI-Dados\V7\alpaca-cleaned-pt-50k\bpe_codigo_8192`
- Registros: 47.500 treino e 2.500 validação
- Segmentos: 92.779 treino e 4.911 validação
- Tokenizador: BPE de código, 8.192 tokens
- Épocas: 5
- Passos por época: 3.000
- Lote: 32
- Contexto: 256
- Taxa inicial: 3e-4
- Warmup: 100 passos
- Backend: banco denso completo
- Projeção de logits: somente alvos, selecionada automaticamente
- AdamW fundido: ativo
- Prefetch e baldes: ativos

Cada época apresentou 96.000 segmentos ao otimizador. Isso cobre os 92.779 segmentos e repete aproximadamente 3,5% deles.

## Resultados

| Época | Loss treino resposta | Loss validação | PPL | BPB | Tempo de treino | Tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,655209 | 4,387640 | 80,450330 | 1,470240 | 114,16 s | 215.268,14 |
| 2 | 4,250306 | 4,204252 | 66,970475 | 1,408789 | 113,92 s | 215.737,60 |
| 3 | 4,103686 | 4,123086 | 61,749509 | 1,381591 | 114,55 s | 214.549,03 |
| 4 | 4,026454 | 4,084009 | 59,383086 | 1,368497 | 114,17 s | 215.254,32 |
| 5 | 3,988377 | 4,069222 | 58,511445 | 1,363542 | 114,42 s | 214.793,89 |

Tempo puro total: 571,2166 segundos, ou 9,5203 minutos. Vazão média: 215.120,60 tokens/s.

## Comparação com a base operacional anterior

- PPL: 166,421520 → 58,511445, redução de 64,84%
- BPB: 1,713809 → 1,363542, redução de 20,44%

A curva confirma que o modelo anterior estava subtreinado. Os ganhos diminuíram entre as épocas 4 e 5, indicando aproximação de um platô para esta configuração.

## Artefato mantido

`modelos/gerador_esparso_v73_bpe8192_5x3000.pt`

SHA-256:

`151e3bed5c9d2c6f366fa4ae5799de8577a4baa41f82b381dbb5039fbfa43b65`

O artefato contém pesos, arquitetura, tokenizador, vocabulário, identidade do dataset e métricas. Estados do otimizador e do agendador foram removidos.

## Limitação conhecida

Apesar da redução forte de PPL e BPB, a geração autorregressiva ainda pode entrar em repetição, principalmente no modo greedy. O modelo foi mantido por ser a melhor variante BPE testada, mas essa limitação permanece registrada.
