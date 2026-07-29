# Status do AI-Esparsa

Autor: Paulo Augusto  
Ano: 2026

## Modelos mantidos

- V6.2: base estável oficial.
- V7.3: base de desenvolvimento ativa.
- Denso: controle experimental.
- Outras versões: removidas.
- Candidato V7.3 BF16: removido.

## Checkpoints

- V6.2: `modelos/gerador_esparso_base.pt`;
- V7.3: `modelos/gerador_esparso_v73_base.pt`;
- denso:
  `resultados/comparacao_esparso_denso_50k/denso/epoca_05.pt`.

V6.2 SHA-256:

```text
daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f
```

## V6.2

- 163.667 parâmetros;
- cinco épocas;
- Q/K e FFN COO;
- cache causal, CSR e gates cacheados;
- checkpoint oficial preservado.

<!-- metricas-desempenho:inicio -->
- Forward paralelo: 145.417,74 tokens/s e 83,42 MiB de VRAM.
- Geração autorregressiva: 313,24 tokens/s, primeiro token em 4,63 ms e
  relato completo em 1,56 s.
<!-- metricas-desempenho:fim -->

## V7.3

- 165.443 parâmetros FP32;
- duas épocas históricas;
- 16 microgrupos, quatro ativos por token;
- 1.820 combinações teóricas por camada;
- kernel CUDA fundido;
- fallback PyTorch condicional;
- zero parâmetros adicionais de runtime;
- base de desenvolvimento;
- ainda não elegível para promoção formal.
- modelo puro: 9/24; sistema controlado: 24/24.

## Qualidade textual

- restrição causal de concordância ativa;
- gramática causal das seis ações válidas;
- objeto de apoio impedido de repetir o objeto principal;
- validador verifica gênero, tarefa, objeto/local e ações;
- `o relatorio foi levada` é bloqueado e reprovado;
- as 24 gerações de cada modelo ficam no relatório comparativo.

## Denso

- 163.003 parâmetros;
- cinco épocas;
- controle de orçamento equivalente;
- não promovível automaticamente.

## Relatórios

- comparação final: `resultados/v73_base/comparacao.json`;
- resumo: `resultados/v73_base/RELATORIO_V73.md`;
- textos: `resultados/v73_base/EXEMPLOS_COMPARATIVOS.md`;
- treino-base V7.3: `resultados/v73_base/treino_base.json`.

## Ciclo V7.3

- o executor aceita a base histórica de duas épocas e candidatos FP32 de uma
  a cinco épocas do ciclo controlado;
- apenas a época 5 recebe `elegivel_promocao = true`;
- o comparador aceita caminhos configuráveis e guarda o SHA-256 do candidato;
- o promotor valida 50 mil amostras, cinco épocas, 500 passos, três avaliações
  de texto, benchmark estatístico, PPL e hash;
- a promoção exige `--confirmar-promocao`; substituição exige também
  `--substituir` e cria backup.

## Situação

A V7.3 é a base para novas melhorias. A V6.2 permanece disponível como
referência estável e o denso permanece somente para controle. Nenhum artefato
das versões intermediárias ou do candidato BF16 deve ser reintroduzido.

<!-- comparacao-v73:inicio -->
- V6.2: mediana de 311,01 tok/s; modelo puro 21/24, greedy restrito 24/24 e sistema 24/24.
- V7.3: mediana de 998,26 tok/s; modelo puro 9/24, greedy restrito 24/24 e sistema 24/24.
- Denso: mediana de 505,69 tok/s; modelo puro 22/24, greedy restrito 24/24 e sistema 24/24.
- benchmark: 2 aquecimentos e 5 repetições, com mediana, média e desvio-padrão registrados no JSON.
- as regras do decodificador não são contabilizadas como capacidade aprendida pelo modelo.
<!-- comparacao-v73:fim -->
