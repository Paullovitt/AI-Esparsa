# Documento técnico — AI-Esparsa

Autor: Paulo Augusto
Ano: 2026

## Escopo vigente

O repositório mantém somente:

- V6.2, base estável oficial;
- V7.3, base de desenvolvimento ativa;
- baseline densa, controle experimental.

Versões intermediárias e o candidato treinado em BF16 foram descartados.

Checkpoints:

```text
modelos/gerador_esparso_base.pt
modelos/gerador_esparso_v73_base.pt
resultados/comparacao_esparso_denso_50k/denso/epoca_05.pt
```

O checkpoint V6.2 não pode ser sobrescrito. A V7.3 atual é FP32, possui duas
épocas históricas e não é elegível para promoção formal. Um futuro candidato
promovível precisa completar cinco épocas controladas.

## Dados e contrato

O corpus procedural é determinístico:

- treino: 50.000 relatos;
- validação: 1.000;
- teste: 1.000;
- 24 frases por relato;
- contexto máximo: 640 tokens;
- seed: 20260728.

Formato público:

```text
Pedido: escreva um relato sobre <pessoa>, com ajuda de <ajudante>, para <tarefa>. Inclua <objeto> no <local> e o problema <problema>. Texto:
```

Campos ausentes, ordem diferente ou tokens desconhecidos são rejeitados.

## V6.2 — base estável

A V6.2 tem dimensão 128, três blocos e 163.667 parâmetros. Cada bloco usa:

1. Q/K COO com fan-in 32;
2. atenção causal Top-32;
3. residual e LayerNorm;
4. FFN COO 128→384→128;
5. GELU e Top-64;
6. residual com gate e LayerNorm.

O embedding/classificador é compartilhado e denso. Normalização, estados
residuais e escores temporários também são densos. Portanto, o núcleo é
estruturalmente esparso, não 100% esparso.

O runtime V6.2 acrescenta, sem novos parâmetros:

- cache causal prealocado;
- projeções lineares convertidas uma vez para CSR em inferência;
- reutilização dos gates;
- reconstrução segura ao exceder a janela de 640 tokens.

Métricas oficiais:

<!-- metricas-desempenho:inicio -->
- forward paralelo, lote 16 e contexto 640: 145.417,74 tokens/s,
  70,42 ms e pico de 83,42 MiB;
- geração autorregressiva de um relato: 313,24 tokens/s;
- latência até o primeiro token: 4,63 ms;
- tempo total para 489 tokens e 2.840 caracteres: 1,56 s.
<!-- metricas-desempenho:fim -->

SHA-256:

```text
daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f
```

## V7.3 — base de desenvolvimento

### Roteamento

A V7.3 preserva Q/K COO e substitui a FFN fixa por um banco compartilhado de
16 microgrupos. Um roteador causal baixo-rank escolhe quatro grupos por token.

- parâmetros totais: 165.443;
- parâmetros do roteador: 1.776;
- conexões FFN totais: 92.160;
- conexões FFN ativas por token: 23.040;
- combinações teóricas por camada: 1.820.

O roteador usa somente estados causais. Grupos inativos não são despachados na
geração. Os pesos são compartilhados entre combinações, atendendo à proposta
de sub-redes dependentes do contexto.

### Runtime CUDA

A extensão C++/CUDA funde:

1. embedding e posição;
2. normalização e Q/K;
3. atenção incremental Top-32;
4. roteamento Top-4;
5. FFN, GELU, Top-64 e residual;
6. normalização final e projeção de saída.

O caminho incremental usa 11 lançamentos de kernel por token. A extensão não
entra no checkpoint e adiciona zero parâmetros.

O kernel é especializado em dimensão 128, 16 grupos, Top-4, microgrupos de 24,
Q/K fan-in 32, contexto 640 e vocabulário atual. Sem toolchain CUDA, o fallback
PyTorch executa o mesmo checkpoint.

Os quatro microgrupos ativos ainda contêm multiplicações por zeros COO dentro
dos blocos locais. No forward em lote, Q/K e atenção são materializados em
temporários densos para reduzir overhead. Essas limitações são consideradas
nas conclusões de desempenho.

### Checkpoint-base

`modelos/gerador_esparso_v73_base.pt` contém:

- pesos FP32;
- duas épocas e 1.000 passos históricos;
- 50.000 exemplos;
- seed 20260728;
- `treinado_em_bf16 = false`;
- `elegivel_promocao = false`.

O runtime e o fallback têm o mesmo `state_dict`.

## Qualidade textual

O corpus flexiona o objeto de apoio de acordo com seu artigo. Além disso, o
decodificador bloqueia causalmente flexões impossíveis, fragmentos de ações e
a repetição do objeto principal na frase destinada ao objeto de apoio.

Exemplo:

```text
prefixo: o relatorio foi
token bloqueado: levada
token válido: levado
```

Não existe correção posterior por substituição de string. O logit incompatível
é removido antes de `argmax` ou amostragem.

As seis ações procedurais são restringidas somente após marcadores como
`o primeiro passo foi`, `começou a`, `decidiram` e `ficou responsável por`.
Isso evita fragmentos como `mover as caixas para uma lista` ou
`ficou responsável por categoria` sem interferir na tarefa contida no prompt.

Na frase `em outra parte do ambiente`, o objeto extraído do pedido é
temporariamente bloqueado. Assim, a frase de apoio não pode mover o próprio
objeto principal para outro local e criar uma contradição.

O validador passou a reprovar:

- `o relatorio foi levada`;
- `a camera foi levado`;
- flexões incompatíveis após `ele` e `ela`;
- contradições objeto/local;
- ações diferentes da tarefa;
- campos ausentes;
- repetição excessiva;
- texto incompleto.

Cada relatório registra a concordância individual das 24 gerações.

## Baseline densa

A baseline usa dimensão 88, três blocos, FFN 88→160→88, atenção causal densa
e 163.003 parâmetros. Residual, LayerNorm, posição, tokenizador,
embedding/saída amarrados e protocolo de treino são equivalentes.

Ela é controle científico e não pode ser promovida sem autorização.

## Treinamento

V6.2:

- cinco épocas;
- 500 passos por época;
- 2.500 atualizações;
- checkpoint separado por época;
- AdamW e clipping de gradiente.

Um novo treino V7.3 usa FP32 e exige o mesmo protocolo de cinco épocas. O
executor de treino vetoriza microgrupos ativos e a matriz de atenção, sem
criar novos Parameters densos persistentes.

Nenhum treinador escreve diretamente em `modelos/`.

O checkpoint de cada época usa o identificador
`gerador-esparso-v73-treino-fp32`. O loader aceita esse identificador e o da
base histórica. Somente a época 5 do protocolo completo é marcada como
candidata; isso não promove o artefato automaticamente.

### Separação de responsabilidades

O antigo treinador monolítico foi dividido sem alterar sua API pública:

- `src/dados_gerador.py`: corpus, codificação e lotes;
- `src/avaliacao_linguagem.py`: loss, PPL e acurácia;
- `src/validacao_gerador.py`: regras e métricas textuais;
- `src/benchmark_gerador.py`: medições básicas;
- `src/benchmark_estatistico.py`: repetições e dispersão;
- `src/relatorio_v73.py`: relatórios e sincronização documental;
- `src/relatorio_esparso_denso.py`: comparação histórica da baseline.

`treinar_gerador_esparso.py` permanece como orquestrador e reexporta os nomes
anteriores para não quebrar scripts e testes existentes.

## Comparação obrigatória

<!-- comparacao-v73:inicio -->
O relatório separa três níveis de qualidade:

- argmax do modelo puro, sem restrições de domínio;
- argmax com restrições causais;
- sistema completo, incluindo retentativas.

Também registra candidatos bloqueados, mudanças do argmax, amostras brutas, mediana, média, desvio-padrão, clocks e energia da GPU quando `nvidia-smi` disponibiliza esses dados.

| Métrica | V6.2 | V7.3 | Denso |
|---|---:|---:|---:|
| Época | 5 | 2 | 5 |
| Parâmetros | 163.667 | 165.443 | 163.003 |
| PPL teste | 1,050526 | 1,054314 | 1,049220 |
| Aprovação — modelo puro | 21/24 | 9/24 | 22/24 |
| Modelo puro — objeto/local | 88% | 83% | 92% |
| Modelo puro — ações | 100% | 88% | 100% |
| Modelo puro — concordância | 100% | 42% | 100% |
| Aprovação — greedy restrito | 24/24 | 24/24 | 24/24 |
| Aprovação — sistema completo | 24/24 | 24/24 | 24/24 |
| Mudanças de argmax pelas regras | 3 | 24 | 3 |
| Tokens bloqueados pelas regras | 179.733 | 177.103 | 168.888 |
| Forward — mediana, média ± DP | 143.184,54 tok/s (média 139.095,63 ± 6.994,69) | 273.029,23 tok/s (média 271.942,27 ± 2.175,18) | 1.695.953,89 tok/s (média 1.702.406,96 ± 16.872,68) |
| Geração — mediana, média ± DP | 311,01 tok/s (média 309,32 ± 12,45) | 998,26 tok/s (média 973,31 ± 54,48) | 505,69 tok/s (média 504,57 ± 3,19) |
| Primeiro token — mediana | 4,17 ms | 5,93 ms | 1,78 ms |
| VRAM forward — mediana | 83,42 MiB | 102,07 MiB | 46,51 MiB |
| Checkpoint | 678.219 bytes | 691.289 bytes | 674.635 bytes |
<!-- comparacao-v73:fim -->

Fonte única:

```text
resultados/v73_base/comparacao.json
```

Relatório legível e textos:

```text
resultados/v73_base/RELATORIO_V73.md
resultados/v73_base/EXEMPLOS_COMPARATIVOS.md
```

## Segurança e reprodutibilidade

- carregar somente com `weights_only=True`;
- recarregar estado com `strict=True`;
- nunca sobrescrever as duas bases;
- salvar épocas separadamente;
- fixar seed e protocolo;
- excluir compilação inicial do benchmark;
- separar forward paralelo de geração percebida;
- provar equivalência de kernels;
- manter as 24 saídas no JSON;
- tratar o JSON como fonte das métricas documentais.

## Critério de promoção

Um futuro checkpoint V7.3 somente pode ser promovido após:

- cinco épocas completas;
- 50.000 exemplos;
- 500 passos por época;
- PPL e acurácia reavaliadas;
- 24 saídas auditáveis do modelo puro;
- 24 saídas auditáveis do greedy restrito;
- 24/24 no sistema completo;
- concordância e intervenções do decodificador registradas separadamente;
- recuperação integral;
- testes de causalidade e equivalência;
- comparação contra V6.2 e denso;
- benchmark com no mínimo cinco repetições após aquecimento;
- PPL no máximo 5% acima do controle denso;
- SHA-256 do relatório igual ao checkpoint candidato;
- aprovação explícita do usuário.

Fluxo:

```powershell
python treinar_gerador_esparso_v73.py
python comparar_v73.py --checkpoint-v73 <epoca_05.pt> --resultados <validacao> --nao-sincronizar-documentacao
python promover_gerador_esparso.py --arquitetura v73 --origem <epoca_05.pt> --relatorio <comparacao.json>
```

O promotor primeiro valida sem copiar. A troca só ocorre com
`--confirmar-promocao`; se já houver base, `--substituir` cria um backup e faz
a troca atomicamente.

## Limite de validade

Os resultados se restringem ao corpus procedural fechado e à RTX 3050 usada.
Não demonstram domínio geral do português. Roteamento combinatório observado
não prova, isoladamente, generalização semântica. Temporários densos e kernels
especializados limitam a portabilidade das conclusões.
