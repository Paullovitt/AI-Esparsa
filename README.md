# AI-Esparsa — Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

O projeto estuda geração causal de relatos longos com um núcleo neural
estruturalmente esparso. O pedido informa pessoa, ajudante, tarefa, objeto,
local e problema; o texto precisa preservar esses campos durante 24 frases.

O repositório mantém somente três modelos:

1. **V6.2**, base estável e checkpoint oficial;
2. **V7.3**, base de desenvolvimento com roteamento combinatório e kernel
   CUDA fundido;
3. **Denso**, controle experimental de orçamento de parâmetros equivalente.

Versões intermediárias e o candidato V7.3 BF16 foram removidos. A V7.3 atual
usa pesos FP32. Ela passou em 24/24 como sistema controlado, mas em 9/24 no
argmax do modelo puro; as duas métricas são intencionalmente separadas.

## Checkpoints

| Modelo | Caminho | Situação |
|---|---|---|
| V6.2 | `modelos/gerador_esparso_base.pt` | Base estável oficial |
| V7.3 | `modelos/gerador_esparso_v73_base.pt` | Base de desenvolvimento |
| Denso | `resultados/comparacao_esparso_denso_50k/denso/epoca_05.pt` | Controle |

O SHA-256 da V6.2 permanece:

```text
daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f
```

Todos os carregamentos usam `torch.load(..., weights_only=True)` e
`load_state_dict(..., strict=True)`.

## Arquitetura

### V6.2 estável

```text
tokens
  → embedding compartilhado
  → 3 blocos causais
      → Q/K COO 128×128, fan-in 32
      → atenção causal Top-32 em blocos
      → residual + LayerNorm
      → FFN COO 128→384→128, GELU e Top-64
      → residual + LayerNorm
  → classificador amarrado ao embedding
```

A inferência V6.2 adiciona cache causal prealocado, matrizes CSR reutilizadas
e cache de gates. O runtime não acrescenta parâmetros.

Métricas oficiais sincronizadas:

<!-- metricas-desempenho:inicio -->
| Medição | Resultado |
|---|---:|
| Forward paralelo, lote 16 × contexto 640 | 145.417,74 tokens/s |
| Pico de VRAM no forward | 83,42 MiB |
| Geração autorregressiva real | 313,24 tokens/s |
| Latência até o primeiro token | 4,63 ms |
| Tempo do relato completo | 1,56 s |
<!-- metricas-desempenho:fim -->

### V7.3 — base de desenvolvimento

A V7.3 conserva Q/K esparsos e divide o banco FFN COO em 16 microgrupos.
Um roteador baixo-rank aprendido escolhe quatro grupos para cada token:

```text
contexto
  → roteador causal Top-4
  → quatro microgrupos ativos
  → kernel CUDA fundido
  → atenção incremental Top-32
  → saída
```

Características:

- 165.443 parâmetros FP32;
- 1.776 parâmetros no roteador;
- 1.820 combinações teóricas por camada;
- 23.040 de 92.160 conexões FFN ativas por token;
- zero parâmetros adicionais no runtime CUDA;
- somente os quatro grupos escolhidos são despachados;
- fallback PyTorch condicional quando o kernel não está disponível;
- restrição causal de concordância, que bloqueia flexões incompatíveis antes
  da escolha do token.

Os microblocos ativos ainda contêm zeros correspondentes a arestas COO
ausentes. O forward em lote materializa projeções e atenção temporárias para
reduzir lançamentos de kernel. Portanto, os parâmetros são esparsos, mas nem
toda operação temporária é fisicamente esparsa.

### Baseline densa

O controle denso possui dimensão 88, três blocos e FFN 88→160→88:

- 163.003 parâmetros;
- Q/K e FFN totalmente conectados;
- atenção causal pelo kernel otimizado do PyTorch;
- embedding/classificador amarrado;
- mesmos dados, tokenizador e contrato de geração.

A baseline não é promovida automaticamente.

## Qualidade textual

O corpus procedural já produz concordância correta para objetos principais e
distratores. O decodificador aplica três restrições causais:

- concordância de gênero;
- conclusão das seis ações válidas somente nos encaixes de ação;
- proibição de repetir o objeto principal como objeto de apoio.

Na concordância, por exemplo,
após sequências como `o relatório foi`, a flexão `levada` recebe logit
`-inf`; a escolha continua sendo feita pelo modelo entre os tokens válidos.

O validador registra:

- recuperação dos seis campos;
- completude e 24 frases;
- consistência objeto/local;
- consistência das ações;
- concordância de gênero;
- repetição de trigramas;
- vazamento de blocos de pergunta/resposta.

## Dependências

- Python 3.11 a 3.14;
- PyTorch 2.11.0;
- CUDA para treino e benchmarks completos;
- CUDA Toolkit e Visual Studio Build Tools C++ para compilar a extensão V7.3;
- `ninja`, disponibilizado pelo Build Tools no ambiente usado.

Ambiente medido: Windows 11, Python 3.14.0, PyTorch 2.11.0+cu128 e NVIDIA
GeForce RTX 3050.

## Instalação

```powershell
python -m venv venv_cuda
.\venv_cuda\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Instale previamente uma distribuição do PyTorch compatível com a GPU e o
CUDA local.

## Execução

V6.2:

```powershell
python executar_v62_cache.py
```

V7.3:

```powershell
python executar_gerador_esparso_v73.py
```

Fallback da V7.3 sem extensão CUDA:

```powershell
python executar_gerador_esparso_v73.py --permitir-fallback
```

Prompt próprio:

```powershell
python executar_gerador_esparso_v73.py --prompt "Pedido: escreva um relato sobre bruno, com ajuda de diego, para organizar uma pequena mostra cultural. Inclua o documento no corredor e o problema um atraso no transporte. Texto:"
```

O formato é fechado e palavras fora do vocabulário são rejeitadas antes da
inferência.

## Treinamento

Treino oficial da V6.2:

```powershell
python treinar_gerador_esparso.py
```

Novo treino controlado V7.3 FP32:

```powershell
python treinar_gerador_esparso_v73.py
```

O treinador V7.3 exige:

- 50.000 relatos;
- cinco épocas;
- 500 passos por época;
- lote 100;
- seed 20260728;
- um checkpoint separado por época;
- diretório novo, sem sobrescrever as bases.

O checkpoint-base V7.3 atual possui duas épocas históricas e, embora seja a
base de desenvolvimento escolhida, não é elegível para promoção formal até
um treino controlado de cinco épocas ser aprovado.

### Ciclo de candidato V7.3

O treino, a validação e a promoção são etapas separadas:

```powershell
python treinar_gerador_esparso_v73.py

python comparar_v73.py `
  --checkpoint-v73 resultados/v73_base/treino_fp32_5epocas/epoca_05.pt `
  --metricas-treino-v73 resultados/v73_base/treino_fp32_5epocas/relatorio_treino.json `
  --resultados resultados/v73_base/treino_fp32_5epocas/validacao `
  --repeticoes 5 `
  --aquecimentos 2 `
  --nao-sincronizar-documentacao

python promover_gerador_esparso.py `
  --arquitetura v73 `
  --origem resultados/v73_base/treino_fp32_5epocas/epoca_05.pt `
  --relatorio resultados/v73_base/treino_fp32_5epocas/validacao/comparacao.json
```

O último comando apenas valida. Depois da revisão humana, a cópia é confirmada
com `--confirmar-promocao --substituir`. A base anterior é preservada em
`.backup` e a troca usa arquivo temporário e `os.replace`.

## Comparação

```powershell
python comparar_v73.py --repeticoes 5 --aquecimentos 2
python comparar_v73.py --somente-documentos
```

<!-- comparacao-v73:inicio -->
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

Protocolo: 2 aquecimentos, 5 repetições; mediana como medida principal e média ± desvio-padrão para dispersão.

Aprovação do modelo puro, do greedy restrito e do sistema completo são métricas separadas. As intervenções do decodificador também são contabilizadas.

Na mediana, a V7.3 atingiu 1.97× a geração densa; o denso atingiu 6.21× o forward da V7.3.
<!-- comparacao-v73:fim -->

Resultados consolidados:

- `resultados/v73_base/comparacao.json`;
- `resultados/v73_base/RELATORIO_V73.md`;
- `resultados/v73_base/EXEMPLOS_COMPARATIVOS.md`;
- `resultados/v73_base/treino_base.json`.

## Principais módulos

- `src/modelo_gerador_esparso.py`: arquitetura esparsa estável;
- `src/modelo_gerador_esparso_v62.py`: cache causal e CSR da V6.2;
- `src/roteamento_combinatorio_v73.py`: roteador e banco COO da V7.3;
- `src/runtime_condicional_v73.py`: fallback condicional PyTorch;
- `src/treino_vetorizado_v73.py`: executor de treino FP32;
- `src/modelo_gerador_esparso_v73.py`: runtime CUDA V7.3;
- `src/kernel_cuda_v73.py`: compilação e carregamento da extensão;
- `src/kernels_v73/`: binding C++ e kernels CUDA;
- `src/modelo_gerador_denso.py`: baseline densa;
- `src/decodificador_gerador.py`: geração e restrições causais;
- `src/dados_gerador.py`: corpus determinístico, codificação e lotes;
- `src/avaliacao_linguagem.py`: loss, PPL e acurácia de token;
- `src/validacao_gerador.py`: contrato de texto e avaliação de 24 relatos;
- `src/benchmark_gerador.py`: medições básicas de forward e geração;
- `src/avaliacao_decodificacao.py`: modelo puro versus sistema controlado;
- `src/benchmark_estatistico.py`: aquecimento, repetições e dispersão;
- `src/relatorio_v73.py`: Markdown e sincronização automática pelo JSON;
- `src/relatorio_esparso_denso.py`: contrato e relatório da baseline;
- `treinar_gerador_esparso.py`: corpus, validação e treino V6.2;
- `treinar_gerador_esparso_v73.py`: treino V7.3 FP32;
- `comparar_v73.py`: orquestra a comparação final dos três modelos;
- `promover_gerador_esparso.py`: valida e promove V6.2 ou V7.3.

## Estrutura

```text
AI-Esparsa/
├── modelos/
│   ├── gerador_esparso_base.pt
│   └── gerador_esparso_v73_base.pt
├── resultados/
│   ├── gerador_esparso_base_50k/
│   ├── v62_base_runtime/
│   ├── comparacao_esparso_denso_50k/
│   └── v73_base/
├── src/
│   ├── modelo_gerador_esparso.py
│   ├── modelo_gerador_esparso_v62.py
│   ├── modelo_gerador_esparso_v73.py
│   ├── roteamento_combinatorio_v73.py
│   ├── runtime_condicional_v73.py
│   ├── treino_vetorizado_v73.py
│   ├── modelo_gerador_denso.py
│   ├── dados_gerador.py
│   ├── avaliacao_linguagem.py
│   ├── validacao_gerador.py
│   ├── benchmark_gerador.py
│   ├── avaliacao_decodificacao.py
│   ├── benchmark_estatistico.py
│   ├── relatorio_v73.py
│   ├── relatorio_esparso_denso.py
│   ├── kernel_cuda_v73.py
│   └── kernels_v73/
├── testes/
├── executar_v62_cache.py
├── executar_gerador_esparso_v73.py
├── treinar_gerador_esparso_v73.py
└── comparar_v73.py
```

## Testes

```powershell
python -m unittest discover -s testes -v
```

A suíte verifica causalidade, equivalência numérica, cache, roteamento,
checkpoint seguro, kernel CUDA, concordância, avaliação pura/restrita,
estatísticas de benchmark, promoção V7.3, documentação e inventário dos três
modelos.

## Limitações

- o domínio é procedural fechado, não português aberto;
- o kernel V7.3 é especializado nas dimensões atuais;
- a primeira execução pode compilar a extensão local;
- o forward em lote e o treino usam temporários densos;
- o checkpoint V7.3 atual tem duas épocas;
- a V7.3 atual aprovou apenas 9/24 no modelo puro, principalmente por
  concordância de gênero; os 24/24 dependem do decodificador controlado;
- PPL e aprovação neste corpus não provam generalização ampla.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
