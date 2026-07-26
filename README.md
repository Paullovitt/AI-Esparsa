# V6.1 Base - Atenção Causal Temporal com FFN Esparsa

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

A V6.1 é o modelo-base oficial do projeto. Ela pesquisa uma arquitetura de
linguagem compacta que preserve causalidade, ordem temporal, recuperação de
contexto e transformações não lineares usando atenção e FFN estruturalmente
esparsas.

O checkpoint canônico está em:

`modelos/v61_base.pt`

O checkpoint anterior foi preservado somente como rollback:

`modelos/v6_rollback.pt`

## Arquitetura

```text
tokens
  -> embedding + posição senoidal fixa + janela local
  -> descritor com slot e papel temporal esparsos
  -> atenção causal com Q/K esparsos e Top-24
  -> residual + LayerNorm
  -> FFN COO 48 -> 144 -> 48, ReLU + Top-12
  -> residual com gate aprendível + LayerNorm
  -> classificador de tokens
```

### Atenção

- descritores causais construídos somente com o passado;
- três códigos de slot e dois códigos de papel temporal;
- Q e K por códigos esparsos de conteúdo;
- seleção Top-24;
- recuperação do fato causal sem acessar tokens futuros;
- cache e backend CUDA validados separadamente.

### FFN

- duas projeções COO reais;
- 3.456 conexões;
- densidade estrutural de 25%;
- ReLU e roteamento Top-12;
- nenhuma `nn.Linear` ou matriz densa mascarada dentro da FFN;
- residual e normalização após a transformação.

O embedding, as normalizações e o classificador continuam densos. Portanto, a
FFN e as projeções Q/K são esparsas, mas o modelo completo não é anunciado como
100% composto apenas por operações esparsas.

## Checkpoint oficial

O checkpoint `modelos/v61_base.pt` corresponde à execução promovida:

- semente `20260727`;
- época `5`;
- 9.632 parâmetros;
- PPL de teste `1,5308`;
- PPL média robusta `1,5317`;
- geração livre exata média `93,06%`;
- tokens livres alinhados `99,17%`;
- localização da resposta `100%`;
- recuperação causal `100%`;
- locais livres `99,54%`.

Ele é carregado com `strict=True` nos testes e não deve ser sobrescrito.

## Resultados do rollback

O rollback V6 foi validado com três sementes, cinco épocas e quinze checkpoints.

| Métrica média | V6 rollback |
|---|---:|
| PPL inédita | 1,9196 |
| Acurácia de token | 77,41% |
| Localização da resposta | 100% |
| Recuperação causal | 100% |
| Primeiro token do local em 72 prompts | 100% |
| PPL com FFN removida | 4,9899 |

A ablação confirma que a FFN participa efetivamente da previsão. O valor de
100% não representa geração livre: ele mede somente o primeiro token do local
correto nos 72 prompts controlados.

## Desempenho

O caminho integrado atual usa `torch.sparse.mm`. A tabela histórica abaixo mede
somente o `forward`, com descritores e candidatos preparados antes do cronômetro:

| Entrada | Média | Mínimo | Máximo |
|---|---:|---:|---:|
| 73 tokens, lote 64 | 0,897 M tokens/s | 0,873 M | 0,913 M |
| 512 tokens, lote 16 | 1,068 M tokens/s | 1,062 M | 1,072 M |

Os valores são médias de três sementes registradas em `v6_ultimo.json`.

O teste isolado também mediu, na mesma execução, o `forward` e o pipeline desde
os tokens, incluindo descritores e seleção de candidatos:

| Entrada | Forward isolado | Pipeline completo |
|---|---:|---:|
| 73 tokens, lote 64 | 0,955 M tokens/s | 0,555 M tokens/s |
| 512 tokens, lote 16 | 0,984 M tokens/s | 0,296 M tokens/s |

Esses números são de uma execução do checkpoint de rollback e não substituem a
média histórica de três sementes. Eles tornam explícito o custo que antes ficava
fora do cronômetro.

## Teste isolado de roteamento

`testar_v6_isolado.py` usa `modelos/v6_rollback.pt` e avalia duas dúvidas:
roteamento sem mapas semânticos manuais e geração autorregressiva até `EOS`.
O roteador experimental usa apenas um codebook esparso treinável, Top-12 e
supervisão da posição de origem do fato. Portanto, ele aprende o roteamento, mas
o teste ainda é supervisionado e não comprova descoberta semântica autônoma.

Resultados nas combinações inéditas:

| Configuração | PPL | Acurácia de token | Local correto | Recuperação |
|---|---:|---:|---:|---:|
| V6 rollback, descritores manuais | 1,9077 | 77,35% | 100% | 100% |
| Roteador aprendido, leitor original | 2,6347 | 72,40% | 87,87% | 99,95% |
| Roteador e leitor Q/K adaptado | 3,6330 | 63,93% | 100% | 100% |

O roteador chegou a `100%` de Top-1 sem mapas manuais, mas a adaptação não
preservou PPL nem qualidade global. Na geração livre da própria V6 rollback, as
72 sequências tiveram `0,00%` de correspondência exata, `61,11%` de término em
`EOS`, `41,07%` de tokens alinhados e `42,13%` de locais alinhados. A variante
experimental não foi promovida. Esse resultado motivou a evolução temporal V6.1.

O relatório reproduzível está em
`resultados/teste_isolado_v6_ultimo.json`.

## V6.1 Base

A V6.1 corrige a ausência de ordem temporal encontrada na geração livre sem
trocar a atenção ou a FFN esparsas. Ela acrescenta:

- posição senoidal fixa, sem matriz de pesos;
- um gate posicional escalar;
- três códigos esparsos de slot e dois de papel, objeto e entidade;
- reutilização dos dois campos de papel existentes, mantendo o descritor com
  largura 6;
- perda reforçada apenas nos tokens que dependem da ordem dos fatos.

Ela possui 9.632 parâmetros, somente 11 a mais que o rollback V6. Q/K continuam sem
projeções densas, a FFN continua COO e nenhuma `nn.Linear` foi adicionada à
atenção ou à FFN.

A validação robusta usou quatro sementes de dados inéditos, 72 gerações por
semente e três rodadas de benchmark:

| Métrica | V6 rollback | V6.1 Base |
|---|---:|---:|
| PPL | 1,9077 | 1,5317 média |
| Acurácia de token | 77,35% | 88,06% média |
| Geração livre exata | 0,00% | 93,06% média |
| Tokens livres alinhados | 41,07% | 99,17% média |
| Localização e recuperação controladas | 100% | 100% |
| Locais livres | 42,13% | 99,54% média |

| Pipeline completo | V6 | V6.1 | Razão V6.1/V6 |
|---|---:|---:|---:|
| 73 tokens, lote 64 | 0,515 M tokens/s | 0,536 M tokens/s | 104,05% |
| 512 tokens, lote 16 | 0,280 M tokens/s | 0,279 M tokens/s | 99,72% |

A VRAM temporária média permaneceu igual nas medições pareadas: 71,22 MiB no
caso curto e 392,68 MiB no longo. Depois de armazenar em cache os códigos
temporais fixos, a V6.1 ficou 4,05% mais rápida no caso curto e
praticamente empatada no longo. Essa diferença pequena deve ser tratada como
equivalência de desempenho, não como aceleração garantida.

O ganho é válido para o corpus controlado. Os códigos de slot e papel usam sua
estrutura fixa; portanto, o resultado ainda não comprova textos livres com
quantidade variável de fatos.

## Dependências

- Windows;
- Python 3.12;
- PyTorch com CUDA;
- GPU NVIDIA compatível.

O ambiente utilizado já está disponível em:

`C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda`

Nenhuma dependência adicional precisa ser instalada.

## Execução

Avaliar o rollback V6:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  executar_v6.py
```

Avaliar o modelo-base V6.1:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  executar_v61.py
```

Repetir sua validação robusta:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  validar_v61_base.py
```

Treinar novamente a V6.1 por cinco épocas:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  testar_v61_posicional.py
```

Treinar novamente as três sementes por cinco épocas:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  treinar_v6.py
```

Executar todos os testes:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  -m unittest discover -s testes -p "test_*.py" -v
```

Executar novamente o experimento isolado de cinco épocas:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  testar_v6_isolado.py
```

A suíte atual possui vinte e três testes automatizados.

## Exemplo de uso

```python
from pathlib import Path

import torch

from executar_v61 import carregar_v61

dispositivo = torch.device("cuda")
modelo, codigo_temporal, metadados = carregar_v61(
    Path("modelos/v61_base.pt"),
    dispositivo,
)
print(modelo.auditoria())
```

## Módulos principais

- `src/modelo_v6.py`: arquitetura anterior preservada como rollback;
- `src/modelo_v61.py`: modelo-base, posição fixa e roteamento temporal esparso;
- `src/atencao_causal_texto.py`: corpus, descritores e atenção causal;
- `src/camada_linear_esparsa.py`: primitiva COO usada pela FFN;
- `executar_v6.py`: carga estrita e avaliação do rollback V6;
- `executar_v61.py`: carga estrita e avaliação da V6.1 Base;
- `treinar_v6.py`: treino histórico do rollback em três sementes;
- `testar_v61_posicional.py`: treino da V6.1 por cinco épocas;
- `validar_v61_base.py`: quatro sementes e três rodadas de benchmark;
- `testar_v6_isolado.py`: auditoria de pipeline, geração livre e roteador
  esparso aprendido sem alterar o checkpoint de rollback;
- `testes/test_modelo_v6.py`: causalidade, topologia, ablação e gradientes;
- `testes/test_checkpoint_v6.py`: contrato e recarga do rollback;
- `testes/test_checkpoint_v61.py`: contrato e recarga da V6.1 Base;
- `testes/test_documentacao_v6.py`: confere métricas documentadas contra o
  relatório canônico;
- `testes/test_v6_isolado.py`: contratos causais e de esparsidade do experimento;
- `testes/test_v61_posicional.py`: slots, papéis, causalidade e esparsidade V6.1.

## Estrutura

```text
COMPARACAO_ATENCAO_DENSA_ESPARSA/
  modelos/
    v61_base.pt
    v6_rollback.pt
  resultados/
    teste_isolado_v6_20260726_150927/
    teste_isolado_v6_ultimo.json
    teste_v61_posicional_20260726_154023/
    teste_v61_posicional_ultimo.json
    v61_base_validacao.json
    v6_20260726_142346/
    v6_ultimo.json
  src/
    modelo_v6.py
    modelo_v61.py
    atencao_causal_texto.py
  testes/
    test_modelo_v6.py
    test_checkpoint_v6.py
    test_checkpoint_v61.py
    test_documentacao_v6.py
    test_v6_isolado.py
    test_v61_posicional.py
  DOCUMENTO_MODELO_V61.md
  executar_v6.py
  executar_v61.py
  testar_v6_isolado.py
  testar_v61_posicional.py
  treinar_v6.py
  validar_v61_base.py
```

## Limitações

- o corpus atual é controlado e possui vocabulário de 49 tokens;
- o resultado ainda não comprova linguagem natural aberta;
- o roteador aprendido usa o rótulo da posição correta do fato durante o treino;
- a V6 ainda não gera sequências livres confiáveis, apesar do acerto controlado
  do primeiro token;
- a seleção atual compara pares de códigos e cresce quadraticamente com a
  sequência e a largura do descritor;
- a V6.1 depende de slots e papéis definidos pela estrutura fixa do corpus;
- a diferença de velocidade entre V6 e V6.1 está próxima do ruído de medição;
- o kernel CUDA isolado da FFN discreta não aceita diretamente os estados
  contínuos usados pela V6;
- qualquer otimização futura deve preservar exatamente a saída PyTorch antes de
  substituir o backend oficial.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
