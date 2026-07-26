# V6 - Atenção Causal com FFN Esparsa

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

A V6 é o modelo-base oficial do projeto. Ela pesquisa uma arquitetura de
linguagem compacta que preserve causalidade, recuperação de contexto e
transformações não lineares usando atenção e FFN estruturalmente esparsas.

O checkpoint canônico está em:

`modelos/v6_base.pt`

## Arquitetura

```text
tokens
  -> embedding e janela local
  -> atenção causal com Q/K esparsos e Top-24
  -> residual + LayerNorm
  -> FFN COO 48 -> 144 -> 48, ReLU + Top-12
  -> residual com gate aprendível + LayerNorm
  -> classificador de tokens
```

### Atenção

- descritores causais construídos somente com o passado;
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

O checkpoint `modelos/v6_base.pt` corresponde à melhor execução:

- semente `20260727`;
- época `5`;
- 9.621 parâmetros;
- PPL inédita de teste `1,9077`;
- localização da resposta `100%`;
- recuperação causal `100%`;
- acurácia do primeiro token do local, em avaliação controlada, `100%`.

Ele é carregado com `strict=True` nos testes e não deve ser sobrescrito.

## Resultados

Protocolo: três sementes, cinco épocas e quinze checkpoints V6.

| Métrica média | V6 |
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

Esses números são de uma execução do checkpoint canônico e não substituem a
média histórica de três sementes. Eles tornam explícito o custo que antes ficava
fora do cronômetro.

## Teste isolado de roteamento

`testar_v6_isolado.py` preserva `modelos/v6_base.pt` e avalia duas dúvidas:
roteamento sem mapas semânticos manuais e geração autorregressiva até `EOS`.
O roteador experimental usa apenas um codebook esparso treinável, Top-12 e
supervisão da posição de origem do fato. Portanto, ele aprende o roteamento, mas
o teste ainda é supervisionado e não comprova descoberta semântica autônoma.

Resultados nas combinações inéditas:

| Configuração | PPL | Acurácia de token | Local correto | Recuperação |
|---|---:|---:|---:|---:|
| V6 canônica, descritores manuais | 1,9077 | 77,35% | 100% | 100% |
| Roteador aprendido, leitor original | 2,6347 | 72,40% | 87,87% | 99,95% |
| Roteador e leitor Q/K adaptado | 3,6330 | 63,93% | 100% | 100% |

O roteador chegou a `100%` de Top-1 sem mapas manuais, mas a adaptação não
preservou PPL nem qualidade global. Na geração livre da própria V6 canônica, as
72 sequências tiveram `0,00%` de correspondência exata, `61,11%` de término em
`EOS`, `41,07%` de tokens alinhados e `42,13%` de locais alinhados. A variante
experimental não foi promovida; a V6 canônica permanece como modelo-base.

O relatório reproduzível está em
`resultados/teste_isolado_v6_ultimo.json`.

## Dependências

- Windows;
- Python 3.12;
- PyTorch com CUDA;
- GPU NVIDIA compatível.

O ambiente utilizado já está disponível em:

`C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda`

Nenhuma dependência adicional precisa ser instalada.

## Execução

Avaliar o checkpoint oficial:

```powershell
C:\Users\USER\Downloads\MeuProjetoIA\venv_cuda\Scripts\python.exe `
  executar_v6.py
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

A suíte atual possui quinze testes automatizados.

## Exemplo de uso

```python
from pathlib import Path

import torch

from executar_v6 import carregar_v6

dispositivo = torch.device("cuda")
modelo, codigo, metadados = carregar_v6(
    Path("modelos/v6_base.pt"),
    dispositivo,
)
print(modelo.auditoria())
```

## Módulos principais

- `src/modelo_v6.py`: arquitetura oficial, FFN COO e auditoria estrutural;
- `src/atencao_causal_texto.py`: corpus, descritores e atenção causal;
- `src/camada_linear_esparsa.py`: primitiva COO usada pela FFN;
- `executar_v6.py`: carga estrita e avaliação do checkpoint oficial;
- `treinar_v6.py`: treino oficial em três sementes e cinco épocas;
- `testar_v6_isolado.py`: auditoria de pipeline, geração livre e roteador
  esparso aprendido sem alterar o checkpoint oficial;
- `testes/test_modelo_v6.py`: causalidade, topologia, ablação e gradientes;
- `testes/test_checkpoint_v6.py`: contrato e recarga do checkpoint canônico.
- `testes/test_documentacao_v6.py`: confere métricas documentadas contra o
  relatório canônico.
- `testes/test_v6_isolado.py`: contratos causais e de esparsidade do experimento.

## Estrutura

```text
COMPARACAO_ATENCAO_DENSA_ESPARSA/
  modelos/
    v6_base.pt
  resultados/
    teste_isolado_v6_20260726_150927/
    teste_isolado_v6_ultimo.json
    v6_20260726_142346/
    v6_ultimo.json
  src/
    modelo_v6.py
    atencao_causal_texto.py
  testes/
    test_modelo_v6.py
    test_checkpoint_v6.py
    test_documentacao_v6.py
    test_v6_isolado.py
  DOCUMENTO_MODELO_V6.md
  executar_v6.py
  testar_v6_isolado.py
  treinar_v6.py
```

## Limitações

- o corpus atual é controlado e possui vocabulário de 49 tokens;
- o resultado ainda não comprova linguagem natural aberta;
- o roteador aprendido usa o rótulo da posição correta do fato durante o treino;
- a V6 ainda não gera sequências livres confiáveis, apesar do acerto controlado
  do primeiro token;
- a seleção atual compara pares de códigos e cresce quadraticamente com a
  sequência e a largura do descritor;
- o kernel CUDA isolado da FFN discreta não aceita diretamente os estados
  contínuos usados pela V6;
- qualquer otimização futura deve preservar exatamente a saída PyTorch antes de
  substituir o backend oficial.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
