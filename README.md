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
- geração controlada `100%`.

Ele é carregado com `strict=True` nos testes e não deve ser sobrescrito.

## Resultados

Protocolo: três sementes, cinco épocas e quinze checkpoints V6.

| Métrica média | V6 |
|---|---:|
| PPL inédita | 1,9196 |
| Acurácia de token | 77,41% |
| Localização da resposta | 100% |
| Recuperação causal | 100% |
| Geração em 72 prompts | 100% |
| PPL com FFN removida | 4,9899 |

A ablação confirma que a FFN participa efetivamente da previsão.

## Desempenho

O caminho integrado atual usa `torch.sparse.mm`:

| Entrada | Média | Mínimo | Máximo |
|---|---:|---:|---:|
| 73 tokens, lote 64 | 0,897 M tokens/s | 0,873 M | 0,913 M |
| 512 tokens, lote 16 | 1,068 M tokens/s | 1,062 M | 1,072 M |

Os valores são médias de três sementes registradas em `v6_ultimo.json`. A
qualidade e a compactação estão aprovadas. O próximo passo de otimização é um
kernel CUDA específico para estados contínuos, fundindo COO, Top-12, residual e
LayerNorm.

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

A suíte atual possui nove testes automatizados.

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
- `testes/test_modelo_v6.py`: causalidade, topologia, ablação e gradientes;
- `testes/test_checkpoint_v6.py`: contrato e recarga do checkpoint canônico.
- `testes/test_documentacao_v6.py`: confere métricas documentadas contra o
  relatório canônico.

## Estrutura

```text
COMPARACAO_ATENCAO_DENSA_ESPARSA/
  modelos/
    v6_base.pt
  resultados/
    v6_20260726_142346/
    v6_ultimo.json
  src/
    modelo_v6.py
    atencao_causal_texto.py
  testes/
    test_modelo_v6.py
    test_checkpoint_v6.py
  DOCUMENTO_MODELO_V6.md
  executar_v6.py
  treinar_v6.py
```

## Limitações

- o corpus atual é controlado e possui vocabulário de 49 tokens;
- o resultado ainda não comprova linguagem natural aberta;
- o kernel CUDA isolado da FFN discreta não aceita diretamente os estados
  contínuos usados pela V6;
- qualquer otimização futura deve preservar exatamente a saída PyTorch antes de
  substituir o backend oficial.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
