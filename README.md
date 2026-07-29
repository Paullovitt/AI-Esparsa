# AI-Esparsa — V8 multimodal

Autor: Paulo Augusto

Ano: 2026

Licença: MIT

## Objetivo

O projeto pesquisa modelos pequenos com computação condicional e matrizes
estruturalmente esparsas. A V8 acrescenta visão ao backbone combinatório V7.3:
uma imagem determina um prefixo de patches e esse contexto participa do
roteamento Top-K dos microgrupos compartilhados da rede.

O repositório mantém exatamente:

| Componente | Estado | Artefato |
|---|---|---|
| V7.3 textual esparsa | base treinada preservada | `modelos/gerador_esparso_v73_base.pt` |
| V8 imagem + texto | arquitetura pronta, ainda sem treino | checkpoint será criado pelo treino |
| Controle denso textual | baseline experimental | `modelos/gerador_denso_controle.pt` |

Nenhum dataset de treinamento é distribuído ou mantido no projeto. Pastas
locais `dados/`, `datasets/` e `cache/` são ignoradas pelo Git.

## Arquitetura

### Backbone V7.3

- dimensão 128 e três blocos causais;
- Q/K com matrizes COO e fan-in 32;
- atenção causal Top-K;
- FFN COO com 384 unidades;
- roteador aprendido por token;
- 16 microgrupos, dos quais quatro ficam ativos;
- 1.820 combinações teóricas por camada;
- residual, normalização e embedding/saída com pesos amarrados;
- cache causal e kernel CUDA opcional para geração textual.

### Caminho multimodal V8

1. A imagem RGB é redimensionada para 64 × 64.
2. O encoder cria 64 patches de 8 × 8.
3. Cada patch de 192 valores é projetado para 128 dimensões por COO com
   densidade 25%; não existe `nn.Linear` no caminho visual.
4. Posição e modalidade são adicionadas aos patches.
5. Os patches formam um prefixo causal antes dos tokens de texto.
6. O backbone V7.3 processa imagem e texto no mesmo contexto.
7. Somente posições textuais produzem logits.

Com o vocabulário atual da V7.3:

| Medida estrutural | Valor |
|---|---:|
| Parâmetros V7.3 | 165.443 |
| Parâmetros visuais e de modalidade | 6.785 |
| Parâmetros V8 total | 172.228 |
| Tensores transferíveis da V7.3 | 56 |
| Tensores novos da V8 | 7 |

A arquitetura passou nos testes de causalidade, influência da imagem,
gradientes do encoder e roteador, shapes e tratamento de erros. Isso ainda não
é evidência de qualidade multimodal: loss, PPL, generalização, throughput e
VRAM somente poderão ser medidos depois do treino com o novo corpus.

## Dependências

- Python 3.11 a 3.14;
- PyTorch 2.11.0;
- Pillow 12.1.1;
- CUDA opcional para a V7.3 e recomendado para o futuro treino V8.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Formato do dataset multimodal

Use arquivos JSONL separados para treino e validação. Cada caminho de imagem
deve ser relativo a `--raiz-imagens`:

```json
{"id":"img-0001","imagem":"subpasta/foto.png","prompt":"o que aparece na imagem?","resposta":"um caderno sobre a mesa"}
```

Campos obrigatórios:

- `id`: identificador único;
- `imagem`: PNG, JPEG, WebP ou BMP local;
- `prompt`: pergunta ou instrução;
- `resposta`: texto-alvo.

O carregador rejeita caminho absoluto, travessia para fora da raiz, IDs
duplicados, campos vazios, JSON inválido e imagem ausente. Imagens são
decodificadas sob demanda; não há cache implícito nem cópia do dataset para os
checkpoints.

## Treinamento V8

O treino não baixa dados e não sobrescreve os modelos preservados. Por padrão,
transfere os 56 tensores compatíveis do backbone V7.3. Use `--do-zero` para
inicialização totalmente nova. O backend de treino materializa apenas
microblocos densos temporários dos grupos ativos, vetoriza a atenção Top-K e
as projeções Q/K, mas mantém como parâmetros somente os valores COO.

```powershell
python treinar_multimodal_v8.py `
  --manifesto-treino C:\dados\treino.jsonl `
  --manifesto-validacao C:\dados\validacao.jsonl `
  --raiz-imagens C:\dados\imagens `
  --resultados C:\experimentos\v8_run_01 `
  --epocas 5 `
  --lote 16
```

Cada época gera `epoca_01.pt`, `epoca_02.pt` e assim por diante. O relatório
salva hashes dos manifestos, mas não incorpora imagens ou textos do dataset.
Somente um ciclo de cinco épocas é elegível para futura comparação; promoção
continua dependendo de testes e autorização.

## Execução

V7.3 textual:

```powershell
python executar_gerador_esparso_v73.py `
  --prompt "ana biblioteca livro" `
  --max-tokens 128 `
  --permitir-fallback
```

V8, depois de existir um checkpoint treinado:

```powershell
python executar_multimodal_v8.py `
  --checkpoint C:\experimentos\v8_run_01\epoca_05.pt `
  --imagem C:\dados\imagens\foto.png `
  --prompt "descreva a imagem" `
  --max-tokens 128
```

## Testes

Para validar sem compilar o kernel CUDA opcional:

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
python -m unittest discover -s testes -v
```

A suíte cobre:

- equivalência das primitivas COO;
- causalidade dos modelos esparso, denso e multimodal;
- cache incremental V7.3;
- roteamento e gradientes;
- influência de imagens distintas nos logits;
- manifesto, segurança de caminhos, decodificação e batching;
- carregamento seguro com `weights_only=True`;
- orçamento de parâmetros da baseline densa.

## Principais módulos

- `src/modelo_gerador_esparso.py`: Q/K COO, atenção Top-K e FFN esparsa;
- `src/roteamento_combinatorio_v73.py`: roteador e microgrupos compartilhados;
- `src/runtime_cache_esparso.py`: cache causal reutilizável;
- `src/modelo_gerador_esparso_v73.py`: runtime e kernel opcional V7.3;
- `src/modelo_multimodal_v8.py`: encoder visual COO e fusão causal;
- `src/treino_vetorizado_v73.py`: backend vetorizado reutilizado no treino V8;
- `src/dados_multimodais_v8.py`: manifesto, imagens e lotes;
- `src/modelo_gerador_denso.py`: controle denso textual;
- `treinar_multimodal_v8.py`: ciclo de treino e checkpoints V8;
- `executar_multimodal_v8.py`: inferência imagem + texto.

## Estrutura

```text
AI-Esparsa/
├── modelos/
│   ├── gerador_denso_controle.pt
│   └── gerador_esparso_v73_base.pt
├── resultados/
│   └── v8_multimodal/
│       └── auditoria_arquitetura.json
├── src/
│   ├── kernels_v73/
│   ├── camada_linear_esparsa.py
│   ├── dados_multimodais_v8.py
│   ├── modelo_gerador_denso.py
│   ├── modelo_gerador_esparso.py
│   ├── modelo_gerador_esparso_v73.py
│   ├── modelo_multimodal_v8.py
│   ├── roteamento_combinatorio_v73.py
│   ├── runtime_cache_esparso.py
│   └── tokenizador_palavras.py
├── testes/
├── executar_gerador_esparso_v73.py
├── executar_multimodal_v8.py
├── treinar_multimodal_v8.py
├── DOCUMENTO_GERADOR_ESPARSO.md
├── STATUS.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Integridade dos modelos preservados

| Arquivo | Bytes | SHA-256 |
|---|---:|---|
| `modelos/gerador_esparso_v73_base.pt` | 691.539 | `BD98206D342A07FDE8D7AB19A48BE26D009E7FF46164AF97C9EEF33D02895185` |
| `modelos/gerador_denso_controle.pt` | 674.635 | `9571B6621237B7C489FA953746CADAD8829192FDFEE82B4F625A94259376860B` |

## Limitações atuais

- a V8 ainda não possui checkpoint treinado;
- não existem métricas de qualidade ou desempenho V8;
- a geração V8 usa recomputação completa, sem cache multimodal incremental;
- o controle denso preservado é textual e não mede vantagem visual;
- o tokenizador atual é de palavras; o novo dataset definirá o vocabulário.
