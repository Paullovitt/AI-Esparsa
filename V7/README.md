# AI-Esparsa V7.3 — BPE-8192

Gerador textual causal experimental com Q/K estruturalmente esparsos em COO, atenção causal Top-K, FFN combinatória roteada e cache incremental.

## Modelo principal

- Checkpoint operacional aprovado: `modelos/gerador_esparso_v73_bpe8192_refino_eos3_ul005_topologia.pt`
- SHA-256: `23af2ea91fb62153559edb9fb2ca3bf33655f1f6d3ab125751f9e33afd3339ad`
- Checkpoint anterior com topologia preservado: `modelos/gerador_esparso_v73_bpe8192_5x3000_topologia.pt`
- SHA-256 anterior: `0fa1e7fc43dcce7fd7d4561c941c3ee4c3d64ee6a2b41b5e199774a2927c21fa`
- Checkpoint original preservado: `modelos/gerador_esparso_v73_bpe8192_5x3000.pt`
- SHA-256 original: `151e3bed5c9d2c6f366fa4ae5799de8577a4baa41f82b381dbb5039fbfa43b65`
- Vocabulário: 8.192 tokens
- Arquitetura: dimensão 128, três camadas, atenção Top-32 e contexto máximo 640
- Parâmetros: 1.179.383
- Treinamento-base: cinco épocas e 15.000 passos
- Refino selecionado: três épocas e 9.000 passos, peso de EOS 3,0 e antirrepetição 0,05

O checkpoint operacional foi migrado para o formato de topologia COO validada, teve os estados do otimizador removidos para distribuição e preserva exatamente os pesos do refino aprovado. O carregador rejeita por padrão checkpoints sem essa identidade ou com conexões divergentes.

## Instalação e teste

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

## Execução

```powershell
python executar_gerador_esparso_v73.py --prompt "Explique inteligência artificial de forma simples." --permitir-fallback
```

O executor aplica automaticamente o formato usado no treinamento:

```text
usuario: <prompt>
assistente:
```

Somente os tokens novos da resposta são impressos. Para fornecer uma sequência já formatada, use `--prompt-cru`. Para auditoria determinística, acrescente `--decodificacao greedy`.

O FFN CUDA pode operar com o vocabulário 8.192. Já o caminho incremental totalmente fundido da projeção vocabular aceita no máximo 512 tokens; por isso o checkpoint principal usa o cache incremental PyTorch para essa parte. A auditoria do modelo informa essa decisão e não declara 11 kernels por token para o BPE-8192.

## Avaliação generativa

```powershell
python avaliar_geracao_v7.py --prompts-jsonl prompts.jsonl --saida resultados\avaliacao.json --permitir-fallback
```

O relatório mede greedy e amostragem, guarda somente a resposta gerada e aplica critérios explícitos à amostragem:

- fração média de tokens repetidos no máximo 0,50;
- repetição consecutiva média no máximo 3;
- `distinct_2` médio no mínimo 0,60;
- taxa de EOS no mínimo 0,20.

Os limites podem ser alterados pelas opções do CLI. O status final é `aprovado` ou `reprovado`; PPL isolada não aprova qualidade gerativa.

## Preparação de dados e treinamento

Os caminhos são portáveis e podem ser informados por CLI ou variáveis de ambiente:

- `AI_ESPARSA_DADOS_BRUTOS`;
- `AI_ESPARSA_DATASET`;
- `AI_ESPARSA_SAIDA`.

Sem configuração, dados e resultados ficam abaixo da pasta do projeto.

```powershell
python preparar_dataset_codigo_v7.py --dataset C:\dados\alpaca --saida C:\dados\v7_bpe8192
python treinar_textual_v7.py --dataset C:\dados\v7_bpe8192 --saida C:\resultados\experimento_01 --epocas 5 --passos-por-epoca 3000 --lote 32 --contexto 256
```

As opções `--dimensao-modelo`, `--camadas-modelo`, `--top-k-atencao-modelo`, `--fan-in-qk-modelo`, `--especialistas-modelo`, `--fan-in-ffn-modelo`, `--fan-out-ffn-modelo` e `--top-k-ffn-modelo` configuram efetivamente o modelo. `--fan-in-qkvo` permanece como alias compatível. A antiga opção de cabeças foi removida porque esta arquitetura não usa atenção multihead.

Cada execução deve usar uma pasta de saída nova. O treinador nunca sobrescreve checkpoints existentes e salva a identidade da topologia em todos os checkpoints novos.

## Migração de checkpoint legado

```powershell
python migrar_checkpoint_topologia_v73.py --entrada modelo_legado.pt --saida modelo_validado.pt
```

O comando nunca sobrescreve a saída. O executor só aceita um checkpoint sem topologia quando `--permitir-checkpoint-legado` é fornecido explicitamente.

## Métricas e limitações

A validação original registrou loss 4,069222, PPL 58,511445 e BPB 1,3635424677. A correção do alfabeto byte-level GPT-2 altera apenas o cálculo histórico do BPB para aproximadamente 1,3635468168; os pesos e tokens não mudam.

O refino selecionado registrou loss de validação 4,023068, PPL 55,872243 e BPB 1,34807665. Na avaliação 2.8.0 com 50 prompts e amostragem controlada, obteve repetição média de 0,1497, `distinct_2` de 0,9178, repetição consecutiva média de 1,32 e taxa de EOS de 0,74, recebendo status `aprovado`.

A geração continua experimental quanto à correção factual e coerência. O modo greedy puro permanece muito repetitivo; para uso normal, mantenha a amostragem controlada padrão.

Detalhes técnicos estão em `DOCUMENTO_GERADOR_ESPARSO_V73.md`; o histórico do treino original permanece em `RELATORIO_MODELO_BPE8192_5X3000_20260803.md`.
