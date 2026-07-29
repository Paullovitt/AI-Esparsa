# Documento técnico — AI-Esparsa V8

Autor: Paulo Augusto

Ano: 2026

## 1. Escopo

O projeto mantém a base textual V7.3, uma arquitetura experimental V8
multimodal e um controle denso textual. A V8 está implementada, mas ainda não
foi treinada. Nenhum resultado de qualidade deve ser atribuído a ela antes do
novo experimento.

## 2. Inventário canônico

- `modelos/gerador_esparso_v73_base.pt`: V7.3 textual, época 2, FP32;
- `modelos/gerador_denso_controle.pt`: baseline densa textual, época 5;
- `src/modelo_multimodal_v8.py`: V8 imagem + texto sem checkpoint.

O checkpoint V7.3 é imutável durante experimentos. A baseline densa é somente
controle e não pode ser promovida.

## 3. Base V7.3

A V7.3 tem 165.443 parâmetros com o vocabulário preservado. Cada um dos três
blocos contém:

- normalização antes da atenção;
- projeções Q/K COO 128 × 128, fan-in 32;
- atenção causal Top-32;
- residual com gate;
- normalização antes da FFN;
- banco FFN COO com 384 unidades;
- 16 microgrupos e quatro grupos ativos por token;
- residual final com gate.

O roteador baixo-rank é aprendido. A escolha de quatro entre 16 grupos produz
1.820 combinações teóricas por camada. Os grupos reutilizam o mesmo banco de
arestas COO; não são especialistas densos independentes.

O runtime textual oferece cache causal prealocado, CSR de inferência, gates
cacheados e kernel CUDA opcional. O código-fonte do kernel fica em
`src/kernels_v73/`; o binário compilado é cache local descartável.

## 4. V8 multimodal

### 4.1 Encoder visual

Entrada esperada: tensor RGB `[lote, 3, 64, 64]`, normalizado em `[-1, 1]`.
A imagem é dividida em patches não sobrepostos de 8 × 8, totalizando 64
tokens visuais. Cada patch possui 192 valores.

A projeção patch → dimensão 128 usa `LinearBlocoEsparsa`:

- 128 saídas;
- fan-in visual 48;
- 6.144 pesos COO;
- densidade 25%;
- nenhuma camada `nn.Linear`.

Uma LayerNorm, posição senoidal, embedding de modalidade e gate escalar
completam o caminho visual. Incluindo biases e modalidade textual, o acréscimo
é de 6.785 parâmetros.

### 4.2 Fusão

Os patches são colocados antes do texto:

```text
[patch_0 ... patch_63] [bos prompt resposta ...]
```

A máscara causal existente faz cada token textual enxergar o prefixo visual e
os tokens textuais anteriores, mas nunca tokens textuais futuros. Os patches
também atravessam os blocos combinatórios, portanto alteram o estado que chega
ao roteador. Logits são calculados apenas para o trecho textual.

### 4.3 Transferência

Com o vocabulário preservado, 56 tensores da V7.3 são transferíveis e sete
tensores são novos. Se o vocabulário do novo corpus mudar, embedding e bias de
saída são reinicializados por incompatibilidade de shape; blocos compatíveis
continuam transferíveis.

### 4.4 Estado comprovado

O relatório `resultados/v8_multimodal/auditoria_arquitetura.json` registra:

- 172.228 parâmetros no teste com vocabulário de 332 tokens;
- logits finitos;
- loss auxiliar de roteamento finita;
- imagens distintas alteram logits;
- gradientes chegam à projeção visual e ao roteador;
- causalidade textual preservada;
- ausência de lineares densas no encoder visual.

Essas são provas estruturais, não métricas de aprendizado.

## 5. Dados

Dados não fazem parte do repositório. O protocolo usa manifestos JSONL locais
com `id`, `imagem`, `prompt` e `resposta`. Caminhos absolutos e travessia para
fora da raiz são rejeitados. O checkpoint guarda somente hashes SHA-256 dos
manifestos.

Treino e validação devem ser separados antes da criação dos manifestos.
Imagens ou famílias quase idênticas não podem aparecer em splits diferentes.

## 6. Protocolo futuro de treino

Um experimento comparável deve:

- fixar seed;
- usar cinco épocas para ser elegível;
- salvar cada época separadamente;
- não sobrescrever os modelos canônicos;
- registrar loss, PPL, acurácia, tempo, throughput, VRAM e checkpoint;
- medir recuperação texto-imagem e geração livre;
- auditar diversidade das máscaras de roteamento;
- comparar inicialização V7.3 contra treino do zero;
- registrar hashes dos manifestos.

O script `treinar_multimodal_v8.py` já implementa o ciclo básico, checkpoint
seguro e avaliação de loss/PPL. Benchmarks de qualidade multimodal serão
definidos quando o dataset for escolhido, porque dependem das tarefas reais.

O treino usa `ModeloMultimodalEsparsoTreinoV8`: somente microgrupos escolhidos
são materializados em pequenos blocos densos temporários, enquanto atenção e
Q/K usam o backend vetorizado. Esses tensores temporários não viram parâmetros
nem alteram o `state_dict`; o checkpoint carrega estritamente no runtime de
referência esparso.

## 7. Limitações e riscos

- não há checkpoint V8 treinado;
- o cache incremental textual não aceita prefixo visual; V8 recompõe o
  forward durante geração;
- a baseline densa atual não é multimodal;
- patches 64 × 64 perdem detalhes finos;
- o tokenizador de palavras pode crescer demais em corpus aberto;
- o caminho PyTorch COO pode ter overhead de treino maior que kernels densos;
- influência inicial da imagem não garante roteamento semanticamente útil.

## 8. Critérios antes de promover uma futura V8

1. Cinco épocas completas com checkpoints íntegros.
2. Loss e PPL de validação sem divergência.
3. Generalização em imagens e prompts fora do treino.
4. Ablação sem imagem e com imagem embaralhada.
5. Máscaras diferentes para contextos diferentes sem colapso.
6. Comparação de parâmetros ativos, VRAM e velocidade.
7. Teste humano de coerência e fidelidade visual.
8. Relatório JSON como fonte única das métricas.
9. Autorização explícita antes de substituir a base V7.3.
