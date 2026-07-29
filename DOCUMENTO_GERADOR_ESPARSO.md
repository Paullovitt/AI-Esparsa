# Documento técnico do Gerador Esparso Coerente

Autor: Paulo Augusto
Ano: 2026

## Escopo

O Gerador Esparso Coerente é a única arquitetura oficial deste repositório.
Seu objetivo é produzir relatos procedurais longos condicionados por campos
explícitos de um pedido. Uma baseline densa existe exclusivamente como
controle experimental autorizado e não altera o checkpoint oficial.

O runtime-base oficial é a V6.2. Ele reutiliza os pesos treinados e acrescenta
cache causal incremental, cache linear CSR e cache dos gates escalares.

Checkpoint oficial:

`modelos/gerador_esparso_base.pt`

## Topologia

O modelo possui dimensão 128, três blocos causais, contexto máximo de 640
tokens e 163.667 parâmetros. Cada bloco contém:

1. projeções Q/K treináveis em COO;
2. seleção causal Top-32 processada em blocos de 32 consultas;
3. residual e LayerNorm;
4. FFN COO 128-384-128 com GELU e Top-64;
5. residual com gate treinável e LayerNorm.

As projeções Q/K e FFN têm 25% de densidade estrutural. Não existe `nn.Linear`
densa no núcleo. O embedding/classificador compartilhado é uma matriz
treinável densa; normalização, residuais, softmax e escores temporários também
operam sobre tensores densos. Por isso, o modelo é descrito como tendo núcleo
estruturalmente esparso, não como 100% esparso.

Cada bloco de consultas é comparado com todo o prefixo causal e conserva os
mesmos 32 maiores escores da referência que materializava
`[lote, tempo, tempo]`. Testes numéricos confirmam equivalência dentro de
tolerância de `1e-6`. A memória de escores passa a ser
`O(lote × bloco × tempo)`, mas a quantidade total de produtos Q/K continua
aproximadamente quadrática no tempo.

As matrizes COO mantêm os valores treináveis na ordem original do checkpoint,
ordenam os índices uma única vez e informam ao PyTorch que o tensor já está
coalescido. Em inferência, a representação esparsa é reutilizada enquanto a
versão do parâmetro não muda; treino, troca de dispositivo ou alteração dos
pesos invalidam esse cache.

## Controle denso equivalente

A baseline usa dimensão 88, três blocos, FFN 88-160-88, Q/K densos e atenção
causal densa por `scaled_dot_product_attention`. Residual, LayerNorm, posição
senoidal, classificador amarrado e contexto 640 são preservados. Ela possui
163.003 parâmetros contra 163.667 do esparso, diferença de 0,41%.

Essa escolha controla o orçamento de parâmetros, mas não a largura: o modelo
esparso opera em dimensão 128. O kernel denso oficial do PyTorch foi usado para
evitar uma baseline artificialmente lenta.

## Causalidade

A máscara impede acesso a tokens futuros. A seleção Top-K ocorre somente entre
posições causais válidas. O teste automatizado compara prefixos antes e depois
de alterações em tokens futuros.

## Dados e tokenização

O corpus procedural é gerado deterministicamente e dividido sem sobreposição:

- treino: 50.000 relatos;
- validação: 1.000 relatos;
- teste: 1.000 relatos;
- 24 frases por relato extenso;
- vocabulário: 332 tokens;
- sequência máxima observada: 528 tokens.

O tokenizador separa palavras e pontuação e é reconstruído pelo vocabulário
persistido no checkpoint.

## Contrato de entrada

O executor aceita o formato:

```text
Pedido: escreva um relato sobre <pessoa>, com ajuda de <ajudante>, para <tarefa>. Inclua <objeto> no <local> e o problema <problema>. Texto:
```

Todos os tokens precisam existir no vocabulário do checkpoint. Formato
incompleto, campos invertidos e palavras desconhecidas são rejeitados antes da
inferência, em vez de serem convertidos silenciosamente para `<unk>`.

## Treinamento oficial

- semente: 20260728;
- cinco épocas;
- lote: 100;
- 500 passos por época;
- 2.500 atualizações totais;
- tempo total: 1.766,86 segundos;
- pico de VRAM: 1.898,33 MiB.

Cada época possui um checkpoint separado. Todos são carregáveis com
`torch.load(..., weights_only=True)`.

## Validação

- PPL final: 1,0505;
- acurácia de token: 97,53%;
- 24/24 gerações aprovadas;
- 24/24 gerações armazenadas no relatório;
- mínimo de 2.806 caracteres;
- média de 2.830,3 caracteres;
- recuperação dos campos: 100%;
- consistência objeto/local e ações: 100%;
- vazamento Q&A: 0%;
- repetição média de trigramas: 1,17%;
- frequência máxima de um trigrama: duas ocorrências;
- retentativa controlada: 3/24 gerações;
- cinco contratos adversariais aprovados.

O validador 2.0 verifica relações de objeto e local por cláusula completa. A
contradição reproduzível `o livro foi levado para o deposito, mas o livro
continuava na sala` é rejeitada; menções a um objeto diferente funcionam como
distratores e não reprovam um texto válido.

## Desempenho e ambiente

A revalidação separa métricas com significados diferentes:

<!-- metricas-desempenho:inicio -->
- forward paralelo, lote 16 e contexto 640: 144.957,06 tokens/s,
  70,64 ms e pico de 83,42 MiB;
- geração autorregressiva de um relato: 314,99 tokens/s;
- latência até o primeiro token: 4,50 ms;
- tempo total para 489 tokens e 2.840 caracteres: 1,55 s.
<!-- metricas-desempenho:fim -->

O teste oficial foi executado em Windows 11, Python 3.14.0,
PyTorch 2.11.0+cu128, CUDA 12.8 e NVIDIA GeForce RTX 3050. `pyproject.toml` e
`requirements.txt` fixam a dependência principal. O GitHub Actions executa a
suíte de CPU; `testes/test_gpu_gerador.py` valida localmente a carga e o
forward do checkpoint em CUDA, ou é ignorado quando não existe GPU.

## Segurança de checkpoint

O executor e o validador usam `weights_only=True` e recarga estrita. O script
de promoção recusa sobrescrever um checkpoint oficial existente e exige cinco
épocas, 50.000 amostras, 500 passos por época e todos os critérios de
revalidação aprovados. Também exige 24 saídas auditáveis, validador 2.0 e
medição autorregressiva válida.

## Comparação experimental

O protocolo fixou os mesmos dados, tokenizador, ordem dos lotes, semente, cinco
épocas, lote 100, 500 passos por época, AdamW, agenda, pesos da perda e
decodificador. O treino esparso oficial já correspondia a esse protocolo; a
baseline densa foi treinada do zero, com cinco checkpoints separados.

| Métrica | Esparso | Denso |
|---|---:|---:|
| Parâmetros | 163.667 | 163.003 |
| PPL de teste | 1,050526 | 1,049220 |
| Acurácia | 97,53% | 97,59% |
| Aprovação/recuperação | 100% / 100% | 100% / 100% |
| Tempo de treino | 1.663,30 s | 281,09 s |
| Pico de VRAM no treino | 1.898,33 MiB | 950,36 MiB |
| Forward atual | 144.957,06 tokens/s | 1.537.708,78 tokens/s |
| Pico de VRAM no forward atual | 83,42 MiB | 122,21 MiB |
| Geração autorregressiva atual | 314,99 tokens/s | 506,44 tokens/s |
| Primeiro token atual | 4,50 ms | 2,07 ms |

A baseline densa empatou em aprovação e recuperação, obteve PPL ligeiramente
menor e venceu tempo de treino, VRAM de treino, forward e geração. A diferença
de geração caiu de 5,49 para 1,61 vezes com a V6.2; o esparso manteve a
vantagem de memória no forward, agora de 31,74%. Logo, os dados ainda não
sustentam superioridade prática geral do esparso. A revalidação independente
reproduziu a PPL densa com erro absoluto de `8,96e-8` e aprovou novamente as
24 gerações.

## Runtime-base V6.2

A V6.2 preserva integralmente o estado treinado, a topologia COO, a atenção
causal Top-32, a FFN COO Top-64, os residuais, os gates e as normalizações. Ela
adiciona um caminho incremental para geração:

1. o prompt executa um prefill completo;
2. cada camada armazena chaves e valores causais em buffers prealocados;
3. cada token seguinte projeta somente o estado novo;
4. a consulta nova seleciona Top-32 no prefixo cacheado;
5. ao preencher 640 posições, a janela é refeita para preservar a posição
   senoidal e o comportamento exato da referência.
6. as matrizes COO imutáveis são materializadas uma vez em CSR na inferência;
7. os sete gates escalares são reutilizados enquanto os parâmetros não mudam.

O runtime não possui parâmetros adicionais e carrega
`modelos/gerador_esparso_base.pt` com `weights_only=True` e recarga estrita.
Não houve novo treino nem criação de checkpoint. A V6.2 foi promovida somente
como runtime-base e o SHA-256 dos pesos permaneceu
`daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f`.

Na RTX 3050, uma geração determinística de 489 tokens e 2.840 caracteres
passou de 92,22 para 314,99 tokens/s, aceleração de 3,42 vezes. O tempo do
relato completo caiu para 1,55 segundo e o primeiro token para 4,50 ms. O
forward atual atingiu 144.957,06 tokens/s e 83,42 MiB.

A PPL V6.2 foi `1,050526398`. A maior diferença numérica entre logits
incrementais e a referência no relatório isolado foi `7,63e-6`, abaixo da
tolerância `2e-5`. As 24 gerações passaram novamente e ficaram textualmente
idênticas às saídas oficiais.

A V6.2 ainda não supera a geração densa de 506,44 tokens/s nem altera o custo
do treino já executado. Testes com FP16 e FFN seletiva foram rejeitados por
divergência acima da tolerância ou regressão de desempenho. `torch.compile`
não executou porque não existe uma instalação Triton funcional neste Windows.

Artefatos:

- `src/modelo_gerador_esparso_v62.py`;
- `executar_v62_cache.py`;
- `experimentar_v62_cache.py`;
- `resultados/v62_base_runtime/relatorio.json`.

## Limite de validade

As métricas comprovam desempenho no domínio procedural treinado. Elas não
demonstram linguagem aberta geral. A avaliação usa combinações disjuntas, mas
segue o mesmo gerador procedural do treino. A seleção é esparsa e não mantém a
matriz completa de atenção, embora a computação Q/K ainda seja quadrática.
Os tempos dos dois treinos foram registrados em execuções distintas na mesma
GPU. A equivalência é por parâmetros, não por largura. O resultado é específico
do corpus procedural, da RTX 3050 e dos kernels PyTorch avaliados; não autoriza
promover automaticamente a baseline nem generalizar a conclusão.
