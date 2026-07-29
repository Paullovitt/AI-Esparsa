# Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

Este projeto mantém o Gerador Esparso Coerente como único modelo oficial. Ele
produz relatos procedurais longos condicionados por um pedido e preserva
agentes, tarefa, objeto, local, problema e ações ao longo do texto. Uma
baseline densa de parâmetros equivalentes é mantida somente como controle
experimental e nunca substitui automaticamente o checkpoint oficial.

O checkpoint oficial é:

`modelos/gerador_esparso_base.pt`

A V6.2 é o runtime-base oficial. Ela reutiliza exatamente o checkpoint e os
163.667 parâmetros treinados, adiciona cache causal incremental, matrizes CSR
cacheadas em inferência e gates escalares reutilizados. Não existe um segundo
checkpoint nem alteração da topologia neural.

SHA-256:

`daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f`

## Arquitetura

```text
tokens
  -> embedding denso + posição senoidal fixa
  -> 3 blocos causais
       -> projeções Q/K COO 128 -> 128
       -> atenção causal Top-32 calculada em blocos de consultas
       -> residual + LayerNorm
       -> FFN COO 128 -> 384 -> 128, GELU + Top-64
       -> residual com gate aprendível + LayerNorm
  -> classificador compartilhado com o embedding
```

O núcleo de transformação é estruturalmente esparso:

- Q/K COO com fan-in 32 e 25% de densidade;
- FFN COO com 25% de densidade;
- atenção limitada aos 32 estados causais mais relevantes;
- consultas processadas em blocos de 32, sem materializar a matriz completa
  `[lote, tempo, tempo]`;
- ativações da FFN limitadas ao Top-64;
- nenhuma `nn.Linear` ou MLP densa dentro dos blocos.

O modelo completo não é 100% esparso. Embedding/classificador, LayerNorm,
estados residuais e o cálculo temporário dos escores de atenção usam tensores
densos. O embedding/classificador é a principal matriz neural densa treinável.
O Top-K em blocos preserva exatamente o resultado da implementação de
referência e limita a memória temporária dos escores a
`O(lote × bloco × tempo)`. A quantidade de produtos Q/K ainda é quadrática no
comprimento da sequência.

### Baseline densa experimental

A comparação autorizada usa uma baseline convencional com:

- 163.003 parâmetros, contra 163.667 do esparso (diferença de 0,41%);
- dimensão 88, três blocos e FFN 88-160-88;
- Q/K e FFN totalmente conectados;
- atenção causal densa pelo kernel otimizado do PyTorch;
- os mesmos residuais, normalizações, posição senoidal e pesos de
  embedding/saída amarrados.

A largura menor compensa a conectividade completa e mantém o orçamento de
parâmetros equivalente. A baseline é um controle científico, não um segundo
checkpoint oficial.

## Dependências

- Python 3.11 a 3.14;
- PyTorch 2.11.0;
- CUDA para treinamento, revalidação completa e benchmark de VRAM;
- GPU usada na execução oficial: NVIDIA GeForce RTX 3050 6 GB.

Não existem dependências de corpus externo. Os dados procedurais são gerados
deterministicamente por `src/corpus_gerador_esparso.py`.

O ambiente da revalidação oficial usou Windows 11, Python 3.14.0,
PyTorch 2.11.0+cu128 e CUDA 12.8. A integração contínua executa compilação e
testes de CPU no GitHub Actions; o teste de checkpoint em CUDA é ignorado
automaticamente quando não existe GPU.

## Instalação

Crie um ambiente virtual e instale uma compilação do PyTorch compatível com a
versão CUDA disponível no computador:

```powershell
python -m venv venv_cuda
.\venv_cuda\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para usar GPU, instale a distribuição PyTorch 2.11.0 compatível com o CUDA do
computador antes de executar a revalidação completa.

## Execução

Geração com o prompt padrão:

```powershell
python executar_gerador_esparso.py
```

Geração com pedido próprio:

```powershell
python executar_gerador_esparso.py "Pedido: escreva um relato sobre bruno, com ajuda de diego, para organizar uma pequena mostra cultural. Inclua o documento no corredor e o problema um atraso no transporte. Texto:"
```

O executor seleciona CUDA quando disponível, usa o runtime-base V6.2 e
recarrega o checkpoint com `weights_only=True` e `strict=True`.

O alias explícito abaixo produz o mesmo resultado:

```powershell
python executar_v62_cache.py
```

Também é possível fornecer o mesmo contrato de prompt aceito pelo executor
oficial:

```powershell
python executar_v62_cache.py "Pedido: escreva um relato sobre bruno, com ajuda de diego, para organizar uma pequena mostra cultural. Inclua o documento no corredor e o problema um atraso no transporte. Texto:"
```

O pedido público tem contrato fechado: deve manter exatamente a ordem dos
campos mostrada acima, terminar com `Texto:` e usar somente palavras do
vocabulário persistido. Um campo ausente, invertido ou desconhecido gera
`ValueError` antes da inferência; o programa não substitui silenciosamente o
conteúdo por `<unk>`.

## Treinamento

O comando padrão reproduz a configuração promovida:

```powershell
python treinar_gerador_esparso.py
```

Configuração oficial:

- 50.000 relatos de treino;
- 1.000 relatos de validação;
- 1.000 relatos de teste;
- cinco épocas;
- lote 100;
- 500 passos por época e 2.500 passos totais;
- semente 20260728;
- 24 frases por relato de geração;
- mínimo exigido de 2.000 caracteres.

Cada época é salva separadamente em
`resultados/gerador_esparso_base_50k/`.

## Validação

```powershell
python validar_gerador_esparso.py
python validar_comparacao_esparso_denso.py
python -m unittest discover -s testes -v
```

Critérios obrigatórios:

- PPL máxima de 1,20;
- 100% de completude;
- 100% de recuperação dos campos;
- consistência integral entre objeto, local e ações, verificada por cláusula;
- repetição média de trigramas de até 1,5%;
- nenhum trigrama repetido mais de duas vezes;
- todas as gerações com pelo menos 2.000 caracteres;
- nenhum vazamento de blocos de pergunta/resposta;
- armazenamento das 24 gerações, para auditoria integral;
- aprovação dos contratos adversariais de contradição, distrator, campo
  ausente, ordem inválida e vocabulário desconhecido.

O validador 2.0 corrige a antiga janela local de quatro tokens. Por exemplo,
`o livro foi levado para o deposito, mas o livro continuava na sala` agora é
rejeitado. Uma menção a outro objeto em outro local não causa falso positivo.

## Resultados oficiais

| Métrica | Resultado |
|---|---:|
| Parâmetros | 163.667 |
| PPL final | 1,0505 |
| Acurácia de token | 97,53% |
| Gerações aprovadas | 24/24 |
| Gerações armazenadas no relatório | 24/24 |
| Caracteres mínimos | 2.806 |
| Caracteres médios | 2.830,3 |
| Frases por relato | 24 |
| Recuperação dos campos | 100% |
| Consistência objeto/local | 100% |
| Uso de retentativa | 12,5% |
| Vazamento Q&A | 0% |
| Pico de VRAM no treino | 1.898,33 MiB |

### Desempenho da última revalidação

<!-- metricas-desempenho:inicio -->
| Medição | Resultado |
|---|---:|
| Forward paralelo, lote 16 × contexto 640 | 144.957,06 tokens/s |
| Pico de VRAM no forward | 83,42 MiB |
| Geração autorregressiva real | 314,99 tokens/s |
| Latência até o primeiro token | 4,50 ms |
| Tempo do relato completo | 1,55 s |
<!-- metricas-desempenho:fim -->

O relatório completo está em
`resultados/gerador_esparso_base_50k/relatorio.json`.

O throughput de forward mede sequências completas em paralelo e não representa
a velocidade percebida durante a geração. Por isso, o relatório mantém as duas
medições separadas. As linhas de desempenho acima são atualizadas
automaticamente por `validar_gerador_esparso.py`; o `relatorio.json` é a fonte
única dos valores.

## Comparação esparso × denso

O experimento utilizou os mesmos 50.000/1.000/1.000 registros, tokenizador,
ordem de lotes, semente, cinco épocas, 500 passos por época, lote 100, AdamW,
agenda de aprendizado e decodificador. O treino oficial esparso já satisfazia
exatamente esse protocolo e foi reutilizado; a baseline densa foi treinada do
zero. Portanto, os tempos foram medidos na mesma GPU, mas em execuções
distintas.

Para criar uma nova rodada em outro diretório:

```powershell
python comparar_esparso_denso.py --resultados resultados/minha_comparacao
python validar_comparacao_esparso_denso.py --resultados resultados/minha_comparacao
```

| Métrica | Esparso | Denso | Melhor |
|---|---:|---:|---|
| Parâmetros | 163.667 | 163.003 | equivalente |
| PPL de teste | 1,050526 | 1,049220 | denso |
| Acurácia de token | 97,53% | 97,59% | denso |
| Gerações aprovadas | 24/24 | 24/24 | empate |
| Recuperação dos campos | 100% | 100% | empate |
| Repetição média de trigramas | 1,17% | 1,03% | denso |
| Uso de retentativa | 12,50% | 8,33% | denso |
| Tempo das cinco épocas | 1.663,30 s | 281,09 s | denso |
| Pico de VRAM no treino | 1.898,33 MiB | 950,36 MiB | denso |
| Forward paralelo atual | 144.957,06 tokens/s | 1.537.708,78 tokens/s | denso |
| Pico de VRAM no forward atual | 83,42 MiB | 122,21 MiB | esparso |
| Geração autorregressiva atual | 314,99 tokens/s | 506,44 tokens/s | denso |
| Primeiro token atual | 4,50 ms | 2,07 ms | denso |
| Checkpoint | 662,32 KiB | 658,82 KiB | equivalente |

Conclusão atual: a baseline densa preservou a qualidade, treinou 5,92× mais
rápido, reduziu a VRAM de treino em 49,94%, fez o forward 10,61× mais rápido e
gerou 1,61× mais rápido. A V6.2 reduziu substancialmente a diferença de geração
e venceu no pico de VRAM do forward, usando 31,74% menos memória.

Essa conclusão é específica deste domínio, hardware e implementação PyTorch;
não demonstra superioridade universal de arquiteturas densas.

Artefatos completos:

- `resultados/comparacao_esparso_denso_50k/comparacao.json`;
- `resultados/comparacao_esparso_denso_50k/COMPARACAO.md`;
- relatórios individuais, 24 textos por modelo e cinco checkpoints densos.

## V6.2 base: cache causal e CSR

A V6.2 é o runtime-base oficial e mantém em buffers prealocados:

- as chaves esparsas já projetadas;
- os estados normalizados usados como valores;
- os tokens da janela causal ativa.

Cada token novo calcula apenas sua consulta, sua chave e o caminho residual
correspondente. Pesos, Q/K COO, Top-32, FFN COO, Top-64, gates, normalizações e
classificador continuam idênticos ao checkpoint treinado. Em inferência, cada
matriz COO é convertida uma única vez para CSR, eliminando a conversão interna
repetida do cuSPARSE, e os sete gates escalares são reutilizados enquanto seus
parâmetros não mudam. O runtime não acrescenta parâmetros. Quando a janela
ultrapassa 640 tokens, ela é refeita para preservar exatamente a convenção
posicional da referência.

O experimento completo pode ser repetido em um novo diretório:

```powershell
python experimentar_v62_cache.py --resultados resultados/minha_v62
```

Resultado oficial revalidado na RTX 3050:

| Métrica autorregressiva | Referência sem cache | V6.2 base |
|---|---:|---:|
| Velocidade | 92,22 tokens/s | 314,99 tokens/s |
| Tempo de 489 tokens | aproximadamente 5,30 s | 1,55 s |
| Primeiro token | 11,95 ms | 4,50 ms |
| Pico de VRAM no forward | 90,09 MiB | 83,42 MiB |
| Caracteres | 2.840 | 2.840 |

A geração ficou **3,42× mais rápida** que a referência original e 40,7% mais
rápida que a primeira implementação V6.2 de 223,82 tokens/s. A PPL atual é
`1,050526398`; as 24 gerações foram aprovadas e permaneceram textualmente
idênticas às saídas oficiais. No relatório isolado, a maior diferença numérica
de logits foi `7,63e-6`, abaixo da tolerância de `2e-5`.

A V6.2 ainda não supera os 506,44 tokens/s da baseline densa e não muda o
treino já executado. O checkpoint oficial não foi sobrescrito: a promoção é
exclusivamente do runtime seguro que o carrega.

Outras variantes foram testadas e rejeitadas:

- `torch.compile`: indisponível sem uma instalação Triton funcional no Windows;
- FP16: apenas 2,1% mais rápido e diferença de logits `0,00952`, acima da
  tolerância;
- FFN seletiva por `scatter_add`: 3,1% mais lenta e com mais VRAM.

Relatório e texto auditável:

- `resultados/v62_base_runtime/relatorio.json`;
- `resultados/v62_base_runtime/texto_benchmark.txt`.

## Principais módulos

- `src/modelo_gerador_esparso.py`: arquitetura causal esparsa;
- `src/modelo_gerador_esparso_v62.py`: runtime-base com cache causal e CSR;
- `src/modelo_gerador_denso.py`: baseline densa experimental equivalente;
- `src/camada_linear_esparsa.py`: projeção treinável COO, com índices
  coalescidos e cache seguro de inferência;
- `src/corpus_gerador_esparso.py`: geração determinística dos registros;
- `src/tokenizador_palavras.py`: tokenizador e contrato de vocabulário;
- `src/decodificador_gerador.py`: geração autorregressiva, repetição e
  telemetria;
- `treinar_gerador_esparso.py`: treino, avaliação e checkpoints;
- `executar_gerador_esparso.py`: inferência pelo checkpoint oficial;
- `executar_v62_cache.py`: alias explícito da inferência V6.2-base;
- `experimentar_v62_cache.py`: equivalência, PPL, geração e benchmark V6.2;
- `validar_gerador_esparso.py`: revalidação independente e adversarial;
- `comparar_esparso_denso.py`: protocolo de treino e comparação justa;
- `validar_comparacao_esparso_denso.py`: recarga e revalidação independente;
- `promover_gerador_esparso.py`: promoção segura de um candidato aprovado.

## Estrutura

```text
AI-Esparsa/
  .github/
    workflows/
      ci.yml
  modelos/
    gerador_esparso_base.pt
  resultados/
    comparacao_esparso_denso_50k/
      esparso/
        relatorio.json
      denso/
        epoca_01.pt ... epoca_05.pt
        relatorio.json
      comparacao.json
      COMPARACAO.md
      protocolo.json
    gerador_esparso_base_50k/
      epoca_01.pt
      epoca_02.pt
      epoca_03.pt
      epoca_04.pt
      epoca_05.pt
      relatorio.json
    v62_base_runtime/
      relatorio.json
      texto_benchmark.txt
  src/
    __init__.py
    camada_linear_esparsa.py
    corpus_gerador_esparso.py
    decodificador_gerador.py
    documentacao_metricas.py
    modelo_gerador_denso.py
    modelo_gerador_esparso.py
    modelo_gerador_esparso_v62.py
    tokenizador_palavras.py
    versao.py
  testes/
    test_comparacao_esparso_denso.py
    test_decodificador_gerador.py
    test_documentacao_metricas.py
    test_gpu_gerador.py
    test_modelo_gerador_denso.py
    test_modelo_gerador_esparso.py
    test_modelo_gerador_esparso_v62.py
    test_pipeline_gerador_esparso.py
  pyproject.toml
  requirements.txt
  comparar_esparso_denso.py
  executar_gerador_esparso.py
  executar_v62_cache.py
  experimentar_v62_cache.py
  promover_gerador_esparso.py
  treinar_gerador_esparso.py
  validar_gerador_esparso.py
  validar_comparacao_esparso_denso.py
```

## Limitações

- o treinamento usa domínio procedural sintético;
- o vocabulário oficial contém 332 tokens;
- a validação não comprova linguagem natural aberta ou conhecimento geral;
- a avaliação de geração usa combinações separadas, mas segue o mesmo gerador
  procedural do treino;
- prompts fora do formato ou do vocabulário são rejeitados;
- a atenção não guarda a matriz completa de escores, mas ainda executa
  aproximadamente `O(tempo²)` produtos Q/K;
- a baseline compara orçamento de parâmetros, mas precisa reduzir a dimensão
  interna de 128 para 88;
- os tempos de treino esparso e denso vêm de execuções distintas na mesma GPU;
- uma única baseline e um domínio procedural não provam uma regra universal.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
