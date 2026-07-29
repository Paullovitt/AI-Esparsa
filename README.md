# Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

Este projeto mantém uma única arquitetura: o Gerador Esparso Coerente. O
modelo produz relatos procedurais longos condicionados por um pedido e preserva
agentes, tarefa, objeto, local, problema e ações ao longo do texto.

O checkpoint oficial é:

`modelos/gerador_esparso_base.pt`

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

O executor seleciona CUDA quando disponível e recarrega o checkpoint com
`weights_only=True` e `strict=True`.

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
| Forward paralelo, lote 16 × contexto 640 | 117.980,83 tokens/s |
| Pico de VRAM no forward | 88,46 MiB |
| Geração autorregressiva real | 97,37 tokens/s |
| Latência até o primeiro token | 5,57 ms |
| Tempo do relato completo | 5,02 s |

O relatório completo está em
`resultados/gerador_esparso_base_50k/relatorio.json`.

O throughput de forward mede sequências completas em paralelo e não representa
a velocidade percebida durante a geração. Por isso, o relatório mantém as duas
medições separadas.

## Principais módulos

- `src/modelo_gerador_esparso.py`: arquitetura causal esparsa;
- `src/camada_linear_esparsa.py`: projeção treinável COO, com índices
  coalescidos e cache seguro de inferência;
- `src/corpus_gerador_esparso.py`: geração determinística dos registros;
- `src/tokenizador_palavras.py`: tokenizador e contrato de vocabulário;
- `src/decodificador_gerador.py`: geração autorregressiva, repetição e
  telemetria;
- `treinar_gerador_esparso.py`: treino, avaliação e checkpoints;
- `executar_gerador_esparso.py`: inferência pelo checkpoint oficial;
- `validar_gerador_esparso.py`: revalidação independente e adversarial;
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
    gerador_esparso_base_50k/
      epoca_01.pt
      epoca_02.pt
      epoca_03.pt
      epoca_04.pt
      epoca_05.pt
      relatorio.json
  src/
    __init__.py
    camada_linear_esparsa.py
    corpus_gerador_esparso.py
    decodificador_gerador.py
    modelo_gerador_esparso.py
    tokenizador_palavras.py
  testes/
    test_decodificador_gerador.py
    test_gpu_gerador.py
    test_modelo_gerador_esparso.py
    test_pipeline_gerador_esparso.py
  pyproject.toml
  requirements.txt
  executar_gerador_esparso.py
  promover_gerador_esparso.py
  treinar_gerador_esparso.py
  validar_gerador_esparso.py
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
- uma baseline neural densa não faz parte do repositório porque o escopo
  autorizado mantém apenas o Gerador Esparso Coerente.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
