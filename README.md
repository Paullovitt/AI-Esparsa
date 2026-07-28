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
       -> atenção causal Top-32
       -> residual + LayerNorm
       -> FFN COO 128 -> 384 -> 128, GELU + Top-64
       -> residual com gate aprendível + LayerNorm
  -> classificador compartilhado com o embedding
```

O núcleo de transformação é estruturalmente esparso:

- Q/K COO com fan-in 32 e 25% de densidade;
- FFN COO com 25% de densidade;
- atenção limitada aos 32 estados causais mais relevantes;
- ativações da FFN limitadas ao Top-64;
- nenhuma `nn.Linear` ou MLP densa dentro dos blocos.

O modelo completo não é 100% esparso. Embedding/classificador, LayerNorm,
estados residuais e o cálculo temporário dos escores de atenção usam tensores
densos. O embedding/classificador é a principal matriz neural densa treinável.

## Dependências

- Python 3.11 ou superior;
- PyTorch 2.11 ou compatível;
- CUDA para treinamento, revalidação completa e benchmark de VRAM;
- GPU usada na execução oficial: NVIDIA GeForce RTX 3050 6 GB.

Não existem dependências de corpus externo. Os dados procedurais são gerados
deterministicamente por `src/corpus_gerador_esparso.py`.

## Instalação

Crie um ambiente virtual e instale uma compilação do PyTorch compatível com a
versão CUDA disponível no computador:

```powershell
python -m venv venv_cuda
.\venv_cuda\Scripts\Activate.ps1
python -m pip install torch
```

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
- consistência integral entre objeto, local e ações;
- repetição média de trigramas de até 1,5%;
- nenhum trigrama repetido mais de duas vezes;
- todas as gerações com pelo menos 2.000 caracteres;
- nenhum vazamento de blocos de pergunta/resposta.

## Resultados oficiais

| Métrica | Resultado |
|---|---:|
| Parâmetros | 163.667 |
| PPL final | 1,0505 |
| Acurácia de token | 97,53% |
| Gerações aprovadas | 24/24 |
| Caracteres mínimos | 2.802 |
| Caracteres médios | 2.829,9 |
| Frases por relato | 24 |
| Recuperação dos campos | 100% |
| Consistência objeto/local | 100% |
| Vazamento Q&A | 0% |
| Pico de VRAM no treino | 1.898,33 MiB |
| Velocidade do benchmark | 142.303,87 tokens/s |

O relatório completo está em
`resultados/gerador_esparso_base_50k/relatorio.json`.

## Principais módulos

- `src/modelo_gerador_esparso.py`: arquitetura causal esparsa;
- `src/camada_linear_esparsa.py`: projeção treinável COO;
- `src/corpus_gerador_esparso.py`: geração determinística dos registros;
- `src/tokenizador_palavras.py`: tokenizador persistido no checkpoint;
- `src/decodificador_gerador.py`: geração autorregressiva e repetição;
- `treinar_gerador_esparso.py`: treino, avaliação e checkpoints;
- `executar_gerador_esparso.py`: inferência pelo checkpoint oficial;
- `validar_gerador_esparso.py`: revalidação independente;
- `promover_gerador_esparso.py`: promoção segura de um candidato aprovado.

## Estrutura

```text
AI-Esparsa/
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
    test_modelo_gerador_esparso.py
    test_pipeline_gerador_esparso.py
  executar_gerador_esparso.py
  promover_gerador_esparso.py
  treinar_gerador_esparso.py
  validar_gerador_esparso.py
```

## Limitações

- o treinamento usa domínio procedural sintético;
- o vocabulário oficial contém 332 tokens;
- a validação não comprova linguagem natural aberta ou conhecimento geral;
- prompts fora do formato documentado podem não preservar os mesmos índices;
- o cálculo de atenção ainda materializa escores densos antes do Top-K.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE`.
