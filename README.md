# Gerador Esparso Coerente Base

Autor: Paulo Augusto  
Ano: 2026

## Objetivo

O Gerador Esparso Coerente é o modelo-base oficial do projeto. Ele produz
relatos procedurais longos condicionados por pedido, preservando agentes,
objeto, local, causa e ações ao longo do texto. A arquitetura mantém atenção
causal Q/K e FFN estruturalmente esparsas.

O checkpoint canônico está em:

`modelos/gerador_esparso_base.pt`

Os checkpoints anteriores foram preservados como controles históricos:

`modelos/v61_base.pt`
`modelos/v6_rollback.pt`

## Arquitetura

```text
tokens
  -> embedding + posição senoidal fixa
  -> 3 blocos de atenção causal Top-32
  -> projeções Q/K COO 128 -> 128
  -> residual + LayerNorm
  -> FFN COO 128 -> 384 -> 128, GELU + Top-64
  -> residual com gate aprendível + LayerNorm
  -> classificador amarrado ao embedding
```

### Atenção

- máscara estritamente causal;
- projeções Q/K COO com densidade estrutural de 25%;
- seleção Top-32 sobre estados anteriores;
- contexto máximo de 640 tokens;
- teste automatizado garante que o prefixo não depende do futuro.

### FFN

- duas projeções COO reais por bloco;
- 92.160 conexões nos três blocos;
- densidade estrutural de 25%;
- GELU e ativações Top-64;
- nenhuma `nn.Linear` ou matriz densa mascarada dentro da FFN;
- residual e normalização após a transformação.

O embedding, as normalizações e o classificador continuam densos. Portanto, a
FFN e as projeções Q/K são esparsas, mas o modelo completo não é anunciado como
100% composto apenas por operações esparsas.

## Checkpoint oficial

O checkpoint `modelos/gerador_esparso_base.pt` corresponde à época 5 da
execução promovida:

- semente `20260728`;
- 50.000 relatos de treino e divisões 1.000/1.000 para validação/teste;
- cinco épocas, 500 passos por época e 2.500 passos totais;
- lote 100, vocabulário 332 e sequência máxima 528;
- 163.667 parâmetros;
- PPL de teste `1,0505` e acurácia de token `97,53%`;
- 24/24 gerações aprovadas, com 24 frases;
- mínimo `2.802` e média `2.829,9` caracteres;
- recuperação e consistência dos campos: `100%`;
- vazamento de blocos Q&A e retentativas: `0%`;
- SHA-256 `daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f`.

O arquivo é carregado com `strict=True` e `weights_only=True`.

## Checkpoint histórico V6.1

O checkpoint `modelos/v61_base.pt` corresponde à execução historicamente
promovida antes do Gerador Esparso:

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

## Desempenho histórico da V6

O caminho integrado da V6 usa `torch.sparse.mm`. A tabela histórica abaixo mede
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
- Python 3.14;
- PyTorch 2.11 com CUDA 12.8;
- GPU NVIDIA compatível.

O treino promovido usou:

`C:\Python314\python.exe`

O `venv_cuda` citado na documentação histórica não estava presente nesta cópia
do workspace. Nenhuma dependência adicional foi instalada.

Os checkpoints são dicionários de tensores e metadados simples. Os carregadores
usam `torch.load(..., weights_only=True)` para impedir desserialização arbitrária
de objetos Python. Não use `weights_only=False` em arquivos não confiáveis.

## Instalação

Em um ambiente novo, crie e ative um ambiente virtual e instale uma versão do
PyTorch compatível com a GPU/CUDA local. O projeto não exige bibliotecas Python
adicionais além do PyTorch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch
```

## Execução

Gerar um relato de pelo menos 2.000 caracteres com a base oficial:

```powershell
python executar_gerador_esparso.py
```

Revalidar a época 5 candidata:

```powershell
python validar_gerador_esparso.py
```

Repetir o treino de 50.000 relatos, cinco épocas e 500 passos por época em uma
pasta vazia:

```powershell
python treinar_gerador_esparso.py
```

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

A suíte atual possui 65 testes automatizados.

## Exemplo de uso

```python
from pathlib import Path

import torch

from executar_gerador_esparso import carregar_gerador
from treinar_gerador_esparso import (
    extrair_campos_pedido,
    gerar_relato_validado,
)

dispositivo = torch.device("cuda")
modelo, tokenizador, metadados = carregar_gerador(
    Path("modelos/gerador_esparso_base.pt"),
    dispositivo,
)
prompt = (
    "Pedido: escreva um relato sobre bruno, com ajuda de tiago, para "
    "preparar uma feira de ciencias. Inclua a caixa na estante e o "
    "problema uma janela quebrada. Texto:"
)
texto, metricas = gerar_relato_validado(
    modelo,
    tokenizador,
    prompt,
    extrair_campos_pedido(prompt),
    dispositivo,
)
print(texto)
print(metricas["caracteres"])
```

## Módulos principais

- `AGENTS.md`: regras operacionais para agentes e segurança de checkpoints;
- `src/modelo_gerador_esparso.py`: arquitetura-base Q/K e FFN COO;
- `src/corpus_gerador_esparso.py`: vocabulário factual e geração controlada dos
  registros usados pelo modelo-base;
- `src/tokenizador_palavras.py`: tokenizador autocontido persistido no
  checkpoint oficial;
- `src/decodificador_contextual.py`: controles de repetição e parada;
- `treinar_gerador_esparso.py`: corpus longo, treino e métricas;
- `executar_gerador_esparso.py`: carga estrita e geração oficial;
- `validar_gerador_esparso.py`: revalidação independente e benchmark;
- `promover_gerador_esparso.py`: promoção segura por critérios e hash;
- `src/modelo_v6.py`: arquitetura anterior preservada como rollback;
- `src/modelo_v61.py`: controle histórico posicional;
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
AI-Esparsa/
  modelos/
    gerador_esparso_base.pt
    v61_base.pt
    v6_rollback.pt
  resultados/
    gerador_esparso_base_50k/
      epoca_01.pt
      epoca_02.pt
      epoca_03.pt
      epoca_04.pt
      epoca_05.pt
      relatorio.json
    teste_isolado_v6_20260726_150927/
    teste_isolado_v6_ultimo.json
    teste_v61_posicional_20260726_154023/
    teste_v61_posicional_ultimo.json
    v61_base_validacao.json
    v6_20260726_142346/
    v6_ultimo.json
  src/
    corpus_gerador_esparso.py
    decodificador_contextual.py
    modelo_gerador_esparso.py
    modelo_v6.py
    modelo_v61.py
    tokenizador_palavras.py
    atencao_causal_texto.py
  testes/
    test_decodificador_contextual.py
    test_modelo_gerador_esparso.py
    test_pipeline_gerador_esparso.py
    test_modelo_v6.py
    test_checkpoint_v6.py
    test_checkpoint_v61.py
    test_documentacao_v6.py
    test_v6_isolado.py
    test_v61_posicional.py
  DOCUMENTO_MODELO_V61.md
  AGENTS.md
  executar_v6.py
  executar_v61.py
  executar_gerador_esparso.py
  treinar_gerador_esparso.py
  validar_gerador_esparso.py
  promover_gerador_esparso.py
  testar_v6_isolado.py
  testar_v61_posicional.py
  treinar_v6.py
  validar_v61_base.py
```

## Limitações

- o gerador-base foi validado em domínio procedural controlado e vocabulário
  de 332 tokens;
- os 50.000 itens são combinações sintéticas, não linguagem natural aberta;
- 2.000 caracteres são um contrato desta tarefa, não garantia para prompts fora
  do formato documentado;
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
## Experimento V7 de texto variável

Foi adicionada uma variante experimental separada da V6.1 oficial para testar
geração de texto português com comprimentos variáveis, sem camadas lineares
densas internas. A arquitetura usa três blocos causais, atenção Q/K esparsa,
índice causal invertido, FFN COO, posição senoidal e mistura local causal
aprendível por deslocamento. O embedding e o classificador final permanecem
densos, como na V6.1.

O treino sintético de cinco épocas obteve PPL de teste `1,5988`, acurácia de
token `81,89%` e gerou parágrafos completos de três frases dentro do domínio
treinado. O modelo possui 92.331 parâmetros e zero `nn.Linear` interno além do
classificador de vocabulário.

Arquivos: `src/modelo_v7_texto.py`, `treinar_v7_texto.py`,
`testes/test_v7_texto.py` e `resultados/v7_texto_esparso.json`. Este resultado
não promove a V7 nem comprova linguagem aberta geral; o corpus ainda é sintético.

## Camada contextual experimental isolada

Foi criado um teste curto, separado da V8 e sem nome de versao, para avaliar se
uma unica camada recorrente esparsa consegue formar contexto antes de reintegrar
atencao ou FFN. A camada possui oito bancos de 16 canais, matrizes densas apenas
dentro de cada banco, fan-in de dois bancos, memoria rapida, memoria lenta,
combinacoes de pares e Top-12 por banco. A densidade estrutural equivalente das
tres matrizes locais e 12,5%; nao existe `nn.Linear` interno.

O teste utilizou textos procedurais de aproximadamente 110 tokens, com 360
exemplos de treino, 72 de validacao, 72 de teste, contexto 128 e cinco epocas.
Cada epoca levou cerca de 4,8 segundos e a execucao completa, incluindo geracao
e benchmark, levou 61,64 segundos. A PPL de teste caiu de 98,996 para 26,460 e
a acuracia de token chegou a 10,80%.

O modelo possui 20.319 parametros no total, dos quais 7.937 pertencem a camada.
A recuperacao livre do local mencionado anteriormente foi 0%; sob teacher
forcing, a preposicao foi prevista corretamente em 37,5% dos casos e o local em
0%. A camada aprendeu formato e sequencia local, mas ainda nao preservou a
informacao exata necessaria para responder. O resultado e diagnostico e nao
promove nem altera a V6.1, V7 ou V8.

Arquivos: `src/camada_contextual_experimental.py`,
`treinar_camada_contextual_rapido.py`,
`testes/test_camada_contextual_experimental.py` e
`resultados/camada_contextual_rapida/relatorio.json`.

## Camada contextual com slots esparsos

Foi testada uma extensao isolada da camada contextual com quatro slots dinamicos
por banco e leitura/escrita Top-2 por conteudo. Os slots armazenam chaves e
valores causais, sem atencao sobre tokens, sem FFN e sem `nn.Linear` interna.

A comparacao usou exatamente o mesmo corpus controlado da camada base: 360
textos de treino, 72 de validacao, 72 de teste, contexto 128 e cinco epocas. A
camada base obteve PPL 26,460, acuracia 10,80%, 17.400 tokens/s e 0% de
recuperacao do local. A variante com slots obteve PPL 28,264, acuracia 10,51%,
6.450 tokens/s e tambem 0% de recuperacao do local. Sob teacher forcing, a
preposicao subiu de 37,50% para 59,72%, mas o local permaneceu em 0%.

A variante possui 27.400 parametros, 34,85% a mais que a base, e foi 62,93%
mais lenta. A primeira implementacao normalizava chaves quase nulas a cada
passo e divergiu numericamente; a execucao valida usa chaves limitadas por
`tanh` e taxa de aprendizado menor. Cinco checkpoints estaveis foram salvos.

Decisao: os slots aprenderam o tipo de resposta, mas nao a associacao entre
objeto e local. A variante nao foi aprovada. Relatorios:
`resultados/camada_contextual_slots_rapida/relatorio.json` e
`resultados/camada_contextual_slots_rapida/comparacao_base.json`.

## Decodificacao da camada recuperadora

A repeticao apos a resposta foi corrigida sem novo treino. O decodificador agora
aplica penalidade de repeticao, bloqueia bigramas e trigramas repetidos e encerra
a geracao na primeira frase completa. No mesmo checkpoint da epoca 5, os 72
textos de teste obtiveram 100% de respostas exatas e 0% de repeticao apos a
resposta. A PPL permanece 14,894 e a acuracia de token 20,46%, pois os pesos nao
foram alterados. O resultado continua restrito ao corpus procedural controlado.

Arquivos: `src/decodificador_contextual.py`,
`avaliar_camada_contextual_recuperadora.py`,
`testes/test_decodificador_contextual.py` e
`resultados/camada_contextual_recuperadora_rapida/relatorio_decodificacao.json`.

## Gerador esparso coerente condicionado

A auditoria de geração longa encontrou causas diferentes que a PPL isolada não
mostrava:

- a camada contextual longa obteve PPL `6,4456`, mas entrava em ciclos durante
  a geração autorregressiva;
- os atalhos de cópia e replay reduziam a perda em teacher forcing, porém
  amplificavam a própria saída durante a geração;
- o bloqueio genérico de bigramas/trigramas impedia repetições semanticamente
  necessárias, como confirmar novamente `câmera no armário`, e podia criar
  contradições;
- o corpus antigo continha `de o`, `de a` e `disposto` para colaboradoras;
- o treino Wikipedia V7/V8 expunha somente um recorte de 384 tokens por artigo:
  `960.000` de `5.321.748` tokens, ou `18,04%` do treino por época. Com PPL
  entre `144` e `153`, esses checkpoints continuam inadequados para anunciar
  linguagem aberta coerente.

A correção inicial foi mantida separada da V6.1. Após autorização explícita e
validação ampliada com 50 mil relatos, a arquitetura foi promovida como nova
base; a V6.1 permaneceu intacta como controle histórico. O modelo usa:

```text
tokens + posição senoidal
  -> 3 blocos
       -> Q/K COO 25% + atenção causal Top-32 aprendida
       -> residual
       -> FFN COO 128 -> 384 -> 128, ativação Top-64
       -> residual e LayerNorm
  -> normalização e classificador amarrado ao embedding
```

O corpus promovido usa 50.000 relatos de treino, 1.000 de validação e 1.000 de
teste, com combinações disjuntas. Foram executadas exatamente cinco épocas e
500 passos por época, salvando `epoca_01.pt` a `epoca_05.pt`. O checkpoint
oficial fica em:

`modelos/gerador_esparso_base.pt`

### Resultado validado

| Métrica | Contextual anterior | Gerador corrigido |
|---|---:|---:|
| PPL | 6,4456 | 1,0505 |
| Acurácia de token | 39,32% | 97,53% |
| Gerações livres avaliadas | 1 amostra salva | 24 |
| Textos completos com 24 frases | não aprovado | 100% |
| Menor saída | — | 2.802 caracteres |
| Recuperação dos 5 campos do pedido | 40% na amostra | 100% |
| Consistência objeto-local | não validada em lote | 100% |
| Consistência das ações | não validada | 100% |
| Vazamento para pergunta/resposta | presente no teste antigo | 0% |
| Repetição média de trigramas | — | 1,13% |

No benchmark pareado, com lote 8, 200 tokens e oito repetições, o modelo
contextual anterior mediu aproximadamente `2.169 tokens/s` e `14,01 MiB` de
VRAM temporária. O gerador novo mediu aproximadamente `92.757 tokens/s` e
`11,75 MiB`, uma razão de `42,77x`. A comparação de geração antiga é apenas
diagnóstica, pois somente uma amostra havia sido salva.

O decodificador usa greedy quando a saída já preserva completude, campos,
ações e relação objeto-local. Se algum desses contratos falhar, faz até três
tentativas de baixa temperatura e seleciona uma saída válida. Na validação
promovida, os 24 prompts passaram no greedy sem retentativa.

Treinar novamente em uma pasta vazia:

```powershell
python treinar_gerador_esparso.py
```

Revalidar o checkpoint final com `weights_only=True`:

```powershell
python validar_gerador_esparso.py
```

Gerar um relato:

```powershell
python executar_gerador_esparso.py `
  "Pedido: escreva um relato sobre bruno, com ajuda de tiago, para preparar uma feira de ciencias. Inclua a caixa na estante e o problema uma janela quebrada. Texto:"
```

Módulos principais desta correção:

- `src/modelo_gerador_esparso.py`: Q/K COO, atenção causal Top-K, FFN COO,
  residuais, normalizações e classificador amarrado;
- `treinar_gerador_esparso.py`: corpus corrigido, treino de cinco épocas,
  avaliação livre, consistência e checkpoints;
- `executar_gerador_esparso.py`: carga estrita e geração com validação;
- `validar_gerador_esparso.py`: revalidação e comparação pareada;
- `testes/test_modelo_gerador_esparso.py`: causalidade, esparsidade e gradientes;
- `testes/test_pipeline_gerador_esparso.py`: corpus, prompt, checkpoint e
  consistência semântica.

O resultado resolve geração completa no domínio procedural treinado. Ele não
transforma os checkpoints Wikipedia em um modelo de linguagem geral. Para isso
seria necessário ampliar substancialmente dados, capacidade e orçamento de
treino, além de corrigir o pipeline para percorrer todos os segmentos dos
artigos.
