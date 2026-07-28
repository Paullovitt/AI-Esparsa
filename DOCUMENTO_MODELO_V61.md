# Modelo-base V6.1

Autor: Paulo Augusto  
Ano: 2026

## Decisão

A V6.1 posicional é oficialmente o modelo-base do projeto. Sua arquitetura
combina atenção causal esparsa, ordem temporal explícita e FFN bloco-esparsa.
A V6 anterior permanece somente como rollback verificável.

Checkpoint oficial:

`modelos/v61_base.pt`

## Arquitetura

```text
tokens
  -> embedding + posição senoidal fixa + estado local
  -> descritor com slot e papel temporal esparsos
  -> atenção causal com Q/K esparsos e Top-24
  -> residual e LayerNorm
  -> FFN COO 48 -> 144 -> 48 com ReLU Top-12
  -> residual aprendível e LayerNorm
  -> classificador de tokens
```

A FFN não contém `nn.Linear` nem matriz densa mascarada. Suas duas projeções
possuem 3.456 conexões e densidade estrutural de 25%. A posição é fixa e os
códigos temporais não possuem pesos. O modelo completo possui 9.632 parâmetros.

## Checkpoint canônico

- semente: `20260727`;
- época: `5`;
- PPL de teste: `1,5308`;
- PPL média em quatro sementes de dados: `1,5317`;
- acurácia de token média: `88,06%`;
- localização da resposta: `100%`;
- recuperação causal: `100%`;
- geração livre exata média: `93,06%`;
- tokens livres alinhados: `99,17%`;
- locais livres: `99,54%`.

## V6 anterior como rollback

O checkpoint anterior foi renomeado para `modelos/v6_rollback.pt`. Seus
resultados históricos foram:

- PPL: `1,9196`;
- melhor PPL de teste: `1,9077`;
- acurácia de token: `77,41%`;
- localização, recuperação e primeiro token controlado: `100%`;
- PPL sem a FFN após ablação: `4,9899`.

A suíte original de vinte e três testes verificava causalidade, topologia COO, gradientes
esparsos, determinismo, recarga estrita dos checkpoints e documentação.

O antigo rótulo “geração 100%” era impreciso: a avaliação verificava somente o
primeiro token do local correto em 72 prompts. Na geração livre até `EOS`, a V6
obteve `0,00%` de sequências exatas, `61,11%` de término, `41,07%` de tokens
alinhados e `42,13%` de locais alinhados.

## Desempenho do rollback

A arquitetura V6 está aprovada em qualidade e compactação. O caminho integrado
com `torch.sparse.mm` ainda é mais lento que o modelo sem FFN. O próximo trabalho
de otimização deve fundir as projeções COO, Top-12, residual e LayerNorm em um
kernel CUDA próprio para estados contínuos, preservando exatamente os pesos do
checkpoint de rollback.

O benchmark histórico mede apenas o `forward`. A auditoria isolada incluiu
descritores e seleção de candidatos: `0,555 M tokens/s` para 73 tokens/lote 64
e `0,296 M tokens/s` para 512 tokens/lote 16. Na mesma execução, o `forward`
isolado mediu respectivamente `0,955 M tokens/s` e `0,984 M tokens/s`.

## Auditoria de roteamento aprendido

Foi treinado por cinco épocas um roteador com codebook esparso e Top-12, sem
mapas manuais de entidade, objeto ou papel. O treino é supervisionado pela
posição correta do fato, logo não representa aprendizado sem rótulos.

- roteamento Top-1 sem mapas manuais: `100%`;
- leitor original: PPL `2,6347`, local `87,87%`, recuperação `99,95%`;
- leitor Q/K adaptado: PPL `3,6330`, local e recuperação `100%`;
- decisão: não promover a variante, pois o roteamento funcionou, mas PPL,
  qualidade global, geração livre, velocidade e VRAM não foram preservados.

O checkpoint V6 não foi modificado durante a auditoria e depois foi renomeado
como rollback. Checkpoints das cinco épocas do
roteador e das cinco épocas do leitor experimental foram mantidos apenas no
diretório do teste, com relatório em
`resultados/teste_isolado_v6_ultimo.json`.

## Evolução para V6.1

A auditoria da geração livre mostrou que a V6 não tinha sinal explícito de
ordem. A V6.1 preserva atenção Q/K e FFN esparsas, mas acrescenta posição
senoidal fixa, um gate escalar e cinco códigos temporais:

- três códigos identificam o slot causal do fato;
- dois códigos distinguem consulta de objeto e entidade;
- os códigos reutilizam os dois campos de papel do descritor;
- a largura permanece 6 e o código temporal não tem pesos treináveis;
- a arquitetura total passa de 9.621 para 9.632 parâmetros.

O treino usou cinco épocas e salvou cada uma. O checkpoint oficial está em
`modelos/v61_base.pt`; `modelos/v6_rollback.pt` preserva a versão anterior.

### Validação robusta

Quatro sementes inéditas, com 2.000 histórias e 72 gerações cada:

- PPL: `1,5317` média, máximo `1,5322`;
- acurácia de token: `88,06%` média;
- localização e recuperação controladas: `100%`;
- geração livre exata: `93,06%` média, mínimo `86,11%`;
- tokens livres alinhados: `99,17%` média;
- locais livres: `99,54%` média, mínimo `99,07%`.

Três rodadas pareadas do pipeline completo:

- 73 tokens/lote 64: V6 `0,515 M tokens/s`, V6.1
  `0,536 M tokens/s`, razão `104,05%`;
- 512 tokens/lote 16: V6 `0,280 M tokens/s`, V6.1
  `0,279 M tokens/s`, razão `99,72%`;
- VRAM temporária média: igual entre os modelos, `71,22 MiB` e `392,68 MiB`.

Os códigos temporais e as posições fixas são armazenados em cache. Isso removeu
a reconstrução repetida no pipeline sem mudar descritores, logits ou checkpoint.
Como as diferenças de throughput são pequenas, o resultado correto é equivalência
de velocidade, não garantia de aceleração em qualquer GPU.

A V6.1 passou todos os critérios registrados e foi promovida pelo responsável
do projeto a modelo-base oficial. Os slots e papéis ainda codificam a estrutura
fixa deste corpus e precisam ser generalizados antes de afirmar funcionamento
em texto aberto ou quantidade variável de fatos.

O próximo experimento deve variar quantidade e tamanho dos fatos e substituir os
slots absolutos por códigos temporais aprendidos ou relativos. A próxima
otimização de desempenho deve atacar a seleção causal de candidatos com um
índice invertido, preservando os resultados do relatório atual.

Relatório: `resultados/v61_base_validacao.json`.

## Regras de evolução

- não sobrescrever `modelos/v61_base.pt`;
- preservar `modelos/v6_rollback.pt` como controle histórico;
- carregar checkpoints somente com `torch.load(..., weights_only=True)`;
- todo experimento deve usar cinco épocas e salvar cada época;
- comparar PPL, geração, recuperação, velocidade e VRAM;
- qualquer backend novo deve provar equivalência com a V6.1 PyTorch;
- uma nova versão só substitui a V6.1 após superar os critérios registrados.
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

Foi testada uma unica camada causal sem atencao e sem FFN, com processamento
denso dentro de oito bancos pequenos e conectividade esparsa entre bancos. A
camada usa memorias rapida e lenta, combinacoes de pares e ativacao Top-12.

O teste rapido usou 360/72/72 textos controlados de cerca de 110 tokens e cinco
epocas. PPL de teste: 26,460; acuracia de token: 10,80%; recuperacao livre do
local: 0%; preposicao sob teacher forcing: 37,5%; local sob teacher forcing: 0%.
A execucao completa levou 61,64 segundos. A camada demonstrou aprendizado local,
mas nao reteve o fato exato ao longo do texto. Ela permanece isolada e nao
substitui nenhum modelo oficial ou experimental existente.

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

## Correção experimental da geração longa

Na etapa experimental descrita nesta seção, a V6.1 ainda era o único
modelo-base ativo e `modelos/v61_base.pt` não foi modificado. O novo gerador
era mantido isolado para o domínio procedural de relatos condicionados por
pedido. A promoção posterior está registrada na seção seguinte.

### Causa raiz

A célula contextual longa havia sido avaliada principalmente com teacher
forcing. Seu checkpoint da época 5 alcançou PPL `6,4456` e acurácia `39,32%`,
mas a geração livre repetia tokens e podia entrar no bloco de perguntas. Os
caminhos de sucessor exato e replay criavam uma discrepância entre treino e
inferência: durante geração, o token produzido voltava como evidência para a
próxima cópia.

Uma ablação em memória desligou cópia e replay, mas a repetição persistiu,
confirmando que a célula recorrente também não modelava bem a transição longa.
Um controle com a V7 no mesmo corpus caiu para PPL `1,4914`, eliminou os ciclos,
mas ainda trocava campos do prompt devido aos descritores hash fixos.

Também foram encontrados dois defeitos de dados e um de decodificação:

- o corpus antigo gerava `de o`, `de a` e flexão masculina para colaboradoras;
- o bloqueio de trigramas proibia a repetição correta da relação objeto-local;
- a avaliação de problema exigia a forma literal, embora o corpus treinasse
  paráfrases causais.

O corpus legado permanece como padrão da função antiga para preservar seus
checkpoints. O novo treino solicita explicitamente a gramática corrigida.

### Arquitetura corrigida

- 151.025 parâmetros;
- três blocos causais;
- projeções Q/K COO com 24.576 conexões e densidade de 25%;
- atenção causal aprendida Top-32;
- FFN COO com 92.160 conexões, 128 -> 384 -> 128 e Top-64;
- posição senoidal fixa;
- residuais e LayerNorm;
- zero `nn.Linear` interna;
- classificador de vocabulário amarrado ao embedding.

O treino utilizou semente `20260811`, 9.000/600/600 combinações disjuntas,
cinco épocas e cinco checkpoints separados. O checkpoint final está em
`resultados/gerador_esparso_coerente/epoca_05.pt` e é carregado com
`torch.load(..., weights_only=True)`.

### Validação

- PPL de teste: `1,1374`;
- acurácia de token: `93,64%`;
- 24/24 textos completos com sete frases;
- recuperação dos cinco campos do pedido: `100%`;
- consistência objeto-local: `100%`;
- consistência das ações: `100%`;
- vazamento para pergunta/resposta: `0%`;
- repetição média de trigramas: `0,42%`;
- uso de retentativa controlada: 5/24 prompts.

No benchmark pareado, lote 8 x 200 tokens, o modelo contextual anterior mediu
aproximadamente `2.176 tokens/s` e `14,01 MiB` temporários. O gerador corrigido
mediu `86.720 tokens/s` e `11,75 MiB`, razão de `39,86x`.

Todos os critérios de revalidação registrados em
`resultados/gerador_esparso_coerente/relatorio.json` foram aprovados. A geração
antiga tinha somente uma amostra salva, portanto a comparação de geração não é
estatística nem pareada.

### Limite

O resultado é válido para a gramática procedural treinada. A auditoria do
Wikipedia mostrou que cada época V7/V8 via somente 18,04% dos tokens de treino
e os checkpoints terminaram com PPL 153,08 e 144,61. Eles não foram promovidos
e não sustentam uma alegação de linguagem aberta coerente. Uma evolução desse
caminho deve segmentar todo o corpus e usar escala de dados/modelo suficiente,
sempre em experimento separado da base vigente.

## Promoção do Gerador Esparso Coerente

Em 28 de julho de 2026, o usuário autorizou explicitamente a substituição da
V6.1 como modelo-base. A promoção ocorreu somente após uma nova execução
objetiva com corpus ampliado e revalidação independente. A V6.1 não foi
apagada nem sobrescrita: `modelos/v61_base.pt` continua sendo o controle
histórico posicional, ao lado de `modelos/v6_rollback.pt`.

O modelo-base ativo passou a ser:

`modelos/gerador_esparso_base.pt`

SHA-256:

`daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f`

### Configuração promovida

- semente `20260728`;
- 50.000 relatos de treino, 1.000 de validação e 1.000 de teste;
- divisões disjuntas;
- cinco épocas;
- 500 passos por época e 2.500 passos totais;
- lote 100;
- contexto máximo 640 e sequência observada máxima 528;
- vocabulário 332;
- 163.667 parâmetros;
- três blocos causais;
- projeções Q/K COO com 24.576 conexões e densidade 25%;
- atenção causal Top-32;
- FFN COO com 92.160 conexões e ativações Top-64;
- residual, LayerNorm e zero `nn.Linear` interna.

Cada época foi salva separadamente em
`resultados/gerador_esparso_base_50k/epoca_01.pt` até `epoca_05.pt`. O
checkpoint promovido é uma cópia byte a byte da época 5 e foi verificado por
SHA-256.

### Auditoria de dados e geração

A primeira execução ampliada revelou concordância incorreta no próprio corpus,
como `a maquete foi levado`. Uma segunda auditoria encontrou o pronome `eles`
em grupos formados por duas mulheres. Essas execuções foram reprovadas e nunca
foram promovidas.

O corpus definitivo corrige gênero do agente, pronome do objeto, particípio do
objeto de apoio, preposição de destino e concordância do grupo. Uma auditoria
prévia percorreu 5.000 relatos, incluindo 1.126 combinações mulher–mulher, sem
encontrar falhas nesses contratos. O detector de vazamento também passou a
reconhecer apenas os marcadores estruturais `Pergunta:`/`Resposta:`, sem
confundir a palavra comum `resposta` com um bloco Q&A.

### Resultado revalidado

- PPL de teste: `1,0505`;
- acurácia de token: `97,53%`;
- 24/24 gerações aprovadas;
- 24 frases por geração;
- mínimo de `2.802` e média de `2.829,9` caracteres;
- recuperação dos cinco campos: `100%`;
- consistência objeto-local: `100%`;
- consistência das ações: `100%`;
- vazamento Q&A: `0%`;
- uso de retentativa: `0%`;
- repetição média de trigramas: `1,13%`;
- nenhum trigrama apareceu mais de duas vezes.

O pico de treino foi aproximadamente `1.897,5 MiB`. Na comparação pareada
8 x 200 tokens, o contextual anterior mediu `2.169 tokens/s` e `14,01 MiB`
temporários; a nova base mediu `92.757 tokens/s` e `11,75 MiB`, razão de
`42,77x`.

Todos os nove critérios de
`resultados/gerador_esparso_base_50k/relatorio.json` foram aprovados após
recarga com `torch.load(..., weights_only=True)`.

### Limite da promoção

A promoção vale para geração procedural condicionada no formato documentado.
Os 50 mil relatos são sintéticos e não demonstram linguagem aberta geral. Os
resultados Wikipedia anteriores continuam reprovados e não fazem parte do
checkpoint-base.
