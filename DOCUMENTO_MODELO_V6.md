# Modelo-base V6

Autor: Paulo Augusto  
Ano: 2026

## Decisão

A V6 é oficialmente o único modelo-base do projeto. Sua arquitetura canônica
combina atenção causal esparsa e FFN bloco-esparsa.

Checkpoint oficial:

`modelos/v6_base.pt`

## Arquitetura

```text
tokens
  -> embedding e estado local
  -> atenção causal com Q/K esparsos e Top-24
  -> residual e LayerNorm
  -> FFN COO 48 -> 144 -> 48 com ReLU Top-12
  -> residual aprendível e LayerNorm
  -> classificador de tokens
```

A FFN não contém `nn.Linear` nem matriz densa mascarada. Suas duas projeções
possuem 3.456 conexões e densidade estrutural de 25%. O modelo completo possui
9.621 parâmetros no corpus controlado atual.

## Checkpoint canônico

- semente: `20260727`;
- época: `5`;
- PPL inédita de teste: `1,9077`;
- localização da resposta: `100%`;
- recuperação causal: `100%`;
- primeiro token do local na avaliação controlada: `100%`.

## Validação

O protocolo oficial usa três sementes, cinco épocas e salva cada época
separadamente. A média da V6 foi:

- PPL: `1,9196`;
- acurácia de token: `77,41%`;
- localização, recuperação e primeiro token controlado: `100%`;
- PPL sem a FFN após ablação: `4,9899`.

A suíte de vinte e três testes verifica causalidade, topologia COO, gradientes esparsos,
determinismo, recarga estrita do checkpoint e consistência da documentação.

O antigo rótulo “geração 100%” era impreciso: a avaliação verificava somente o
primeiro token do local correto em 72 prompts. Na geração livre até `EOS`, a V6
obteve `0,00%` de sequências exatas, `61,11%` de término, `41,07%` de tokens
alinhados e `42,13%` de locais alinhados.

## Estado de desempenho

A arquitetura V6 está aprovada em qualidade e compactação. O caminho integrado
com `torch.sparse.mm` ainda é mais lento que o modelo sem FFN. O próximo trabalho
de otimização deve fundir as projeções COO, Top-12, residual e LayerNorm em um
kernel CUDA próprio para estados contínuos, preservando exatamente os pesos do
checkpoint canônico.

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

O checkpoint canônico não foi modificado. Checkpoints das cinco épocas do
roteador e das cinco épocas do leitor experimental foram mantidos apenas no
diretório do teste, com relatório em
`resultados/teste_isolado_v6_ultimo.json`.

## V6.1 posicional candidata

A auditoria da geração livre mostrou que a V6 não tinha sinal explícito de
ordem. A V6.1 preserva atenção Q/K e FFN esparsas, mas acrescenta posição
senoidal fixa, um gate escalar e cinco códigos temporais:

- três códigos identificam o slot causal do fato;
- dois códigos distinguem consulta de objeto e entidade;
- os códigos reutilizam os dois campos de papel do descritor;
- a largura permanece 6 e o código temporal não tem pesos treináveis;
- a arquitetura total passa de 9.621 para 9.632 parâmetros.

O treino isolado usou cinco épocas e salvou cada uma. O checkpoint aprovado está
em `modelos/v61_candidata.pt`; `modelos/v6_base.pt` continua intacto para
controle e rollback.

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

A V6.1 foi aprovada como candidata porque todos os critérios registrados no
relatório passaram. Ela ainda não substitui automaticamente a V6: os slots e
papéis codificam a estrutura fixa deste corpus e precisam ser generalizados
antes de afirmar funcionamento em texto aberto ou quantidade variável de fatos.

O próximo experimento deve variar quantidade e tamanho dos fatos e substituir os
slots absolutos por códigos temporais aprendidos ou relativos. A próxima
otimização de desempenho deve atacar a seleção causal de candidatos com um
índice invertido, preservando os resultados do relatório atual.

Relatório: `resultados/v61_candidata_validacao.json`.

## Regras de evolução

- não sobrescrever `modelos/v6_base.pt`;
- todo experimento deve usar cinco épocas e salvar cada época;
- comparar PPL, geração, recuperação, velocidade e VRAM;
- qualquer backend novo deve provar equivalência com a V6 PyTorch;
- uma nova versão só substitui a V6 após superar os critérios registrados aqui.
