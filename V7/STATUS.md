# Estado da V7.3

## Linha mantida

Somente a V7.3 textual BPE-8192 está ativa nesta pasta. A versão do código é 2.8.0.

Checkpoint operacional:

`modelos/gerador_esparso_v73_bpe8192_refino_eos3_ul005_topologia.pt`

SHA-256:

`23af2ea91fb62153559edb9fb2ca3bf33655f1f6d3ab125751f9e33afd3339ad`

O checkpoint anterior com topologia permanece preservado com SHA-256 `0fa1e7fc43dcce7fd7d4561c941c3ee4c3d64ee6a2b41b5e199774a2927c21fa`. O checkpoint original também permanece intacto, com SHA-256 `151e3bed5c9d2c6f366fa4ae5799de8577a4baa41f82b381dbb5039fbfa43b65`.

## Correções da versão 2.8.0

- executor aplica o template `usuario/assistente` e imprime somente a resposta;
- flags de arquitetura configuram o modelo; a flag multihead inválida foi removida;
- checkpoints novos incluem versão e hash da topologia COO;
- carregador rejeita topologia ausente ou divergente por padrão;
- auditoria CUDA reconhece que o BPE-8192 não usa o caminho incremental totalmente fundido;
- auditoria declara a memória quadrática do forward vetorizado em lote;
- alfabeto byte-level GPT-2 legado corrigido;
- caminhos fixos em `D:\AI-Dados` substituídos por CLI e variáveis de ambiente;
- critérios mínimos de qualidade generativa adicionados;
- `pyproject.toml` e `.gitignore` adicionados;
- identidade do pacote limitada à V7.3.

## Resultado histórico do treino

- Época selecionada: 5
- Passos totais: 15.000
- Loss de treino da resposta: 3,988377
- Loss de validação: 4,069222
- PPL: 58,511445
- BPB publicado: 1,3635424677
- BPB corrigido: aproximadamente 1,3635468168
- Vazão média: 215.120,60 tokens/s
- Tempo puro de treino: 571,22 segundos
- Pico de VRAM observado na época 5: 608,62 MiB

## Validação do projeto

A suíte ampliada cobre prompt real, topologia adulterada, opções arquiteturais, BPE legado, CPU e CUDA. Em 3 de agosto de 2026, 43 testes passaram em 11,88 s.

## Refino promovido

O refino de três épocas e 9.000 passos foi validado novamente pelo avaliador 2.8.0 em 50 prompts. A amostragem controlada obteve repetição média 0,1497, `distinct_2` 0,9178, repetição consecutiva média 1,32 e taxa de EOS 0,74; o resultado final foi `aprovado`.

A aprovação cobre os critérios mecânicos de diversidade e término. A correção factual e a coerência ainda são experimentais, e o modo greedy puro continua inadequado por repetição elevada.
