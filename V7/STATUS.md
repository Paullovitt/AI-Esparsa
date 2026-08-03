# Estado da V7.3

## Linha mantida

Somente a V7.3 textual BPE-8192 está ativa nesta pasta. A versão do código é 2.8.0.

Checkpoint operacional:

`modelos/gerador_esparso_v73_bpe8192_5x3000_topologia.pt`

SHA-256:

`0fa1e7fc43dcce7fd7d4561c941c3ee4c3d64ee6a2b41b5e199774a2927c21fa`

O checkpoint original permanece preservado, sem alteração, com SHA-256 `151e3bed5c9d2c6f366fa4ae5799de8577a4baa41f82b381dbb5039fbfa43b65`.

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

A suíte ampliada cobre prompt real, topologia adulterada, opções arquiteturais, BPE legado, CPU e CUDA. Em 3 de agosto de 2026, 43 testes passaram em 11,84 s.

## Pendência científica

Os defeitos de integração e auditoria foram corrigidos. A qualidade aprendida do checkpoint não pode ser reparada sem novo treinamento: a geração ainda pode ser repetitiva ou pouco coerente e deve ser considerada experimental até passar nos critérios do avaliador.
