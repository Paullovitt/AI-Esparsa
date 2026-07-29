# Status do projeto

Autor: Paulo Augusto

Ano: 2026

Data: 29 de julho de 2026

## Modelos

- V7.3 textual: preservada e carregável.
- V8 multimodal: código, dados, treino e executor implementados; não treinada.
- Controle denso textual: checkpoint final preservado.

## V8 montada

- entrada RGB 64 × 64;
- 64 patches de 8 × 8;
- projeção visual COO, 25% de densidade;
- prefixo visual causal;
- backbone combinatório V7.3;
- 172.228 parâmetros com o vocabulário atual;
- 6.785 parâmetros adicionais;
- 56 tensores transferíveis da V7.3;
- nenhuma linear densa no encoder visual.

## Validação estrutural

- compilação Python: aprovada;
- 37 testes executados;
- 35 aprovados;
- 2 testes CUDA ignorados na execução CPU;
- imagem altera logits: aprovado;
- causalidade com prefixo visual: aprovada;
- gradientes visuais e do roteador: aprovados;
- backend vetorizado preserva parâmetros e gradientes: aprovado;
- manifesto e segurança de caminhos: aprovados.

## Dados e caches

- nenhum dataset está dentro do projeto;
- corpus procedural anterior e seus resultados foram removidos;
- caches Python, build local e cache compilado do kernel foram apagados;
- `dados/`, `datasets/`, `.cache/` e `cache/` estão ignorados.

## Próxima etapa

Escolher e preparar o novo dataset multimodal, criar os manifestos separados,
treinar por cinco épocas e então medir qualidade, generalização, velocidade,
VRAM, diversidade de roteamento e ablações.

Não há afirmação de superioridade da V8 enquanto esse experimento não existir.
