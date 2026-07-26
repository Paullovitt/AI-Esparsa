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
- geração controlada: `100%`.

## Validação

O protocolo oficial usa três sementes, cinco épocas e salva cada época
separadamente. A média da V6 foi:

- PPL: `1,9196`;
- acurácia de token: `77,41%`;
- localização, recuperação e geração: `100%`;
- PPL sem a FFN após ablação: `4,9899`.

A suíte de nove testes verifica causalidade, topologia COO, gradientes esparsos,
determinismo, recarga estrita do checkpoint e consistência da documentação.

## Estado de desempenho

A arquitetura V6 está aprovada em qualidade e compactação. O caminho integrado
com `torch.sparse.mm` ainda é mais lento que o modelo sem FFN. O próximo trabalho
de otimização deve fundir as projeções COO, Top-12, residual e LayerNorm em um
kernel CUDA próprio para estados contínuos, preservando exatamente os pesos do
checkpoint canônico.

## Regras de evolução

- não sobrescrever `modelos/v6_base.pt`;
- todo experimento deve usar cinco épocas e salvar cada época;
- comparar PPL, geração, recuperação, velocidade e VRAM;
- qualquer backend novo deve provar equivalência com a V6 PyTorch;
- uma nova versão só substitui a V6 após superar os critérios registrados aqui.
