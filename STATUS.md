# Status do Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

- Arquiteturas mantidas: 1.
- Modelo: Gerador Esparso Coerente.
- Checkpoint oficial: `modelos/gerador_esparso_base.pt`.
- SHA-256: `daba162081b351fe44bd9179c7a4f5ec374e691841d12774789916f846ac215f`.
- Parâmetros: 163.667.
- Contexto máximo: 640 tokens.
- Q/K: COO, Top-32 e 25% de densidade.
- FFN: COO 128-384-128, Top-64 e 25% de densidade.
- `nn.Linear` dentro do núcleo: zero.
- Componentes densos: embedding/classificador compartilhado, LayerNorm,
  residuais e escores temporários de atenção.
- Dados: 50.000/1.000/1.000 relatos disjuntos.
- Treino: cinco épocas, 500 passos por época e 2.500 passos totais.
- PPL final: 1,0505.
- Acurácia de token: 97,53%.
- Geração livre: 24/24 textos aprovados.
- Tamanho mínimo: 2.802 caracteres.
- Recuperação dos campos: 100%.
- Consistência objeto/local e ações: 100%.
- Vazamento Q&A: 0%.
- Relatório: `resultados/gerador_esparso_base_50k/relatorio.json`.
- Situação: aprovado e único modelo do projeto.
