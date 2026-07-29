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
- Atenção: Top-32 global em blocos de 32 consultas, sem matriz completa
  `[lote, tempo, tempo]`; custo aritmético ainda quadrático.
- COO: índices consolidados uma vez e matriz de inferência reutilizada com
  invalidação segura.
- `nn.Linear` dentro do núcleo: zero.
- Componentes densos: embedding/classificador compartilhado, LayerNorm,
  residuais e escores temporários de cada bloco de atenção.
- Dados: 50.000/1.000/1.000 relatos disjuntos.
- Treino: cinco épocas, 500 passos por época e 2.500 passos totais.
- PPL final: 1,0505.
- Acurácia de token: 97,53%.
- Geração livre: 24/24 textos aprovados.
- Gerações auditáveis salvas: 24/24.
- Tamanho mínimo: 2.806 caracteres.
- Recuperação dos campos: 100%.
- Consistência objeto/local e ações: 100%.
- Retentativas: 3/24 (12,5%).
- Vazamento Q&A: 0%.
- Validador: 2.0.0, com varredura por cláusulas e suíte adversarial aprovada.
- Entrada pública: formato fechado e rejeição explícita de tokens `<unk>`.
- Forward paralelo: 117.980,83 tokens/s e 88,46 MiB de VRAM.
- Geração autorregressiva: 97,37 tokens/s, primeiro token em 5,57 ms e
  relato completo em 5,02 s.
- Ambiente revalidado: Windows 11, Python 3.14.0, PyTorch 2.11.0+cu128,
  CUDA 12.8 e NVIDIA GeForce RTX 3050.
- Relatório: `resultados/gerador_esparso_base_50k/relatorio.json`.
- Situação: aprovado e único modelo do projeto.
