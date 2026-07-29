# Status do Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

- Modelos oficiais: 1.
- Modelo oficial: Gerador Esparso Coerente.
- Controle experimental autorizado: baseline densa de 163.003 parâmetros.
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
<!-- metricas-desempenho:inicio -->
- Forward paralelo: 111.834,05 tokens/s e 88,46 MiB de VRAM.
- Geração autorregressiva: 94,48 tokens/s, primeiro token em 5,58 ms e
  relato completo em 5,18 s.
<!-- metricas-desempenho:fim -->
- Ambiente revalidado: Windows 11, Python 3.14.0, PyTorch 2.11.0+cu128,
  CUDA 12.8 e NVIDIA GeForce RTX 3050.
- Relatório: `resultados/gerador_esparso_base_50k/relatorio.json`.
- Comparação equivalente:
  - esparso: 163.667 parâmetros, PPL 1,050526 e 24/24 gerações;
  - denso: 163.003 parâmetros, PPL 1,049220 e 24/24 gerações;
  - denso: treino 5,92×, forward 12,67× e geração 5,49× mais rápidos;
  - denso: 49,94% menos VRAM no treino;
  - esparso: 26,29% menos VRAM no forward;
  - conclusão: vantagem prática esparsa não sustentada neste protocolo.
- Revalidação densa: aprovada, cinco checkpoints `weights_only=True`, PPL
  reproduzida e 24/24 gerações novamente aprovadas.
- Relatório comparativo:
  `resultados/comparacao_esparso_denso_50k/comparacao.json`.
- Situação: esparso aprovado como modelo oficial; baseline densa mantida
  apenas para controle e investigação.
