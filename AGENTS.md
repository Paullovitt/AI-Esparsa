# Instrucoes do projeto - Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

Antes de modificar, treinar ou otimizar o modelo, leia integralmente:

`DOCUMENTO_GERADOR_ESPARSO.md`

Regras obrigatorias:

- o Gerador Esparso Coerente e o unico modelo oficial do projeto;
- a V7.3 CUDA e a base de desenvolvimento ativa;
- a V6.2 com cache causal, CSR e gates cacheados e a base estavel preservada;
- a baseline densa e somente um controle experimental autorizado;
- nunca promover a baseline densa sem nova autorizacao explicita;
- o checkpoint oficial e `modelos/gerador_esparso_base.pt`;
- o checkpoint da base de desenvolvimento e
  `modelos/gerador_esparso_v73_base.pt`;
- manter somente V6.2, V7.3 e a baseline densa;
- nao reintroduzir V7, V7.1, V7.2 nem o candidato V7.3 BF16;
- nunca sobrescrever o checkpoint oficial durante experimentos;
- preservar atencao causal Q/K esparsa, FFN COO, residual e normalizacao;
- carregar checkpoints com `torch.load(..., weights_only=True)`;
- nunca habilitar `weights_only=False` para checkpoints nao confiaveis;
- todo experimento promovivel deve usar cinco epocas;
- a base V7.3 atual possui duas epocas FP32 e permanece inelegivel para
  promocao formal ate completar um treino controlado de cinco epocas;
- salvar cada epoca em checkpoint separado;
- comparar PPL, geracao livre, recuperacao, velocidade e VRAM;
- separar geracao do modelo puro, greedy restrito e sistema completo;
- registrar tokens bloqueados e mudancas de argmax por regra do decodificador;
- separar throughput de forward e velocidade autorregressiva percebida;
- usar aquecimento e no minimo cinco repeticoes nos benchmarks publicados;
- registrar mediana, media, desvio-padrao, amostras e estado da GPU;
- tratar o relatorio JSON como fonte unica e sincronizar metricas documentais;
- salvar no relatorio todas as 24 geracoes usadas na aprovacao;
- rejeitar prompts publicos fora do formato ou com tokens desconhecidos;
- qualquer kernel novo deve provar equivalencia com a referencia PyTorch;
- atualizar README, documento tecnico, status e testes apos mudancas;
- usar o ambiente CUDA existente sem instalar dependencias desnecessarias;
- outra arquitetura alem da baseline so pode ser adicionada com autorizacao
  explicita.
