# Instrucoes do projeto - Gerador Esparso Coerente

Autor: Paulo Augusto  
Ano: 2026

Antes de modificar, treinar ou otimizar o modelo, leia integralmente:

`DOCUMENTO_GERADOR_ESPARSO.md`

Regras obrigatorias:

- o Gerador Esparso Coerente e o unico modelo oficial do projeto;
- o runtime-base ativo e a V6.2 com cache causal, CSR de inferencia e gates
  cacheados;
- a baseline densa e somente um controle experimental autorizado;
- nunca promover a baseline densa sem nova autorizacao explicita;
- o checkpoint oficial e `modelos/gerador_esparso_base.pt`;
- nunca sobrescrever o checkpoint oficial durante experimentos;
- preservar atencao causal Q/K esparsa, FFN COO, residual e normalizacao;
- carregar checkpoints com `torch.load(..., weights_only=True)`;
- nunca habilitar `weights_only=False` para checkpoints nao confiaveis;
- todo experimento de treino deve usar cinco epocas;
- salvar cada epoca em checkpoint separado;
- comparar PPL, geracao livre, recuperacao, velocidade e VRAM;
- separar throughput de forward e velocidade autorregressiva percebida;
- tratar o relatorio JSON como fonte unica e sincronizar metricas documentais;
- salvar no relatorio todas as 24 geracoes usadas na aprovacao;
- rejeitar prompts publicos fora do formato ou com tokens desconhecidos;
- qualquer kernel novo deve provar equivalencia com a referencia PyTorch;
- atualizar README, documento tecnico, status e testes apos mudancas;
- usar o ambiente CUDA existente sem instalar dependencias desnecessarias;
- outra arquitetura alem da baseline so pode ser adicionada com autorizacao
  explicita.
