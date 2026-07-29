# Instruções do projeto — AI-Esparsa

Autor: Paulo Augusto  
Ano: 2026

Antes de modificar, treinar ou otimizar, leia integralmente:

`DOCUMENTO_GERADOR_ESPARSO.md`

Regras obrigatórias:

- a V7.3 é a base textual preservada;
- a V8 é a arquitetura multimodal experimental ativa;
- a baseline densa é somente controle experimental;
- nunca promover a baseline densa sem autorização explícita;
- o checkpoint textual é `modelos/gerador_esparso_v73_base.pt`;
- o controle denso é `modelos/gerador_denso_controle.pt`;
- nunca sobrescrever checkpoints preservados durante experimentos;
- preservar Q/K COO, atenção causal Top-K, FFN combinatória, residuais e
  normalização;
- preservar a projeção visual COO e a fusão por prefixo causal da V8;
- carregar checkpoints com `torch.load(..., weights_only=True)`;
- nunca habilitar `weights_only=False` para checkpoints não confiáveis;
- nenhum dataset deve ser versionado ou copiado para checkpoint;
- manifestos devem usar caminhos relativos e splits sem vazamento;
- todo experimento promovível deve usar cinco épocas;
- salvar cada época em checkpoint separado;
- comparar PPL, geração, fidelidade visual, velocidade e VRAM;
- separar throughput de forward e velocidade autorregressiva;
- tratar o relatório JSON como fonte única das métricas;
- qualquer kernel novo deve provar equivalência com a referência PyTorch;
- atualizar README, documento técnico, status e testes após mudanças;
- usar o ambiente CUDA existente sem instalar dependências desnecessárias;
- outra arquitetura exige autorização explícita;
- uma atualização solicitada para GitHub só termina após chegar a `main` e o
  README padrão remoto ser verificado.
