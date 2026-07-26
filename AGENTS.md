# Instrucoes do projeto V6.1

Autor: Paulo Augusto  
Ano: 2026

Antes de modificar, treinar ou otimizar o modelo, leia integralmente:

`DOCUMENTO_MODELO_V61.md`

Regras obrigatorias:

- a V6.1 posicional e o unico modelo-base ativo;
- o checkpoint oficial e `modelos/v61_base.pt`;
- nunca sobrescrever o checkpoint oficial durante experimentos;
- preservar `modelos/v6_rollback.pt` como controle historico;
- preservar atencao causal Q/K esparsa, FFN COO, residual e normalizacao;
- carregar checkpoints com `torch.load(..., weights_only=True)`;
- nunca habilitar `weights_only=False` para checkpoints nao confiaveis;
- todo experimento de treino deve usar cinco epocas;
- salvar cada epoca em checkpoint separado;
- comparar PPL, geracao livre, recuperacao, velocidade e VRAM;
- qualquer kernel novo deve provar equivalencia com a referencia PyTorch;
- atualizar README, documento tecnico, status e testes apos mudancas;
- usar o ambiente `venv_cuda` existente sem instalar dependencias desnecessarias;
- uma versao futura so substitui a V6.1 apos validacao objetiva e autorizacao.
