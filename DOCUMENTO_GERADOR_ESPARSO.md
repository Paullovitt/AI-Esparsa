# Documento técnico do Gerador Esparso Coerente

Autor: Paulo Augusto
Ano: 2026

## Escopo

O Gerador Esparso Coerente é a única arquitetura mantida neste repositório. Seu
objetivo é produzir relatos procedurais longos condicionados por campos
explícitos de um pedido.

Checkpoint oficial:

`modelos/gerador_esparso_base.pt`

## Topologia

O modelo possui dimensão 128, três blocos causais, contexto máximo de 640
tokens e 163.667 parâmetros. Cada bloco contém:

1. projeções Q/K treináveis em COO;
2. seleção causal Top-32;
3. residual e LayerNorm;
4. FFN COO 128-384-128 com GELU e Top-64;
5. residual com gate treinável e LayerNorm.

As projeções Q/K e FFN têm 25% de densidade estrutural. Não existe `nn.Linear`
densa no núcleo. O embedding/classificador compartilhado é uma matriz
treinável densa; normalização, residuais, softmax e escores temporários também
operam sobre tensores densos. Por isso, o modelo é descrito como tendo núcleo
estruturalmente esparso, não como 100% esparso.

## Causalidade

A máscara impede acesso a tokens futuros. A seleção Top-K ocorre somente entre
posições causais válidas. O teste automatizado compara prefixos antes e depois
de alterações em tokens futuros.

## Dados e tokenização

O corpus procedural é gerado deterministicamente e dividido sem sobreposição:

- treino: 50.000 relatos;
- validação: 1.000 relatos;
- teste: 1.000 relatos;
- 24 frases por relato extenso;
- vocabulário: 332 tokens;
- sequência máxima observada: 528 tokens.

O tokenizador separa palavras e pontuação e é reconstruído pelo vocabulário
persistido no checkpoint.

## Treinamento oficial

- semente: 20260728;
- cinco épocas;
- lote: 100;
- 500 passos por época;
- 2.500 atualizações totais;
- tempo total: 1.766,86 segundos;
- pico de VRAM: 1.898,33 MiB.

Cada época possui um checkpoint separado. Todos são carregáveis com
`torch.load(..., weights_only=True)`.

## Validação

- PPL final: 1,0505;
- acurácia de token: 97,53%;
- 24/24 gerações aprovadas;
- mínimo de 2.802 caracteres;
- média de 2.829,9 caracteres;
- recuperação dos campos: 100%;
- consistência objeto/local e ações: 100%;
- vazamento Q&A: 0%;
- repetição média de trigramas: 1,13%;
- frequência máxima de um trigrama: duas ocorrências.

## Segurança de checkpoint

O executor e o validador usam `weights_only=True` e recarga estrita. O script
de promoção recusa sobrescrever um checkpoint oficial existente e exige cinco
épocas, 50.000 amostras, 500 passos por época e todos os critérios de
revalidação aprovados.

## Limite de validade

As métricas comprovam desempenho no domínio procedural treinado. Elas não
demonstram linguagem aberta geral. O cálculo dos escores de atenção também
continua denso antes da seleção Top-K.
