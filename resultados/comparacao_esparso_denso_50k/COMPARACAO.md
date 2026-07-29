# Comparação justa: esparso × denso

Autor: Paulo Augusto
Ano: 2026

Mesmos dados, ordem de lotes, cinco épocas, 500 passos por época, lote 100, AdamW, agenda, semente e decodificador.

| Métrica | Esparso | Denso | Vencedor |
|---|---:|---:|---|
| Parâmetros | 163.667 | 163.003 | denso |
| PPL de teste | 1,050526 | 1,049220 | denso |
| Acurácia de token | 97,53% | 97,59% | denso |
| Aprovação de geração | 100,00% | 100,00% | empate |
| Recuperação dos campos | 100,00% | 100,00% | empate |
| Repetição média de trigramas | 1,17% | 1,03% | denso |
| Taxa de retentativa | 12,50% | 8,33% | denso |
| Tempo de treino (s) | 1.663,30 | 281,09 | denso |
| Pico de VRAM no treino (MiB) | 1.898,33 | 950,36 | denso |
| Forward (tokens/s) | 132.431,00 | 1.730.921,20 | denso |
| Pico de VRAM no forward (MiB) | 82,77 | 43,17 | denso |
| Geração real (tokens/s) | 296,01 | 534,80 | denso |
| Primeiro token (ms) | 4,95 | 1,98 | denso |
| Checkpoint (bytes) | 678.219 | 674.635 | denso |

## Conclusão automática

A vantagem prática esparsa não foi sustentada de forma majoritária neste protocolo.

A baseline densa treinou 5,92× mais rápido, executou o forward 13,07× mais rápido e gerou texto 1,81× mais rápido.
O denso usou 47,84% menos VRAM no forward; o denso usou 49,94% menos VRAM no treino.

O resultado vale para este domínio, hardware e orçamento. Uma única configuração não prova superioridade universal.
