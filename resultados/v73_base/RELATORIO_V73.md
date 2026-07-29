# Comparação auditável — V6.2, V7.3 e denso

Autor: Paulo Augusto
Ano: 2026

Benchmark com 2 aquecimentos e 5 repetições. A mediana é o valor principal; média e desvio-padrão permanecem no JSON.

| Métrica | V6.2 | V7.3 | Denso |
|---|---:|---:|---:|
| Época | 5 | 2 | 5 |
| Parâmetros | 163.667 | 165.443 | 163.003 |
| PPL teste | 1,050526 | 1,054314 | 1,049220 |
| Aprovação — modelo puro | 21/24 | 9/24 | 22/24 |
| Modelo puro — objeto/local | 88% | 83% | 92% |
| Modelo puro — ações | 100% | 88% | 100% |
| Modelo puro — concordância | 100% | 42% | 100% |
| Aprovação — greedy restrito | 24/24 | 24/24 | 24/24 |
| Aprovação — sistema completo | 24/24 | 24/24 | 24/24 |
| Mudanças de argmax pelas regras | 3 | 24 | 3 |
| Tokens bloqueados pelas regras | 179.733 | 177.103 | 168.888 |
| Forward — mediana, média ± DP | 143.184,54 tok/s (média 139.095,63 ± 6.994,69) | 273.029,23 tok/s (média 271.942,27 ± 2.175,18) | 1.695.953,89 tok/s (média 1.702.406,96 ± 16.872,68) |
| Geração — mediana, média ± DP | 311,01 tok/s (média 309,32 ± 12,45) | 998,26 tok/s (média 973,31 ± 54,48) | 505,69 tok/s (média 504,57 ± 3,19) |
| Primeiro token — mediana | 4,17 ms | 5,93 ms | 1,78 ms |
| VRAM forward — mediana | 83,42 MiB | 102,07 MiB | 46,51 MiB |
| Checkpoint | 678.219 bytes | 691.289 bytes | 674.635 bytes |

## Leitura correta da qualidade

A linha **modelo puro** usa argmax sem as regras de repetição, gênero, ações ou objeto de apoio. A linha **greedy restrito** mede uma única geração causal com as regras. A linha **sistema completo** inclui o validador e eventuais retentativas.

Assim, 24/24 do sistema não é apresentado como capacidade exclusiva dos pesos aprendidos.

## Conclusão

A V7.3 foi **1.97×** mais rápida que o denso na mediana da geração.
O denso foi **6.21×** mais rápido na mediana do forward.

## Reprodução

```powershell
python comparar_v73.py --repeticoes 5 --aquecimentos 2
python comparar_v73.py --somente-documentos
```
