#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int DIMENSAO = 128;
constexpr int GRUPOS_ATIVOS = 4;
constexpr int UNIDADES_GRUPO = 24;
constexpr int ATIVACOES = GRUPOS_ATIVOS * UNIDADES_GRUPO;
constexpr int TOP_K = 64;
constexpr int THREADS = 128;
constexpr int FAN_IN_QK = 32;
constexpr int MAXIMO_CONTEXTO = 640;
constexpr int TOP_K_ATENCAO = 32;
constexpr int MAXIMO_VOCABULARIO = 512;

__device__ __forceinline__ float gelu_exata(float valor) {
  return 0.5f * valor
      * (1.0f + erff(valor * 0.70710678118654752440f));
}

__global__ void ffn_roteada_kernel(
    const float* __restrict__ entrada,
    const int64_t* __restrict__ rotas,
    const float* __restrict__ gates,
    const float* __restrict__ pesos_1,
    const float* __restrict__ bias_1,
    const float* __restrict__ pesos_2,
    const float* __restrict__ bias_2,
    float* __restrict__ saida,
    int quantidade_tokens) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  if (token >= quantidade_tokens || thread >= THREADS) {
    return;
  }

  __shared__ float ordenacao_valores[THREADS];
  __shared__ int ordenacao_indices[THREADS];
  __shared__ float ativacoes_podadas[ATIVACOES];

  if (thread < ATIVACOES) {
    const int posicao_rota = thread / UNIDADES_GRUPO;
    const int unidade = thread % UNIDADES_GRUPO;
    const int grupo = static_cast<int>(
        rotas[token * GRUPOS_ATIVOS + posicao_rota]);
    const int deslocamento_peso =
        (grupo * UNIDADES_GRUPO + unidade) * DIMENSAO;
    float acumulado = bias_1[grupo * UNIDADES_GRUPO + unidade];
    #pragma unroll 4
    for (int entrada_indice = 0;
         entrada_indice < DIMENSAO;
         ++entrada_indice) {
      acumulado +=
          entrada[token * DIMENSAO + entrada_indice]
          * pesos_1[deslocamento_peso + entrada_indice];
    }
    const float ativacao = gelu_exata(acumulado);
    ordenacao_valores[thread] = ativacao;
    ordenacao_indices[thread] = thread;
    ativacoes_podadas[thread] = 0.0f;
  } else {
    ordenacao_valores[thread] = -3.402823466e+38F;
    ordenacao_indices[thread] = thread;
  }
  __syncthreads();

  // A rede de ordenacao mantém o Top-64 inteiro dentro do mesmo bloco CUDA.
  for (int tamanho = 2; tamanho <= THREADS; tamanho <<= 1) {
    for (int distancia = tamanho >> 1; distancia > 0; distancia >>= 1) {
      const int parceiro = thread ^ distancia;
      if (parceiro > thread) {
        const bool crescente = (thread & tamanho) == 0;
        const float valor_atual = ordenacao_valores[thread];
        const float valor_parceiro = ordenacao_valores[parceiro];
        if ((valor_atual > valor_parceiro) == crescente) {
          ordenacao_valores[thread] = valor_parceiro;
          ordenacao_valores[parceiro] = valor_atual;
          const int indice_atual = ordenacao_indices[thread];
          ordenacao_indices[thread] = ordenacao_indices[parceiro];
          ordenacao_indices[parceiro] = indice_atual;
        }
      }
      __syncthreads();
    }
  }

  if (thread >= THREADS - TOP_K) {
    const int indice_original = ordenacao_indices[thread];
    if (indice_original < ATIVACOES) {
      ativacoes_podadas[indice_original] = ordenacao_valores[thread];
    }
  }
  __syncthreads();

  float acumulado_saida = bias_2[thread];
  #pragma unroll
  for (int posicao_rota = 0;
       posicao_rota < GRUPOS_ATIVOS;
       ++posicao_rota) {
    const int grupo = static_cast<int>(
        rotas[token * GRUPOS_ATIVOS + posicao_rota]);
    const float gate = gates[token * GRUPOS_ATIVOS + posicao_rota];
    const int deslocamento_peso =
        (grupo * DIMENSAO + thread) * UNIDADES_GRUPO;
    float soma_grupo = 0.0f;
    #pragma unroll
    for (int unidade = 0; unidade < UNIDADES_GRUPO; ++unidade) {
      soma_grupo +=
          ativacoes_podadas[
              posicao_rota * UNIDADES_GRUPO + unidade]
          * pesos_2[deslocamento_peso + unidade];
    }
    acumulado_saida += gate * soma_grupo;
  }
  saida[token * DIMENSAO + thread] = acumulado_saida;
}

__device__ __forceinline__ float sigmoide(float valor) {
  return 1.0f / (1.0f + __expf(-valor));
}

__global__ void normalizar_qk_kernel(
    const float* __restrict__ entrada,
    const float* __restrict__ peso_normalizacao,
    const float* __restrict__ bias_normalizacao,
    float epsilon,
    const int64_t* __restrict__ indices_q,
    const float* __restrict__ valores_q,
    const float* __restrict__ bias_q,
    const int64_t* __restrict__ indices_k,
    const float* __restrict__ valores_k,
    const float* __restrict__ bias_k,
    float* __restrict__ normalizados,
    float* __restrict__ consultas,
    float* __restrict__ chaves,
    int quantidade_tokens) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  if (token >= quantidade_tokens || thread >= DIMENSAO) {
    return;
  }
  __shared__ float reducao[DIMENSAO];
  __shared__ float valores_normalizados[DIMENSAO];

  const float valor = entrada[token * DIMENSAO + thread];
  reducao[thread] = valor;
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float media = reducao[0] / static_cast<float>(DIMENSAO);
  const float diferenca = valor - media;
  reducao[thread] = diferenca * diferenca;
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float inverso_desvio = rsqrtf(
      reducao[0] / static_cast<float>(DIMENSAO) + epsilon);
  const float normalizado =
      diferenca * inverso_desvio * peso_normalizacao[thread]
      + bias_normalizacao[thread];
  valores_normalizados[thread] = normalizado;
  normalizados[token * DIMENSAO + thread] = normalizado;
  __syncthreads();

  float consulta = bias_q[thread];
  float chave = bias_k[thread];
  const int deslocamento = thread * FAN_IN_QK;
  #pragma unroll
  for (int indice = 0; indice < FAN_IN_QK; ++indice) {
    consulta +=
        valores_normalizados[indices_q[deslocamento + indice]]
        * valores_q[deslocamento + indice];
    chave +=
        valores_normalizados[indices_k[deslocamento + indice]]
        * valores_k[deslocamento + indice];
  }
  consultas[token * DIMENSAO + thread] = consulta;
  chaves[token * DIMENSAO + thread] = chave;
}

__global__ void contexto_incremental_kernel(
    const float* __restrict__ consultas,
    const float* __restrict__ chaves,
    const float* __restrict__ valores,
    float* __restrict__ contexto,
    int lote,
    int capacidade,
    int comprimento) {
  const int item = blockIdx.x;
  const int thread = threadIdx.x;
  if (item >= lote || thread >= DIMENSAO) {
    return;
  }
  __shared__ float pontuacoes[MAXIMO_CONTEXTO];
  __shared__ float melhores_pesos[TOP_K_ATENCAO];
  __shared__ int melhores_indices[TOP_K_ATENCAO];

  for (int posicao = thread; posicao < comprimento; posicao += DIMENSAO) {
    float produto = 0.0f;
    const int base_chave = (item * capacidade + posicao) * DIMENSAO;
    #pragma unroll 4
    for (int dimensao = 0; dimensao < DIMENSAO; ++dimensao) {
      produto +=
          consultas[item * DIMENSAO + dimensao]
          * chaves[base_chave + dimensao];
    }
    pontuacoes[posicao] = produto * 0.08838834764831843f;
  }
  __syncthreads();

  const int quantidade = comprimento < TOP_K_ATENCAO
      ? comprimento
      : TOP_K_ATENCAO;
  if (thread == 0) {
    for (int escolhido = 0; escolhido < quantidade; ++escolhido) {
      float melhor = -3.402823466e+38F;
      int melhor_indice = 0;
      for (int posicao = 0; posicao < comprimento; ++posicao) {
        if (pontuacoes[posicao] > melhor) {
          melhor = pontuacoes[posicao];
          melhor_indice = posicao;
        }
      }
      melhores_pesos[escolhido] = melhor;
      melhores_indices[escolhido] = melhor_indice;
      pontuacoes[melhor_indice] = -3.402823466e+38F;
    }
    float maximo = melhores_pesos[0];
    for (int indice = 1; indice < quantidade; ++indice) {
      maximo = fmaxf(maximo, melhores_pesos[indice]);
    }
    float soma = 0.0f;
    for (int indice = 0; indice < quantidade; ++indice) {
      const float exponencial = __expf(melhores_pesos[indice] - maximo);
      melhores_pesos[indice] = exponencial;
      soma += exponencial;
    }
    for (int indice = 0; indice < quantidade; ++indice) {
      melhores_pesos[indice] /= soma;
    }
  }
  __syncthreads();

  float acumulado = 0.0f;
  for (int indice = 0; indice < quantidade; ++indice) {
    const int base_valor =
        (item * capacidade + melhores_indices[indice]) * DIMENSAO;
    acumulado += melhores_pesos[indice] * valores[base_valor + thread];
  }
  contexto[item * DIMENSAO + thread] = acumulado;
}

__global__ void residual_ffn_roteada_kernel(
    const float* __restrict__ estados,
    const float* __restrict__ contexto,
    const float* __restrict__ gate_atencao,
    const float* __restrict__ peso_normalizacao,
    const float* __restrict__ bias_normalizacao,
    float epsilon,
    const float* __restrict__ projecao_contexto,
    const float* __restrict__ codigos_grupos,
    const float* __restrict__ bias_roteador,
    const float* __restrict__ pesos_1,
    const float* __restrict__ bias_1,
    const float* __restrict__ pesos_2,
    const float* __restrict__ bias_2,
    const float* __restrict__ gate_ffn,
    float* __restrict__ saida,
    int quantidade_tokens) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  if (token >= quantidade_tokens || thread >= THREADS) {
    return;
  }
  __shared__ float reducao[THREADS];
  __shared__ float residuais[DIMENSAO];
  __shared__ float normalizados[DIMENSAO];
  __shared__ float latentes[4];
  __shared__ float logits_roteador[16];
  __shared__ int rotas[GRUPOS_ATIVOS];
  __shared__ float gates[GRUPOS_ATIVOS];
  __shared__ float ordenacao_valores[THREADS];
  __shared__ int ordenacao_indices[THREADS];
  __shared__ float ativacoes_podadas[ATIVACOES];

  const float residual =
      estados[token * DIMENSAO + thread]
      + sigmoide(gate_atencao[0])
          * contexto[token * DIMENSAO + thread];
  residuais[thread] = residual;
  reducao[thread] = residual;
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float media = reducao[0] / static_cast<float>(DIMENSAO);
  const float diferenca = residual - media;
  reducao[thread] = diferenca * diferenca;
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float inverso_desvio = rsqrtf(
      reducao[0] / static_cast<float>(DIMENSAO) + epsilon);
  normalizados[thread] =
      diferenca * inverso_desvio * peso_normalizacao[thread]
      + bias_normalizacao[thread];
  __syncthreads();

  if (thread < 4) {
    float acumulado = 0.0f;
    for (int dimensao = 0; dimensao < DIMENSAO; ++dimensao) {
      acumulado +=
          normalizados[dimensao]
          * projecao_contexto[dimensao * 4 + thread];
    }
    latentes[thread] = tanhf(acumulado);
  }
  __syncthreads();
  if (thread < 16) {
    float acumulado = bias_roteador[thread];
    #pragma unroll
    for (int latente = 0; latente < 4; ++latente) {
      acumulado += latentes[latente] * codigos_grupos[latente * 16 + thread];
    }
    logits_roteador[thread] = acumulado;
  }
  __syncthreads();
  if (thread == 0) {
    float logits_copia[16];
    #pragma unroll
    for (int grupo = 0; grupo < 16; ++grupo) {
      logits_copia[grupo] = logits_roteador[grupo];
    }
    float maximo = -3.402823466e+38F;
    #pragma unroll
    for (int posicao = 0; posicao < GRUPOS_ATIVOS; ++posicao) {
      float melhor = -3.402823466e+38F;
      int indice_melhor = 0;
      #pragma unroll
      for (int grupo = 0; grupo < 16; ++grupo) {
        if (logits_copia[grupo] > melhor) {
          melhor = logits_copia[grupo];
          indice_melhor = grupo;
        }
      }
      rotas[posicao] = indice_melhor;
      gates[posicao] = melhor;
      maximo = fmaxf(maximo, melhor);
      logits_copia[indice_melhor] = -3.402823466e+38F;
    }
    float soma = 0.0f;
    #pragma unroll
    for (int posicao = 0; posicao < GRUPOS_ATIVOS; ++posicao) {
      gates[posicao] = __expf(gates[posicao] - maximo);
      soma += gates[posicao];
    }
    #pragma unroll
    for (int posicao = 0; posicao < GRUPOS_ATIVOS; ++posicao) {
      gates[posicao] = gates[posicao] * GRUPOS_ATIVOS / soma;
    }
  }
  __syncthreads();

  if (thread < ATIVACOES) {
    const int posicao_rota = thread / UNIDADES_GRUPO;
    const int unidade = thread % UNIDADES_GRUPO;
    const int grupo = rotas[posicao_rota];
    const int deslocamento_peso =
        (grupo * UNIDADES_GRUPO + unidade) * DIMENSAO;
    float acumulado = bias_1[grupo * UNIDADES_GRUPO + unidade];
    #pragma unroll 4
    for (int entrada_indice = 0;
         entrada_indice < DIMENSAO;
         ++entrada_indice) {
      acumulado +=
          normalizados[entrada_indice]
          * pesos_1[deslocamento_peso + entrada_indice];
    }
    const float ativacao = gelu_exata(acumulado);
    ordenacao_valores[thread] = ativacao;
    ordenacao_indices[thread] = thread;
    ativacoes_podadas[thread] = 0.0f;
  } else {
    ordenacao_valores[thread] = -3.402823466e+38F;
    ordenacao_indices[thread] = thread;
  }
  __syncthreads();
  for (int tamanho = 2; tamanho <= THREADS; tamanho <<= 1) {
    for (int distancia = tamanho >> 1; distancia > 0; distancia >>= 1) {
      const int parceiro = thread ^ distancia;
      if (parceiro > thread) {
        const bool crescente = (thread & tamanho) == 0;
        const float valor_atual = ordenacao_valores[thread];
        const float valor_parceiro = ordenacao_valores[parceiro];
        if ((valor_atual > valor_parceiro) == crescente) {
          ordenacao_valores[thread] = valor_parceiro;
          ordenacao_valores[parceiro] = valor_atual;
          const int indice_atual = ordenacao_indices[thread];
          ordenacao_indices[thread] = ordenacao_indices[parceiro];
          ordenacao_indices[parceiro] = indice_atual;
        }
      }
      __syncthreads();
    }
  }
  if (thread >= THREADS - TOP_K) {
    const int indice_original = ordenacao_indices[thread];
    if (indice_original < ATIVACOES) {
      ativacoes_podadas[indice_original] = ordenacao_valores[thread];
    }
  }
  __syncthreads();

  float acumulado_saida = bias_2[thread];
  #pragma unroll
  for (int posicao_rota = 0;
       posicao_rota < GRUPOS_ATIVOS;
       ++posicao_rota) {
    const int grupo = rotas[posicao_rota];
    const int deslocamento_peso =
        (grupo * DIMENSAO + thread) * UNIDADES_GRUPO;
    float soma_grupo = 0.0f;
    #pragma unroll
    for (int unidade = 0; unidade < UNIDADES_GRUPO; ++unidade) {
      soma_grupo +=
          ativacoes_podadas[
              posicao_rota * UNIDADES_GRUPO + unidade]
          * pesos_2[deslocamento_peso + unidade];
    }
    acumulado_saida += gates[posicao_rota] * soma_grupo;
  }
  saida[token * DIMENSAO + thread] =
      residuais[thread] + sigmoide(gate_ffn[0]) * acumulado_saida;
}

__global__ void embedding_posicao_kernel(
    const int64_t* __restrict__ tokens,
    const float* __restrict__ embedding,
    const float* __restrict__ posicao,
    const float* __restrict__ gate_posicao,
    float* __restrict__ saida,
    int quantidade_tokens) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  if (token >= quantidade_tokens || thread >= DIMENSAO) {
    return;
  }
  const int64_t token_id = tokens[token];
  saida[token * DIMENSAO + thread] =
      embedding[token_id * DIMENSAO + thread] * 11.313708498984761f
      + sigmoide(gate_posicao[0]) * posicao[thread];
}

__global__ void saida_vocabulario_kernel(
    const float* __restrict__ estados,
    const float* __restrict__ peso_normalizacao,
    const float* __restrict__ bias_normalizacao,
    float epsilon,
    const float* __restrict__ embedding,
    const float* __restrict__ bias_saida,
    float* __restrict__ logits,
    int quantidade_tokens,
    int vocabulario) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  if (token >= quantidade_tokens || thread >= MAXIMO_VOCABULARIO) {
    return;
  }
  __shared__ float reducao[DIMENSAO];
  __shared__ float normalizados[DIMENSAO];
  if (thread < DIMENSAO) {
    reducao[thread] = estados[token * DIMENSAO + thread];
  }
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float media = reducao[0] / static_cast<float>(DIMENSAO);
  if (thread < DIMENSAO) {
    const float diferenca =
        estados[token * DIMENSAO + thread] - media;
    reducao[thread] = diferenca * diferenca;
  }
  __syncthreads();
  for (int passo = DIMENSAO / 2; passo > 0; passo >>= 1) {
    if (thread < passo) {
      reducao[thread] += reducao[thread + passo];
    }
    __syncthreads();
  }
  const float inverso_desvio = rsqrtf(
      reducao[0] / static_cast<float>(DIMENSAO) + epsilon);
  if (thread < DIMENSAO) {
    normalizados[thread] =
        (estados[token * DIMENSAO + thread] - media)
        * inverso_desvio
        * peso_normalizacao[thread]
        + bias_normalizacao[thread];
  }
  __syncthreads();
  if (thread < vocabulario) {
    float acumulado = bias_saida[thread];
    for (int dimensao = 0; dimensao < DIMENSAO; ++dimensao) {
      acumulado +=
          normalizados[dimensao]
          * embedding[thread * DIMENSAO + dimensao];
    }
    logits[token * vocabulario + thread] = acumulado;
  }
}

void validar_tensor(
    const at::Tensor& tensor,
    const char* nome,
    at::ScalarType tipo) {
  TORCH_CHECK(tensor.is_cuda(), nome, " deve estar em CUDA");
  TORCH_CHECK(tensor.is_contiguous(), nome, " deve ser contiguo");
  TORCH_CHECK(tensor.scalar_type() == tipo, nome, " possui dtype invalido");
}

}  // namespace

at::Tensor ffn_roteada_cuda(
    at::Tensor entrada,
    at::Tensor rotas,
    at::Tensor gates,
    at::Tensor pesos_1,
    at::Tensor bias_1,
    at::Tensor pesos_2,
    at::Tensor bias_2) {
  validar_tensor(entrada, "entrada", at::kFloat);
  validar_tensor(rotas, "rotas", at::kLong);
  validar_tensor(gates, "gates", at::kFloat);
  validar_tensor(pesos_1, "pesos_1", at::kFloat);
  validar_tensor(bias_1, "bias_1", at::kFloat);
  validar_tensor(pesos_2, "pesos_2", at::kFloat);
  validar_tensor(bias_2, "bias_2", at::kFloat);
  TORCH_CHECK(entrada.dim() == 2 && entrada.size(1) == DIMENSAO);
  TORCH_CHECK(rotas.sizes() == at::IntArrayRef({entrada.size(0), 4}));
  TORCH_CHECK(gates.sizes() == at::IntArrayRef({entrada.size(0), 4}));
  TORCH_CHECK(
      pesos_1.sizes() == at::IntArrayRef({16, 24, DIMENSAO}));
  TORCH_CHECK(bias_1.sizes() == at::IntArrayRef({16, 24}));
  TORCH_CHECK(
      pesos_2.sizes() == at::IntArrayRef({16, DIMENSAO, 24}));
  TORCH_CHECK(bias_2.numel() == DIMENSAO);

  c10::cuda::CUDAGuard guarda(entrada.device());
  auto saida = at::empty_like(entrada);
  const int quantidade_tokens = static_cast<int>(entrada.size(0));
  ffn_roteada_kernel<<<
      quantidade_tokens,
      THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          entrada.data_ptr<float>(),
          rotas.data_ptr<int64_t>(),
          gates.data_ptr<float>(),
          pesos_1.data_ptr<float>(),
          bias_1.data_ptr<float>(),
          pesos_2.data_ptr<float>(),
          bias_2.data_ptr<float>(),
          saida.data_ptr<float>(),
          quantidade_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return saida;
}

std::vector<at::Tensor> normalizar_qk_cuda(
    at::Tensor entrada,
    at::Tensor peso_normalizacao,
    at::Tensor bias_normalizacao,
    double epsilon,
    at::Tensor indices_q,
    at::Tensor valores_q,
    at::Tensor bias_q,
    at::Tensor indices_k,
    at::Tensor valores_k,
    at::Tensor bias_k) {
  validar_tensor(entrada, "entrada", at::kFloat);
  validar_tensor(
      peso_normalizacao,
      "peso_normalizacao",
      at::kFloat);
  validar_tensor(
      bias_normalizacao,
      "bias_normalizacao",
      at::kFloat);
  validar_tensor(indices_q, "indices_q", at::kLong);
  validar_tensor(valores_q, "valores_q", at::kFloat);
  validar_tensor(bias_q, "bias_q", at::kFloat);
  validar_tensor(indices_k, "indices_k", at::kLong);
  validar_tensor(valores_k, "valores_k", at::kFloat);
  validar_tensor(bias_k, "bias_k", at::kFloat);
  TORCH_CHECK(entrada.dim() == 2 && entrada.size(1) == DIMENSAO);
  TORCH_CHECK(indices_q.sizes() == at::IntArrayRef({128, 32}));
  TORCH_CHECK(valores_q.sizes() == at::IntArrayRef({128, 32}));
  TORCH_CHECK(indices_k.sizes() == at::IntArrayRef({128, 32}));
  TORCH_CHECK(valores_k.sizes() == at::IntArrayRef({128, 32}));

  c10::cuda::CUDAGuard guarda(entrada.device());
  auto normalizados = at::empty_like(entrada);
  auto consultas = at::empty_like(entrada);
  auto chaves = at::empty_like(entrada);
  const int quantidade_tokens = static_cast<int>(entrada.size(0));
  normalizar_qk_kernel<<<
      quantidade_tokens,
      THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          entrada.data_ptr<float>(),
          peso_normalizacao.data_ptr<float>(),
          bias_normalizacao.data_ptr<float>(),
          static_cast<float>(epsilon),
          indices_q.data_ptr<int64_t>(),
          valores_q.data_ptr<float>(),
          bias_q.data_ptr<float>(),
          indices_k.data_ptr<int64_t>(),
          valores_k.data_ptr<float>(),
          bias_k.data_ptr<float>(),
          normalizados.data_ptr<float>(),
          consultas.data_ptr<float>(),
          chaves.data_ptr<float>(),
          quantidade_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {normalizados, consultas, chaves};
}

at::Tensor contexto_incremental_cuda(
    at::Tensor consulta,
    at::Tensor chaves,
    at::Tensor valores,
    int64_t comprimento) {
  validar_tensor(consulta, "consulta", at::kFloat);
  validar_tensor(chaves, "chaves", at::kFloat);
  validar_tensor(valores, "valores", at::kFloat);
  TORCH_CHECK(consulta.dim() == 2 && consulta.size(1) == DIMENSAO);
  TORCH_CHECK(chaves.dim() == 3 && chaves.size(2) == DIMENSAO);
  TORCH_CHECK(valores.sizes() == chaves.sizes());
  TORCH_CHECK(chaves.size(0) == consulta.size(0));
  TORCH_CHECK(
      comprimento > 0
      && comprimento <= chaves.size(1)
      && comprimento <= MAXIMO_CONTEXTO);

  c10::cuda::CUDAGuard guarda(consulta.device());
  auto contexto = at::empty_like(consulta);
  contexto_incremental_kernel<<<
      consulta.size(0),
      THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          consulta.data_ptr<float>(),
          chaves.data_ptr<float>(),
          valores.data_ptr<float>(),
          contexto.data_ptr<float>(),
          static_cast<int>(consulta.size(0)),
          static_cast<int>(chaves.size(1)),
          static_cast<int>(comprimento));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return contexto;
}

at::Tensor residual_ffn_roteada_cuda(
    at::Tensor estados,
    at::Tensor contexto,
    at::Tensor gate_atencao,
    at::Tensor peso_normalizacao,
    at::Tensor bias_normalizacao,
    double epsilon,
    at::Tensor projecao_contexto,
    at::Tensor codigos_grupos,
    at::Tensor bias_roteador,
    at::Tensor pesos_1,
    at::Tensor bias_1,
    at::Tensor pesos_2,
    at::Tensor bias_2,
    at::Tensor gate_ffn) {
  validar_tensor(estados, "estados", at::kFloat);
  validar_tensor(contexto, "contexto", at::kFloat);
  validar_tensor(gate_atencao, "gate_atencao", at::kFloat);
  validar_tensor(
      peso_normalizacao,
      "peso_normalizacao",
      at::kFloat);
  validar_tensor(
      bias_normalizacao,
      "bias_normalizacao",
      at::kFloat);
  validar_tensor(
      projecao_contexto,
      "projecao_contexto",
      at::kFloat);
  validar_tensor(codigos_grupos, "codigos_grupos", at::kFloat);
  validar_tensor(bias_roteador, "bias_roteador", at::kFloat);
  validar_tensor(pesos_1, "pesos_1", at::kFloat);
  validar_tensor(bias_1, "bias_1", at::kFloat);
  validar_tensor(pesos_2, "pesos_2", at::kFloat);
  validar_tensor(bias_2, "bias_2", at::kFloat);
  validar_tensor(gate_ffn, "gate_ffn", at::kFloat);
  TORCH_CHECK(estados.dim() == 2 && estados.size(1) == DIMENSAO);
  TORCH_CHECK(contexto.sizes() == estados.sizes());
  TORCH_CHECK(projecao_contexto.sizes() == at::IntArrayRef({128, 4}));
  TORCH_CHECK(codigos_grupos.sizes() == at::IntArrayRef({4, 16}));
  TORCH_CHECK(pesos_1.sizes() == at::IntArrayRef({16, 24, 128}));
  TORCH_CHECK(pesos_2.sizes() == at::IntArrayRef({16, 128, 24}));

  c10::cuda::CUDAGuard guarda(estados.device());
  auto saida = at::empty_like(estados);
  residual_ffn_roteada_kernel<<<
      estados.size(0),
      THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          estados.data_ptr<float>(),
          contexto.data_ptr<float>(),
          gate_atencao.data_ptr<float>(),
          peso_normalizacao.data_ptr<float>(),
          bias_normalizacao.data_ptr<float>(),
          static_cast<float>(epsilon),
          projecao_contexto.data_ptr<float>(),
          codigos_grupos.data_ptr<float>(),
          bias_roteador.data_ptr<float>(),
          pesos_1.data_ptr<float>(),
          bias_1.data_ptr<float>(),
          pesos_2.data_ptr<float>(),
          bias_2.data_ptr<float>(),
          gate_ffn.data_ptr<float>(),
          saida.data_ptr<float>(),
          static_cast<int>(estados.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return saida;
}

at::Tensor embedding_posicao_cuda(
    at::Tensor tokens,
    at::Tensor embedding,
    at::Tensor posicao,
    at::Tensor gate_posicao) {
  validar_tensor(tokens, "tokens", at::kLong);
  validar_tensor(embedding, "embedding", at::kFloat);
  validar_tensor(posicao, "posicao", at::kFloat);
  validar_tensor(gate_posicao, "gate_posicao", at::kFloat);
  TORCH_CHECK(tokens.dim() == 1);
  TORCH_CHECK(embedding.dim() == 2 && embedding.size(1) == DIMENSAO);
  TORCH_CHECK(posicao.numel() == DIMENSAO);

  c10::cuda::CUDAGuard guarda(tokens.device());
  auto saida = at::empty(
      {tokens.size(0), DIMENSAO},
      embedding.options());
  embedding_posicao_kernel<<<
      tokens.size(0),
      THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          tokens.data_ptr<int64_t>(),
          embedding.data_ptr<float>(),
          posicao.data_ptr<float>(),
          gate_posicao.data_ptr<float>(),
          saida.data_ptr<float>(),
          static_cast<int>(tokens.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return saida;
}

at::Tensor saida_vocabulario_cuda(
    at::Tensor estados,
    at::Tensor peso_normalizacao,
    at::Tensor bias_normalizacao,
    double epsilon,
    at::Tensor embedding,
    at::Tensor bias_saida) {
  validar_tensor(estados, "estados", at::kFloat);
  validar_tensor(
      peso_normalizacao,
      "peso_normalizacao",
      at::kFloat);
  validar_tensor(
      bias_normalizacao,
      "bias_normalizacao",
      at::kFloat);
  validar_tensor(embedding, "embedding", at::kFloat);
  validar_tensor(bias_saida, "bias_saida", at::kFloat);
  TORCH_CHECK(estados.dim() == 2 && estados.size(1) == DIMENSAO);
  TORCH_CHECK(embedding.dim() == 2 && embedding.size(1) == DIMENSAO);
  TORCH_CHECK(embedding.size(0) <= MAXIMO_VOCABULARIO);

  c10::cuda::CUDAGuard guarda(estados.device());
  auto logits = at::empty(
      {estados.size(0), embedding.size(0)},
      estados.options());
  saida_vocabulario_kernel<<<
      estados.size(0),
      MAXIMO_VOCABULARIO,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          estados.data_ptr<float>(),
          peso_normalizacao.data_ptr<float>(),
          bias_normalizacao.data_ptr<float>(),
          static_cast<float>(epsilon),
          embedding.data_ptr<float>(),
          bias_saida.data_ptr<float>(),
          logits.data_ptr<float>(),
          static_cast<int>(estados.size(0)),
          static_cast<int>(embedding.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return logits;
}
