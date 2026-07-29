#include <torch/extension.h>

torch::Tensor ffn_roteada_cuda(
    torch::Tensor entrada,
    torch::Tensor rotas,
    torch::Tensor gates,
    torch::Tensor pesos_1,
    torch::Tensor bias_1,
    torch::Tensor pesos_2,
    torch::Tensor bias_2);

std::vector<torch::Tensor> normalizar_qk_cuda(
    torch::Tensor entrada,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor indices_q,
    torch::Tensor valores_q,
    torch::Tensor bias_q,
    torch::Tensor indices_k,
    torch::Tensor valores_k,
    torch::Tensor bias_k);

torch::Tensor contexto_incremental_cuda(
    torch::Tensor consulta,
    torch::Tensor chaves,
    torch::Tensor valores,
    int64_t comprimento);

torch::Tensor residual_ffn_roteada_cuda(
    torch::Tensor estados,
    torch::Tensor contexto,
    torch::Tensor gate_atencao,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor projecao_contexto,
    torch::Tensor codigos_grupos,
    torch::Tensor bias_roteador,
    torch::Tensor pesos_1,
    torch::Tensor bias_1,
    torch::Tensor pesos_2,
    torch::Tensor bias_2,
    torch::Tensor gate_ffn);

torch::Tensor embedding_posicao_cuda(
    torch::Tensor tokens,
    torch::Tensor embedding,
    torch::Tensor posicao,
    torch::Tensor gate_posicao);

torch::Tensor saida_vocabulario_cuda(
    torch::Tensor estados,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor embedding,
    torch::Tensor bias_saida);

torch::Tensor ffn_roteada(
    torch::Tensor entrada,
    torch::Tensor rotas,
    torch::Tensor gates,
    torch::Tensor pesos_1,
    torch::Tensor bias_1,
    torch::Tensor pesos_2,
    torch::Tensor bias_2) {
  TORCH_CHECK(entrada.is_cuda(), "entrada deve estar em CUDA");
  TORCH_CHECK(rotas.is_cuda(), "rotas devem estar em CUDA");
  TORCH_CHECK(gates.is_cuda(), "gates devem estar em CUDA");
  return ffn_roteada_cuda(
      entrada,
      rotas,
      gates,
      pesos_1,
      bias_1,
      pesos_2,
      bias_2);
}

std::vector<torch::Tensor> normalizar_qk(
    torch::Tensor entrada,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor indices_q,
    torch::Tensor valores_q,
    torch::Tensor bias_q,
    torch::Tensor indices_k,
    torch::Tensor valores_k,
    torch::Tensor bias_k) {
  TORCH_CHECK(entrada.is_cuda(), "entrada deve estar em CUDA");
  return normalizar_qk_cuda(
      entrada,
      peso_normalizacao,
      bias_normalizacao,
      epsilon,
      indices_q,
      valores_q,
      bias_q,
      indices_k,
      valores_k,
      bias_k);
}

torch::Tensor contexto_incremental(
    torch::Tensor consulta,
    torch::Tensor chaves,
    torch::Tensor valores,
    int64_t comprimento) {
  TORCH_CHECK(consulta.is_cuda(), "consulta deve estar em CUDA");
  return contexto_incremental_cuda(
      consulta,
      chaves,
      valores,
      comprimento);
}

torch::Tensor residual_ffn_roteada(
    torch::Tensor estados,
    torch::Tensor contexto,
    torch::Tensor gate_atencao,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor projecao_contexto,
    torch::Tensor codigos_grupos,
    torch::Tensor bias_roteador,
    torch::Tensor pesos_1,
    torch::Tensor bias_1,
    torch::Tensor pesos_2,
    torch::Tensor bias_2,
    torch::Tensor gate_ffn) {
  TORCH_CHECK(estados.is_cuda(), "estados devem estar em CUDA");
  return residual_ffn_roteada_cuda(
      estados,
      contexto,
      gate_atencao,
      peso_normalizacao,
      bias_normalizacao,
      epsilon,
      projecao_contexto,
      codigos_grupos,
      bias_roteador,
      pesos_1,
      bias_1,
      pesos_2,
      bias_2,
      gate_ffn);
}

torch::Tensor embedding_posicao(
    torch::Tensor tokens,
    torch::Tensor embedding,
    torch::Tensor posicao,
    torch::Tensor gate_posicao) {
  TORCH_CHECK(tokens.is_cuda(), "tokens devem estar em CUDA");
  return embedding_posicao_cuda(
      tokens,
      embedding,
      posicao,
      gate_posicao);
}

torch::Tensor saida_vocabulario(
    torch::Tensor estados,
    torch::Tensor peso_normalizacao,
    torch::Tensor bias_normalizacao,
    double epsilon,
    torch::Tensor embedding,
    torch::Tensor bias_saida) {
  TORCH_CHECK(estados.is_cuda(), "estados devem estar em CUDA");
  return saida_vocabulario_cuda(
      estados,
      peso_normalizacao,
      bias_normalizacao,
      epsilon,
      embedding,
      bias_saida);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, modulo) {
  modulo.def(
      "ffn_roteada",
      &ffn_roteada,
      "FFN combinatoria V7.3 fundida (CUDA)");
  modulo.def(
      "normalizar_qk",
      &normalizar_qk,
      "LayerNorm e Q/K esparsas fundidas (CUDA)");
  modulo.def(
      "contexto_incremental",
      &contexto_incremental,
      "Atencao Top-K incremental fundida (CUDA)");
  modulo.def(
      "residual_ffn_roteada",
      &residual_ffn_roteada,
      "Residual, roteador e FFN fundidos (CUDA)");
  modulo.def(
      "embedding_posicao",
      &embedding_posicao,
      "Embedding e posicao fundidos (CUDA)");
  modulo.def(
      "saida_vocabulario",
      &saida_vocabulario,
      "Normalizacao final e vocabulario fundidos (CUDA)");
}
