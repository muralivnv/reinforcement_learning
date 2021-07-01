#include <benchmark/benchmark.h>

#include "../global_typedef.h"

#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

#include "../ANN_v2/ANN_core.h"

#pragma comment(lib, "Shlwapi.lib")

template<int BatchSize>
static float loss_fcn(const eig::Array<float, BatchSize, 1>& pred,  
                      const eig::Array<float, BatchSize, 1>& ref_out)
{
  auto pos_class = pred.unaryExpr([](float v){return v < 0.5F?1e-8F:1.0F; });
  auto neg_class = pos_class.unaryExpr([](float v){return (1.0F - v)<1e-5F?1e-8F:(1.0F-v);});

  auto log_pos_pred = eig::log(pos_class);
  auto log_neg_pred = eig::log(neg_class);

  auto neg_ref = (1.0F - ref_out);

  float loss = (log_pos_pred*ref_out).mean() + (log_neg_pred*neg_ref).mean();
  
  return -loss;
}


template<int BatchSize>
static 
eig::Array<float, BatchSize, 1>
loss_grad_fcn(const eig::Array<float, BatchSize, 1>& pred, 
              const eig::Array<float, BatchSize, 1>& ref_out)
{
  eig::Array<float, BatchSize, 1> grad = (pred - ref_out);
  return grad;
}



static void BM_forward_batch(benchmark::State& state) 
{
  ann::ArtificialNeuralNetwork<3> network;
  ann::set_layer_config(network, 2, 4, 1);
  ann::set_activations(network,  ann::RELU(),     ann::SIGMOID());
  ann::set_initializers(network, ann::HeNormal(), ann::XavierNormal());

  ann::ArrayXf_t input(512, 4);
  input.fill(0.0F);
  for (auto _ : state) 
  {
    auto out = ann::forward_batch<512>(network, input);
  }
}

static void BM_gradient_batch(benchmark::State& state)
{
  ann::ArtificialNeuralNetwork<3> network;
  ann::set_layer_config(network, 2, 4, 1);
  ann::set_activations(network,  ann::RELU(),     ann::SIGMOID());
  ann::set_initializers(network, ann::HeNormal(), ann::XavierNormal());
  ann::ArrayXf_t input(512, 4);
  ann::ArrayXf_t output(512, 1);
  input.fill(1.0F);
  output.fill(2.5F);

  for(auto _ : state)
  {
    auto [loss, w_grad, b_grad] = ann::gradient_batch<512>(network, 
                                                           input, 
                                                           output, 
                                                           loss_fcn<512>, 
                                                           loss_grad_fcn<512>);
  }
}

// Register the function as a benchmark
BENCHMARK(BM_forward_batch);
BENCHMARK(BM_gradient_batch);

// Run the benchmark
BENCHMARK_MAIN();