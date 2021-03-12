#include <iostream>
#include <random>

#include "global_typedef.h"
#include "environment_util.h"
#include "ANN/ANN.h"
#include "cppyplot.hpp"

using namespace ANN;
#define WORLD_FILE ("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/world_barriers.csv")

// template<typename EigenDerived1, typename EigenDerived2>
eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>
loss(const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& pred_out, 
     const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& ref_out)
{
  eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor> 
  retval (pred_out.rows(), 1);
  retval(all, 0)  = ((pred_out(all, 0) - ref_out(all, 0))*(pred_out(all, 0) - ref_out(all, 0)));
  retval(all, 0) += ((pred_out(all, 1) - ref_out(all, 1))*(pred_out(all, 1) - ref_out(all, 1)));

  retval *= 0.5F;
  return retval;
}

// template<typename EigenDerived1, typename EigenDerived2>
eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>
loss_grad(const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& pred_out, 
          const eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor>& ref_out)
{
  eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor> 
  retval (pred_out.rows(), pred_out.cols());
  retval(all, 0) = (pred_out(all, 0) - ref_out(all, 0));
  retval(all, 1) = (pred_out(all, 1) - ref_out(all, 1));
  return retval;
}

template<typename EigenDerived1, typename EigenDerived2>
auto test_gradient(ArtificialNeuralNetwork<2, 3, 3, 2>& ann, 
                   const eig::ArrayBase<EigenDerived1>& X,
                   const eig::ArrayBase<EigenDerived2>& Y)
{
  decltype(ann.weight) w = ann.weight;
  decltype(ann.bias)   b = ann.bias;

  decltype(w) weight_grad;
  decltype(b) bias_grad;

  auto pred_out = forward_batch<25>(ann, X);
  float epsilon = 1e-8F;

  // calculate weight gradient
  auto loss_noperturbed = loss(pred_out, Y);
  float loss_noperturbed_normalized = loss_noperturbed.mean();
  for (int i = 0; i < w.rows(); i++)
  {
    ann.weight(i, 0) -= epsilon;
    auto pred_out_perturbed = forward_batch<25>(ann, X);
    ann.weight(i, 0) += epsilon;

    auto loss_perturbed = loss(pred_out_perturbed, Y);
    float loss_perturbed_normalizedLeft = loss_perturbed.mean();

    ann.weight(i, 0) += epsilon;
    pred_out_perturbed = forward_batch<25>(ann, X);
    ann.weight(i, 0) -= epsilon;

    loss_perturbed = loss(pred_out_perturbed, Y);
    float loss_perturbed_normalizedRight = loss_perturbed.mean();

    weight_grad(i, 0) = (loss_perturbed_normalizedLeft - loss_perturbed_normalizedRight)/(2.0F*epsilon);
  }

  // calculate bias gradient
  for (int i = 0; i < b.rows(); i++)
  {
    ann.bias(i, 0) -= epsilon;
    auto pred_out_perturbed = forward_batch<25>(ann, X);
    ann.bias(i, 0) += epsilon;

    auto loss_perturbed = loss(pred_out_perturbed, Y);
    float loss_perturbed_normalizedLeft = loss_perturbed.mean();

    ann.bias(i, 0) += epsilon;
    pred_out_perturbed = forward_batch<25>(ann, X);
    ann.bias(i, 0) -= epsilon;

    loss_perturbed = loss(pred_out_perturbed, Y);
    float loss_perturbed_normalizedRight = loss_perturbed.mean();

    bias_grad(i, 0) = (loss_perturbed_normalizedLeft - loss_perturbed_normalizedRight)/(2.0F*epsilon);
  }

  return std::make_tuple(weight_grad, bias_grad);
}

int main()
{
  // auto world_barriers = ENV::read_world(WORLD_FILE);
  
  // TODO: add robot parameter file for experimentation 
  // auto robot_params   = ENV::read_robot_config("X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/robot_params.yaml");

  ArtificialNeuralNetwork<2, 3, 3, 2> sampling_policy;
  sampling_policy.dense(ReLU(Initializer(HE)), 
                        ReLU(Initializer(HE)), 
                        ReLU(Initializer(HE)) );
  
  eig::Array<float, 25, 2, eig::RowMajor> X;
  X.setRandom();
  
  eig::Array<float, 25, 2, eig::RowMajor> Y;
  Y.setRandom();

  auto [weight_grad, bias_grad] = gradient_batch<25>(sampling_policy, X, Y, loss, loss_grad);
  auto [weight_grad_ref, bias_grad_ref] = test_gradient(sampling_policy, X, Y);

  std::cout << "weight_error: \n" << (weight_grad - weight_grad_ref) << "\n";
  std::cout << "bias_error: \n" << (bias_grad - bias_grad_ref) << "\n";

  // TODO: Make it compile
  // TODO: Test ANN with a sample test
  // ENV::realtime_visualizer_init(WORLD_FILE);
  // float x = 0.0F, y = 0.0F, Vx = 4.5F, Vy = 2.5F;
  // std::random_device seed;
  // std::mt19937 rand_gen(seed());
  // std::uniform_real_distribution<float> uniform_dist(0.0F, 4.5F);

  // for (float t = 0.04F; t < 10.0F; t+=0.04F)
  // {
  //   Vx = uniform_dist(rand_gen);
  //   Vy = uniform_dist(rand_gen);
  //   x += Vx*t;
  //   y += Vy*t;
  //   ENV::update_visualizer({x,y}, {Vx, Vy}, uniform_dist(rand_gen));
  //   std::this_thread::sleep_for(10ms);
  // }

  std::cout << "Compiled\n";

  return EXIT_SUCCESS;
}