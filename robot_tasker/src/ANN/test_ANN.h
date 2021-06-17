#ifndef _TEST_ANN_H_
#define _TEST_ANN_H_

#include "../global_typedef.h"
#include "ANN.h"
#include "ANN_optimizers.h"
#include "cppyplot.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

using namespace ANN;

auto load_moon_data(const std::string& data_file)
{
  eig::Array<float, 5000, 2, eig::RowMajor> X;
  eig::Array<float, 5000, 1>                Y;

  std::ifstream in_file(data_file);
  if (in_file.is_open())
  {
    std::stringstream file_buf;
    file_buf << in_file.rdbuf();
    string line;
    
    // read and skip first line (header)
    std::getline(file_buf, line);
    size_t row = 0u;
    while(std::getline(file_buf, line))
    {
      if (!line.empty())
      {
        string data;
        float x1, x2, y;
        std::stringstream line_buf (line);
        
        std::getline(line_buf, data, ',');
        str2float(data, x1);

        std::getline(line_buf, data, ',');
        str2float(data, x2);

        std::getline(line_buf, data, ',');
        str2float(data, y);

        X(row, 0) = x1;
        X(row, 1) = x2;
        Y(row, 0) = y;
      }
      row++;
    }
  }
  in_file.close();
  
  return std::make_tuple(X, Y);
}

template<int BatchSize>
float loss_fcn(const eig::Array<float, BatchSize, 1>& pred,  
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
eig::Array<float, BatchSize, 1>
loss_grad_fcn(const eig::Array<float, BatchSize, 1>& pred, 
              const eig::Array<float, BatchSize, 1>& ref_out)
{
  eig::Array<float, BatchSize, 1> grad = (pred - ref_out);
  return grad;
}

template<int N>
std::array<int, N> get_n_shuffled_idx(const int container_size)
{
  std::vector<int> indices(container_size);
  std::array<int, N> shuffled_n_idx;
  std::random_device rd;
  std::mt19937 g(rd());

  std::generate(indices.begin(), indices.end(), [n = 0] () mutable { return n++; });
  std::shuffle(indices.begin(), indices.end(), g);

  for (int i = 0; i < N; i++)
  { shuffled_n_idx[i] = indices[i]; }

  return shuffled_n_idx;
}

template<int BatchSize, typename Network, typename InputDerived, typename OutputDerived>
void calc_accuracy(const Network& network, 
                   const eig::ArrayBase<InputDerived>& input, 
                   const eig::ArrayBase<OutputDerived>& output)
{
  auto pred_out = forward_batch<BatchSize>(network, input);
  auto pred_out_filtered = pred_out.unaryExpr([](float v){return v < 0.5F?0.0F:1.0F;});
  int n_correct_predictions = 0;
  for (int i = 0; i < BatchSize; i++)
  {
    if ((int)pred_out_filtered(i, 0) == (int)output(i, 0))
    {
      n_correct_predictions++;
    }
  }
  std::cout << "Accuracy: " << (float)(n_correct_predictions)/(float)BatchSize;
}


template<int BatchSize>
void test_ann()
{
  std::string data_file = "X:/Video_Lectures/ReinforcementLearning/scripts/robot_tasker/ann_test_moon_data.csv";
  Cppyplot::cppyplot pyp;
  auto [input, ref_out] = load_moon_data(data_file);
  ArtificialNeuralNetwork<2, 4, 1> network;
  network.dense(Activation(RELU, HE), 
                Activation(SIGMOID, XAVIER));

  AdamOptimizer network_opt((int)network.weight.rows(), (int)network.bias.rows(), 0.001F);
  constexpr size_t max_epoch = 500u;
  eig::Array<float, max_epoch, 1> loss_hist;
  size_t current_epoch       = 0u;
  while(current_epoch < max_epoch)
  {
    auto n_shuffled_idx = get_n_shuffled_idx<BatchSize>(3500u);
    auto [loss, weight_grad, bias_grad] = gradient_batch<BatchSize>(network, 
                                                                    input(n_shuffled_idx, all), 
                                                                    ref_out(n_shuffled_idx, all), 
                                                                    loss_fcn<BatchSize>, loss_grad_fcn<BatchSize>);

    network_opt.step(weight_grad, bias_grad, network.weight, network.bias);
    std::cout << "Epoch: " << current_epoch << ", Loss: " << loss << '\n';
    loss_hist(current_epoch, 0) = loss;
    current_epoch++;
  }
  // debug code --start
  
  pyp.raw(R"pyp(
  plt.figure(figsize=(12,4))
  plt.plot(loss_hist, 'r-o', markersize=1, linewidth=0.8)
  plt.grid(True)
  plt.xlabel("Epoch", fontsize=12)
  plt.ylabel("Loss", fontsize=12)
  plt.show()
  )pyp", _p(loss_hist));
  // debug code --end
  
  // calculate accuracy
  calc_accuracy<1500>(network, input(seq(3500, last), all), ref_out(seq(3500, last), all));

}

#endif
