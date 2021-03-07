#ifndef _ANN_ACTIVATION_H_
#define _ANN_ACTIVATION_H_

#include <random>
#include "../global_typedef.h"
namespace ANN
{

enum InitializerIdentifier{
  RANDOM,
  XAVIER,
  HE
};

class Initializer{
  private:
    std::random_device rand_seed_;
    std::mt19937       rand_gen_;
  public:
    InitializerIdentifier init_type_;
    
    Initializer() = default;
    Initializer(InitializerIdentifier initialization_type)
    {
      init_type_ = initialization_type;
      rand_gen_  = std::mt19937(rand_seed_());
    }
    
    ~Initializer(){}

    Initializer(const Initializer& right)
    { init_type_ = right.init_type_; }

    Initializer(Initializer& right)
    { init_type_ = right.init_type_; }

    Initializer& operator=(Initializer& right)
    { 
      init_type_= right.init_type_; 
      return *this;
    }

    Initializer& operator=(const Initializer& right)
    { 
      init_type_= right.init_type_; 
      return *this;
    }

    template<typename EigenDerived1, typename EigenDerived2>
    auto init(eig::DenseBase<EigenDerived1>& weights,
              eig::DenseBase<EigenDerived2>& bias,
              int                            n_nodes_last_layer)
    {
      float std_dev = 1.0F;
      switch(init_type_)
      {
        case XAVIER:
          std_dev = 1.0F/std::sqrtf((float)n_nodes_last_layer);
          break;
        case HE:
          std_dev = 2.0F/std::sqrtf((float)n_nodes_last_layer);
          break;
        case RANDOM:
        default:
          std_dev = 1.0F;
          break;
      }
      std::normal_distribution<float> norm(0.0F, std_dev);
      
      // fill weights
      for (size_t i = 0u; i < (size_t)weights.size(); i++)
      { weights[i] = norm(rand_gen_); }

      // fill bias
      for (size_t i = 0u; i < (size_t)bias.size(); i++)
      { bias[i] = norm(rand_gen_); }
    }
};

class ActivationBase{
  public:
    Initializer initializer;
    ActivationBase(){}
    ActivationBase(Initializer& param_initializer) : initializer(param_initializer){}
    virtual ~ActivationBase(){}

    template<typename EigenDerived>
    auto activation_batch(const eig::DenseBase<EigenDerived>& wx_b) const
    { return RL::MatrixX<float>{}; }

    float activation(const float wx_b) const
    { return wx_b; }
    
    template<typename EigenDerived>
    auto grad(const eig::DenseBase<EigenDerived>& activation) const
    {  return activation;  }

    template<typename EigenDerived1, typename EigenDerived2>
    auto grad_batch(const eig::DenseBase<EigenDerived2>& activation) const
    {  return std::make_tuple(eig::DenseBase<EigenDerived1>(), eig::DenseBase<EigenDerived2>());  }
};

class Sigmoid: public ActivationBase{
  public:
    Initializer initializer;
    Sigmoid(){}
    Sigmoid(Initializer& param_initializer): initializer(param_initializer){}

    ~Sigmoid(){}

    template<typename EigenDerived>
    auto activation_batch(const eig::DenseBase<EigenDerived>& wx_b) const
    {
      auto retval = 1.0F + eig::exp(wx_b);
      return 1.0F/retval;
    }

    float activation(const float wx_b) const
    { return 1.0F/(1.0F + std::expf(wx_b)); }
    
    template<typename EigenDerived1, typename EigenDerived2>
    auto grad(const eig::DenseBase<EigenDerived1>& before_layer_output,
              const eig::DenseBase<EigenDerived2>& activation) const
    {
      auto gradient = activation.array()*(1.0F - activation.array());
      size_t before_layer_n_nodes = before_layer_output.rows();
      size_t cur_layer_n_nodes    = activation.rows();

      RL::VectorX<float> weight_grad (before_layer_n_nodes*cur_layer_n_nodes);

      for (int i = 0u; i < cur_layer_n_nodes; i++)
      { weight_grad(seq(i*before_layer_n_nodes, (i+1)*before_layer_n_nodes-1)) = gradient.array() * before_layer_output.array();  }

      return std::make_tuple(weight_grad, gradient);
    }

    template<int BatchSize, typename EigenDerived1, typename EigenDerived2>
    auto grad_batch(const eig::DenseBase<EigenDerived1>& before_layer_output,
                    const eig::DenseBase<EigenDerived2>& activation) const
    {
      auto gradient = activation.array()*(1.0F - activation.array());
      size_t before_layer_n_nodes = before_layer_output.cols();
      size_t cur_layer_n_nodes    = activation.cols();

      RL::MatrixX<float> weight_grad (BatchSize, before_layer_n_nodes*cur_layer_n_nodes);

      for (size_t i = 0u; i < cur_layer_n_nodes; i++)
      {
        for (size_t j = 0u; j < before_layer_n_nodes; j++) 
        {
          weight_grad(all, (i*before_layer_n_nodes + j)) = gradient(all, i).array() * before_layer_output(all, j).array();  
        }
      }

      return std::make_tuple(weight_grad, gradient);
    }
};

class ReLU: public ActivationBase{
  public:
    Initializer initializer;
    ReLU(){}
    ReLU(Initializer& param_initializer):initializer(param_initializer){}
    ~ReLU(){}

    template<typename EigenDerived>
    auto activation_batch(const eig::DenseBase<EigenDerived>& wx_b) const
    {  return wx_b.unaryExpr([](float v){return v < 0.0F?0.0F:v; });  }

    float activation(const float wx_b) const
    {  return wx_b < 0.0F?0.0F:wx_b;  }
    
    template<typename EigenDerived1, typename EigenDerived2>
    auto grad(const eig::DenseBase<EigenDerived1>& before_layer_output,
              const eig::DenseBase<EigenDerived2>& activation) const
    {
      auto gradient = activation.unaryExpr([](float v){return v < 0.0F?0.0F:1.0F;});
      size_t before_layer_n_nodes = before_layer_output.rows();
      size_t cur_layer_n_nodes    = activation.rows();

      RL::VectorX<float> weight_grad (before_layer_n_nodes*cur_layer_n_nodes);
      for (int i = 0u; i < cur_layer_n_nodes; i++)
      { weight_grad(seq(i*before_layer_n_nodes, (i+1)*before_layer_n_nodes-1)) = gradient.array() * before_layer_output.array();  }
      
      return std::make_tuple(weight_grad, gradient);
    }

    template<int BatchSize, typename EigenDerived1, typename EigenDerived2>
    auto grad_batch(const eig::DenseBase<EigenDerived1>& before_layer_output,
                    const eig::DenseBase<EigenDerived2>& activation) const
    {
      auto gradient = activation.unaryExpr([](float v){return v < 0.0F?0.0F:1.0F;});
      size_t before_layer_n_nodes = before_layer_output.cols();
      size_t cur_layer_n_nodes    = activation.cols();

      RL::MatrixX<float> weight_grad (BatchSize, before_layer_n_nodes*cur_layer_n_nodes);

      for (size_t i = 0u; i < cur_layer_n_nodes; i++)
      {
        for (size_t j = 0u; j < before_layer_n_nodes; j++) 
        {
          weight_grad(all, (i*before_layer_n_nodes + j)) = gradient(all, i).array() * before_layer_output(all, j).array();  
        }
      }

      return std::make_tuple(weight_grad, gradient);
    }
};


} // namespace {ANN}
#endif
