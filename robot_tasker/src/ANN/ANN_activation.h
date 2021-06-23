#ifndef _ANN_ACTIVATION_H_
#define _ANN_ACTIVATION_H_

#include <random>
#include "../global_typedef.h"
namespace ANN
{

enum InitializerIdentifier{
  RANDOM,
  XAVIER_UNIFORM,
  HE_UNIFORM
};

enum ActivationIdentifier{
  SIGMOID,
  RELU, 
  NONE
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
    auto init(eig::ArrayBase<EigenDerived1>& weights,
              eig::ArrayBase<EigenDerived2>& bias,
              int                            n_nodes_last_layer)
    {
      float std_dev = 0.1F;
      switch(init_type_)
      {
        case XAVIER_UNIFORM:
          std_dev = 1.0F/std::sqrtf((float)n_nodes_last_layer);
          break;
        case HE_UNIFORM:
          std_dev = 2.0F/std::sqrtf((float)n_nodes_last_layer);
          break;
        case RANDOM:
        default:
          std_dev = 0.1F;
          break;
      }
      std::uniform_real_distribution<float> weights_dist(-std_dev, std_dev);
      
      // fill weights
      for (size_t i = 0u; i < (size_t)weights.size(); i++)
      {
        weights[i] = weights_dist(rand_gen_);
        (void)weights_dist(rand_gen_);
      }

      // fill bias
      for (size_t i = 0u; i < (size_t)bias.size(); i++)
      {
        bias[i] = 0.0F;//0.5F*weights_dist(rand_gen_);
        // (void)weights_dist(rand_gen_);
      }
    }
};

class Activation{
  private:
    ActivationIdentifier activation_e;
  public:
    Initializer initializer;
    Activation(){}
    Activation(ActivationIdentifier&& activation, Initializer&& param_initializer) : activation_e(activation), 
                                                                                     initializer(param_initializer){}
    ~Activation(){}

    template<typename EigenDerived>
    auto activation_batch(const eig::ArrayBase<EigenDerived>& wx_b) const
    {
      eig::Array<float, eig::Dynamic, 1> retval(wx_b.rows());
      switch(activation_e)
      {
        case SIGMOID:
        {
          auto retval_expr = 1.0F/(1.0F + eig::exp(-wx_b));
          retval = retval_expr.eval();
          break;
        }
        case RELU:
        {
          // auto retval_expr = wx_b.unaryExpr([](float v){return v < 0.0F?0.0F:v; });
          auto retval_expr = wx_b.unaryExpr([](float v){return v < 0.0F?0.5F*v:v; }); // LEAKY-RELU
          retval = retval_expr.eval();
          break;
        }
        default:
        {
          retval = wx_b;
          break;
        }
      }
      return retval;
    }

    float activation(const float wx_b) const
    { 
      float retval;
      switch(activation_e)
      {
        case SIGMOID:
        {
          retval  = 1.0F/(1.0F + std::expf(-wx_b));
          break;
        }
        case RELU:
        {
          // retval = wx_b < 0.0F?0.0F:wx_b;
          retval = wx_b < 0.0F?0.5F*wx_b:wx_b; // LEAKY-RELU
          break;
        }
        default:
        {
          retval = wx_b;
          break;
        }
      }
      return retval;
    }
    
    template<typename EigenDerived>
    auto grad(const eig::ArrayBase<EigenDerived>& activation) const
    {
      eig::Array<float, eig::Dynamic, eig::Dynamic, eig::RowMajor> retval(activation.rows(), activation.cols());
      switch(activation_e)
      {
        case SIGMOID:
        {
          auto retval_expr = activation*(1.0F - activation);
          retval = retval_expr.eval();
          break;
        }
        case RELU:
        {
          // auto retval_expr = activation.unaryExpr([](float v){return v < 1e-8F?0.0F:1.0F;});
          auto retval_expr = activation.unaryExpr([](float v){return v < 1e-10F?0.5F:1.0F;}); // LEAKY-RELU
          retval = retval_expr.eval();
          break;
        }
        default:
        {
          retval.fill(1.0F);
          break;
        }
      }
      return retval;
    }
};


class InitializerBase
{
  public:
    virtual void initialize(const int n_nodes_prev_layer, 
                            eig::Ref<eig::ArrayXf> weights, 
                            eig::Ref<eig::ArrayXf> bias) const = 0;
};

class XavierUniform : public InitializerBase
{
  public:
    void initialize(const int n_nodes_prev_layer, 
                    eig::Ref<eig::ArrayXf> weights, 
                    eig::Ref<eig::ArrayXf> bias) const override
    {
      const float bound = 1.0F/std::sqrtf((float)n_nodes_prev_layer);
      std::random_device seed;
      std::mt19937 rand_gen(seed());
      std::uniform_real_distribution<float> param_dist(-bound, bound);
      auto param_filler = [&rand_gen, &param_dist](float val){val = param_dist(rand_gen); (void)param_dist(rand_gen); };

      std::for_each(weights.begin(), weights.end(), param_filler);
      std::for_each(bias.begin(), bias.end(), param_filler);
    }
};

class HeUniform : public InitializerBase
{
  public:
    void initialize(const int n_nodes_prev_layer, 
                     eig::Ref<eig::ArrayXf> weights, 
                     eig::Ref<eig::ArrayXf> bias) const override
    {
      const float bound = 2.0F/std::sqrtf((float)n_nodes_prev_layer);
      std::random_device seed;
      std::mt19937 rand_gen(seed());
      std::uniform_real_distribution<float> param_dist(-bound, bound);
      auto param_filler = [&rand_gen, &param_dist](float val){val = param_dist(rand_gen); (void)param_dist(rand_gen); };

      std::for_each(weights.begin(), weights.end(), param_filler);
      std::for_each(bias.begin(), bias.end(), param_filler);
    }
};

class ActivationBase
{
  public:
  virtual ~ActivationBase(){}
  virtual void activation(const eig::Ref<const eig::ArrayXf> wx_b, 
                                eig::Ref<eig::ArrayXf> out) const = 0;

  virtual void activation(const float wx_b, float& out) const = 0;

  virtual void grad(const eig::Ref<const eig::ArrayXf> z, 
                          eig::Ref<eig::ArrayXf> out) const = 0;
};

class NoActivation : public ActivationBase
{
  public:
    ~NoActivation(){}
    void activation(const eig::Ref<const eig::ArrayXf> wx_b, 
                          eig::Ref<eig::ArrayXf> out) const override
    {
      out = wx_b;
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = wx_b;
    }

    void grad(const eig::Ref<const eig::ArrayXf> z, 
                    eig::Ref<eig::ArrayXf> out) const override
    {
      out.fill(1.0F);
    }
};

class Sigmoid : public ActivationBase
{
  public:
    ~Sigmoid(){}
    void activation(const eig::Ref<const eig::ArrayXf> wx_b, 
                          eig::Ref<eig::ArrayXf> out) const override
    {
      out = 1.0F/(1.0F + eig::exp(-wx_b));
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = 1.0F/(1.0F + std::expf(-wx_b));
    }

    void grad(const eig::Ref<const eig::ArrayXf> z, 
                    eig::Ref<eig::ArrayXf> out) const override
    {
      out = z*(1.0F - z);
    }
};

class ReLU : public ActivationBase
{
  public:
    ~ReLU(){}
    void activation(const eig::Ref<const eig::ArrayXf> wx_b, 
                          eig::Ref<eig::ArrayXf> out) const override
    {
      out = wx_b.unaryExpr([](float v){return v < 0.0F?0.0F:v; });
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = (wx_b < 0.0F)?0.0F:wx_b;
    }

    void grad(const eig::Ref<const eig::ArrayXf> z, 
                    eig::Ref<eig::ArrayXf> out) const override
    {
      out = z.unaryExpr([](float v){return v < 0.0F?0.0F:1.0F;});
    }
};

class LeakyReLU : public ActivationBase
{
  private:
    float leakyness_factor_ = 0.5F;
  public:
    LeakyReLU() = default;
    LeakyReLU(const float alpha): leakyness_factor_(alpha){}
    ~LeakyReLU(){}
    void activation(const eig::Ref<const eig::ArrayXf> wx_b, 
                          eig::Ref<eig::ArrayXf> out) const override
    {

      out = wx_b.unaryExpr([this](float v){return v < 0.0F?this->leakyness_factor_*v:v; });
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = (wx_b < 0.0F)?this->leakyness_factor_*wx_b:wx_b;
    }

    void grad(const eig::Ref<const eig::ArrayXf> z, 
                    eig::Ref<eig::ArrayXf> out) const override
    {
      out = z.unaryExpr([this](float v){return v < 0.0F?this->leakyness_factor_:1.0F;});
    }
};


} // namespace {ANN}
#endif
