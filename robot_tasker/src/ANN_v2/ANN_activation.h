#ifndef _ANN_ACTIVATION_H_
#define _ANN_ACTIVATION_H_

#include <random>
#include <memory>

#include "../global_typedef.h"

namespace ANN
{

class ActivationBase
{
  public:
  virtual ~ActivationBase(){}
  virtual void activation(const eig::Ref<const ArrayXf_t> wx_b, 
                                eig::Ref<ArrayXf_t> out) const = 0;

  virtual void activation(const float wx_b, float& out) const = 0;

  virtual void grad(const eig::Ref<const ArrayXf_t> z, 
                          eig::Ref<ArrayXf_t> out) const = 0;
};

class NoActivation : public ActivationBase
{
  public:
    ~NoActivation(){}
    void activation(const eig::Ref<const ArrayXf_t> wx_b, 
                          eig::Ref<ArrayXf_t> out) const override
    {
      out = wx_b;
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = wx_b;
    }

    void grad(const eig::Ref<const ArrayXf_t> z, 
                    eig::Ref<ArrayXf_t> out) const override
    {
      out.fill(1.0F);
    }
};

class Sigmoid : public ActivationBase
{
  public:
    ~Sigmoid(){}
    void activation(const eig::Ref<const ArrayXf_t> wx_b, 
                          eig::Ref<ArrayXf_t> out) const override
    {
      out = 1.0F/(1.0F + eig::exp(-wx_b));
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = 1.0F/(1.0F + std::expf(-wx_b));
    }

    void grad(const eig::Ref<const ArrayXf_t> z, 
                    eig::Ref<ArrayXf_t> out) const override
    {
      out = z*(1.0F - z);
    }
};

class ReLU : public ActivationBase
{
  public:
    ~ReLU(){}
    void activation(const eig::Ref<const ArrayXf_t> wx_b, 
                          eig::Ref<ArrayXf_t> out) const override
    {
      out = wx_b.unaryExpr([](float v){return v < 0.0F?0.0F:v; });
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = (wx_b < 0.0F)?0.0F:wx_b;
    }

    void grad(const eig::Ref<const ArrayXf_t> z, 
                    eig::Ref<ArrayXf_t> out) const override
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
    void activation(const eig::Ref<const ArrayXf_t> wx_b, 
                          eig::Ref<ArrayXf_t> out) const override
    {

      out = wx_b.unaryExpr([this](float v){return v < 0.0F?this->leakyness_factor_*v:v; });
    }
    
    void activation(const float wx_b, float& out) const override
    {
      out = (wx_b < 0.0F)?this->leakyness_factor_*wx_b:wx_b;
    }

    void grad(const eig::Ref<const ArrayXf_t> z, 
                    eig::Ref<ArrayXf_t> out) const override
    {
      out = z.unaryExpr([this](float v){return v < 0.0F?this->leakyness_factor_:1.0F;});
    }
};

constexpr auto NO_ACTIVATION = std::make_unique<NoActivation>;
constexpr auto SIGMOID       = std::make_unique<Sigmoid>;
constexpr auto RELU          = std::make_unique<ReLU>;
constexpr auto LEAKY_RELU    = std::make_unique<LeakyReLU>;

} // namespace {ANN}

#endif
