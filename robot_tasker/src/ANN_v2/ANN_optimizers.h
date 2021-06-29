#ifndef _ANN_OPTIMIZERS_H_
#define _ANN_OPTIMIZERS_H_

#include <random>
#include "../global_typedef.h"

#include "ANN_typedef.h"
#include "ANN.h"

namespace ANN
{
  
class SteepestDescentOptimizer{
  private:
    float step_size_;
  public:
    SteepestDescentOptimizer(float step_size=1e-3F)
    { step_size_ = step_size; }

    template<typename EigenDerived1, typename EigenDerived2, typename EigenDerived3, typename EigenDerived4>
    void step(const eig::ArrayBase<EigenDerived1>& weights_grad, 
              const eig::ArrayBase<EigenDerived2>& bias_grad,
              eig::ArrayBase<EigenDerived3>&       weights,
              eig::ArrayBase<EigenDerived4>&       bias)
    {
      weights -= (step_size_*weights_grad);
      bias    -= (step_size_*bias_grad);
    }
};

class AdamOptimizer{
  private:
    float step_size_;
    
    eig::Array<float, eig::Dynamic, 1> dw_momentum_;
    eig::Array<float, eig::Dynamic, 1> db_momentum_;
    float momentum_step_;
    float momentum_step_pow_;

    eig::Array<float, eig::Dynamic, 1> dw_sq_;
    eig::Array<float, eig::Dynamic, 1> db_sq_;
    float grad_sq_step_;
    float grad_sq_step_pow_;
    size_t optim_counter_;
  public:
    AdamOptimizer(int n_weights, int n_bias, float step_size=1e-3F, float momentum_step=0.90F, float grad_sq_step=0.99F)
    {
      dw_momentum_.resize(n_weights);
      db_momentum_.resize(n_bias);

      dw_sq_.resize(n_weights);
      db_sq_.resize(n_bias);

      step_size_     = step_size;
      momentum_step_ = momentum_step;
      grad_sq_step_  = grad_sq_step;
      
      optim_counter_  = 0u;
      momentum_step_pow_ = 1.0F;
      grad_sq_step_pow_  = 1.0F;
    }

    template<typename EigenDerived1, typename EigenDerived2, typename EigenDerived3, typename EigenDerived4>
    void step(const eig::ArrayBase<EigenDerived1>& weights_grad, 
              const eig::ArrayBase<EigenDerived2>& bias_grad,
              eig::ArrayBase<EigenDerived3>&       weights,
              eig::ArrayBase<EigenDerived4>&       bias)
    {
      optim_counter_++;
      // Initializer momentum and gradient square
      if (optim_counter_ == 1u)
      {
        // initialize momentum
        dw_momentum_ = weights_grad;
        db_momentum_ = bias_grad;

        // initialize uncentered 2nd moment of gradient
        dw_sq_ = weights_grad.square();
        db_sq_ = bias_grad.square();
      }
      else 
      {
        // update momentum
        dw_momentum_ *= momentum_step_;
        dw_momentum_ += (1.0F - momentum_step_)*weights_grad;

        db_momentum_ *= momentum_step_;
        db_momentum_ += (1.0F - momentum_step_)*bias_grad;

        // update 2nd moment
        auto dw_sq = weights_grad.square();
        auto db_sq = bias_grad.square();

        dw_sq_ *= grad_sq_step_;
        dw_sq_ += (1.0F - grad_sq_step_)*dw_sq;

        db_sq_ *= grad_sq_step_;
        db_sq_ += (1.0F - grad_sq_step_)*db_sq;
      }
      
      momentum_step_pow_ *= momentum_step_;
      grad_sq_step_pow_  *= grad_sq_step_;
      float momentum_bias_correction   = 1.0F/(1.0F - momentum_step_pow_);
      float grad_sq_bias_correction = 1.0F/(1.0F - grad_sq_step_pow_);
      auto dw_momentum_corrected  = dw_momentum_*momentum_bias_correction;
      auto db_momentum_corrected  = db_momentum_*momentum_bias_correction;
      auto dw_sq_corrected        = dw_sq_*grad_sq_bias_correction;
      auto db_sq_corrected        = db_sq_*grad_sq_bias_correction;

      auto delta_w = (step_size_*dw_momentum_corrected)/(eig::sqrt(dw_sq_corrected + 1e-8F));
      auto delta_b = (step_size_*db_momentum_corrected)/(eig::sqrt(db_sq_corrected  + 1e-8F));

      // update parameters
      weights -= delta_w;
      bias    -= delta_b;
    }

};

} // namespace {ANN}

#endif
