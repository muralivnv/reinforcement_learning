#ifndef _ANN_INITIALIZATION_H_
#define _ANN_INITIALIZATION_H_

#include <random>
#include <memory>

#include "../global_typedef.h"

namespace ann
{
#define UNIFORM (1)
#define NORMAL  (2)

#define NO_NORMALIZATION (0)
#define L2_NORMALIZATION (2)

// Weights Initialization
class ParamInitializerBase
{
  public:
    virtual ~ParamInitializerBase() {}
    virtual void initialize(const int n_nodes_prev_layer, 
                            eig::Ref<ArrayXf_t> weights) const = 0;
};

template<int SamplingDist=UNIFORM>
class ParamSampler
{
  public:
    void fill (const float stddev, eig::Ref<ArrayXf_t> weights) const
    {
      if constexpr (SamplingDist == UNIFORM)
      {
        uniform_dist_(-stddev, stddev, weights);
      }
      else if constexpr (SamplingDist == NORMAL)
      {
        normal_dist_(stddev, weights);
      }
      else
      { }
    }
  private:
    void uniform_dist_(const float left_bound, const float right_bound, 
                       eig::Ref<ArrayXf_t> weights) const
    {
      std::random_device seed;
      std::mt19937 rand_gen(seed());
      std::uniform_real_distribution<float> param_dist(left_bound, right_bound);

      for (int i = 0; i < weights.rows(); i++)
      {
        for (int j = 0; j < weights.cols(); j++)
        {
          weights(i, j) = param_dist(rand_gen);
          (void)param_dist(rand_gen);
        }
      }
    }
    
    void normal_dist_(const float stddev, eig::Ref<ArrayXf_t> weights) const
    {
      std::random_device seed;
      std::mt19937 rand_gen(seed());
      std::normal_distribution<float> param_dist(0.0F, stddev);

      for (int i = 0; i < weights.rows(); i++)
      {
        for (int j = 0; j < weights.cols(); j++)
        {
          weights(i, j) = param_dist(rand_gen);
          (void)param_dist(rand_gen);
        }
      }
    }
};

template<int NormalizationType=L2_NORMALIZATION>
class ParamNormalize 
{
  public:
    void normalize(eig::Ref<ArrayXf_t> weights) const
    {
      if constexpr (NormalizationType == L2_NORMALIZATION)
      {
        l2_normalize_(weights);
      }
      else if constexpr (NormalizationType != NO_NORMALIZATION)
      { }
    }
  
  private:
    void l2_normalize_(eig::Ref<ArrayXf_t> weights) const
    {
      const float l2_norm = std::sqrtf(weights.square().sum());
      weights /= l2_norm;
    }
};

template<int SamplingDist=UNIFORM, int NormalizationType=L2_NORMALIZATION>
class Xavier : public ParamInitializerBase, 
               public ParamSampler<SamplingDist>, 
               public ParamNormalize<NormalizationType>
{
  public:
    void initialize(const int n_nodes_prev_layer, 
                    eig::Ref<ArrayXf_t> weights) const override
    {
      const float stddev = std::sqrtf(1.0F/(float)n_nodes_prev_layer);
      this->fill(stddev, weights);
      this->normalize(weights);
    }
};

template<int SamplingDist, int NormalizationType>
class He : public ParamInitializerBase, 
           public ParamSampler<SamplingDist>, 
           public ParamNormalize<NormalizationType>
{
  public:
    void initialize(const int n_nodes_prev_layer, 
                    eig::Ref<ArrayXf_t> weights) const override
    {
      const float stddev = std::sqrtf(2.0F/(float)n_nodes_prev_layer);
      this->fill(stddev, weights);
      this->normalize(weights);
    }
};


constexpr static auto XavierUniform           = std::make_unique<Xavier<UNIFORM, NO_NORMALIZATION>>;
constexpr static auto XavierUniformNormalized = std::make_unique<Xavier<UNIFORM, L2_NORMALIZATION>>;
constexpr static auto XavierNormal            = std::make_unique<Xavier<NORMAL, NO_NORMALIZATION>>;
constexpr static auto XavierNormalNormalized  = std::make_unique<Xavier<NORMAL, L2_NORMALIZATION>>;

constexpr static auto HeUniform           = std::make_unique<He<UNIFORM, NO_NORMALIZATION>>;
constexpr static auto HeUniformNormalized = std::make_unique<He<UNIFORM, L2_NORMALIZATION>>;
constexpr static auto HeNormal            = std::make_unique<He<NORMAL, NO_NORMALIZATION>>;
constexpr static auto HeNormalNormalized  = std::make_unique< He<NORMAL, L2_NORMALIZATION> >;

} // namespace {ann}
#endif
