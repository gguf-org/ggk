#ifndef __SD_MODEL_VAE_AUDIO_VAE_HPP__
#define __SD_MODEL_VAE_AUDIO_VAE_HPP__

#include <map>
#include <string>

#include "core/tensor_ggml.hpp"
#include "model/common/block.hpp"

struct AudioVAERunner : public GGMLRunner {
    explicit AudioVAERunner(ggml_backend_t backend,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager) {}

    virtual ~AudioVAERunner() = default;

    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_mem_size()                                         = 0;
    virtual std::string get_desc()                                               = 0;
    virtual int output_sample_rate() const                                       = 0;
    virtual sd::Tensor<float> encode(int n_threads,
                                     const sd::Tensor<float>& waveform) {
        SD_UNUSED(n_threads);
        SD_UNUSED(waveform);
        return {};
    }
    virtual sd::Tensor<float> decode(int n_threads,
                                     const sd::Tensor<float>& latent_tensor)      = 0;
};

#endif  // __SD_MODEL_VAE_AUDIO_VAE_HPP__
