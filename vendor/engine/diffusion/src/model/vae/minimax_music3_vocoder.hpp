#ifndef __SD_MODEL_VAE_MINIMAX_MUSIC3_VOCODER_HPP__
#define __SD_MODEL_VAE_MINIMAX_MUSIC3_VOCODER_HPP__

#include <cmath>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/vae/audio_vae.hpp"

// MiniMax Music 3 vocoder: a DAC-style decoder over the flow latents.
// [frames, 128] latents split into two stereo halves of 64 channels; each half
// runs dec_in_proj (1x1) -> weight-norm conv stack with snake activations and
// 4 transposed-conv upsamples (8, 8, 4, 2 -> hop 512) -> tanh, giving
// 44.1 kHz audio, one batch pass per stereo channel (gk's conv_1d family
// scrambles batched input, see model/vae/ace_vae.hpp).
//
// The checkpoint keeps torch weight-norm form (weight_v / weight_g); folding
// happens inside the graph, matching the ACE VAE approach.
namespace MiniMaxMusic3 {

    struct Music3VocoderConfig {
        int64_t latent_channels   = 128;
        int64_t decoder_input_dim = 1024;
        int64_t decoder_hidden    = 1536;
        int sample_rate           = 44100;
        int64_t hop_length        = 512;
        std::vector<int> upsample_ratios = {8, 8, 4, 2};

        static Music3VocoderConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                       const std::string& prefix) {
            Music3VocoderConfig config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = tensor_storage_map.find(prefix + "." + suffix);
                return it != tensor_storage_map.end() ? &it->second : nullptr;
            };
            if (const TensorStorage* w = find("dec_in_proj.weight")) {
                config.latent_channels   = w->ne[1] * 2;
                config.decoder_input_dim = w->ne[2];
            }
            if (const TensorStorage* w = find("decoder.model.0.weight_v")) {
                config.decoder_hidden = w->ne[2];
            }
            return config;
        }
    };

    // w = v * g / ||v||, norm over (ne0, ne1) per ne2 slice
    __STATIC_INLINE__ ggml_tensor* mm3_fold_weight_norm(ggml_context* ctx,
                                                        ggml_tensor* weight_v,
                                                        ggml_tensor* weight_g) {
        const int64_t k  = weight_v->ne[0];
        const int64_t c1 = weight_v->ne[1];
        const int64_t c2 = weight_v->ne[2];
        auto v           = weight_v;
        if (v->type != GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F32);
        }
        auto flat = ggml_reshape_2d(ctx, v, k * c1, c2);
        auto norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, flat)));  // [1, c2]
        auto g    = weight_g;
        if (g->type != GGML_TYPE_F32) {
            g = ggml_cast(ctx, g, GGML_TYPE_F32);
        }
        g          = ggml_reshape_2d(ctx, g, 1, c2);
        auto scale = ggml_div(ctx, g, norm);      // [1, c2]
        auto w     = ggml_mul(ctx, flat, scale);  // [k*c1, c2]
        return ggml_reshape_3d(ctx, w, k, c1, c2);
    }

    // snake(x) = x + sin^2(alpha * x) / alpha, alpha per channel of [L, C, N]
    struct Music3Snake1d : public UnaryBlock {
        int64_t channels;

        Music3Snake1d(int64_t channels)
            : channels(channels) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["alpha"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx  = ctx->ggml_ctx;
            auto alpha = params["alpha"];
            auto ax    = ggml_mul(gctx, x, alpha);
            auto s     = ggml_sin(gctx, ax);
            auto s2    = ggml_mul(gctx, s, s);
            auto frac  = ggml_div(gctx, s2, alpha);
            return ggml_add(gctx, x, frac);
        }
    };

    struct Music3WNConv1d : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int padding;
        int dilation;

        Music3WNConv1d(int64_t in_channels, int64_t out_channels, int kernel_size, int padding, int dilation = 1)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              padding(padding),
              dilation(dilation) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight_v"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
            params["weight_g"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, out_channels);
            params["bias"]     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto w    = mm3_fold_weight_norm(gctx, params["weight_v"], params["weight_g"]);
            x         = ggml_conv_1d(gctx, w, x, 1, padding, dilation);
            return ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
        }
    };

    struct Music3WNConvTranspose1d : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;

        Music3WNConvTranspose1d(int64_t in_channels, int64_t out_channels, int kernel_size, int stride, int padding)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight_v"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, out_channels, in_channels);
            params["weight_g"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, in_channels);
            params["bias"]     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto w    = mm3_fold_weight_norm(gctx, params["weight_v"], params["weight_g"]);
            x         = ggml_conv_transpose_1d(gctx, w, x, stride, 0, 1);
            if (padding > 0) {
                x = ggml_ext_slice(gctx, x, 0, padding, x->ne[0] - padding);
            }
            return ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
        }
    };

    // decoder.model.N.block.{2,3,4}: snake -> k7 dilated conv -> snake -> 1x1 conv, residual
    struct Music3ResidualUnit : public GGMLBlock {
        Music3ResidualUnit(int64_t channels, int dilation) {
            blocks["block.0"] = std::make_shared<Music3Snake1d>(channels);
            blocks["block.1"] = std::make_shared<Music3WNConv1d>(channels, channels, 7, 3 * dilation, dilation);
            blocks["block.2"] = std::make_shared<Music3Snake1d>(channels);
            blocks["block.3"] = std::make_shared<Music3WNConv1d>(channels, channels, 1, 0);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto h = std::dynamic_pointer_cast<Music3Snake1d>(blocks["block.0"])->forward(ctx, x);
            h      = std::dynamic_pointer_cast<Music3WNConv1d>(blocks["block.1"])->forward(ctx, h);
            h      = std::dynamic_pointer_cast<Music3Snake1d>(blocks["block.2"])->forward(ctx, h);
            h      = std::dynamic_pointer_cast<Music3WNConv1d>(blocks["block.3"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, h);
        }
    };

    // decoder.model.N (N in 1..4): snake -> transposed conv upsample -> 3 residual units
    struct Music3DecoderBlock : public GGMLBlock {
        Music3DecoderBlock(int64_t in_channels, int64_t out_channels, int stride) {
            blocks["block.0"] = std::make_shared<Music3Snake1d>(in_channels);
            blocks["block.1"] = std::make_shared<Music3WNConvTranspose1d>(in_channels, out_channels,
                                                                          2 * stride, stride, stride / 2);
            const int dilations[3] = {1, 3, 9};
            for (int i = 0; i < 3; i++) {
                blocks["block." + std::to_string(i + 2)] = std::make_shared<Music3ResidualUnit>(out_channels, dilations[i]);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<Music3Snake1d>(blocks["block.0"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Music3WNConvTranspose1d>(blocks["block.1"])->forward(ctx, x);
            for (int i = 0; i < 3; i++) {
                x = std::dynamic_pointer_cast<Music3ResidualUnit>(blocks["block." + std::to_string(i + 2)])->forward(ctx, x);
            }
            return x;
        }
    };

    struct Music3PlainConv1d : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int padding;

        Music3PlainConv1d(int64_t in_channels, int64_t out_channels, int kernel_size, int padding)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              padding(padding) {}

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            x         = ggml_conv_1d(gctx, params["weight"], x, 1, padding, 1);
            return ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
        }
    };

    struct Music3Vocoder : public GGMLBlock {
        Music3VocoderConfig config;

        explicit Music3Vocoder(const Music3VocoderConfig& config)
            : config(config) {
            blocks["dec_in_proj"] = std::make_shared<Music3PlainConv1d>(config.latent_channels / 2,
                                                                        config.decoder_input_dim, 1, 0);
            blocks["decoder.model.0"] = std::make_shared<Music3WNConv1d>(config.decoder_input_dim,
                                                                         config.decoder_hidden, 7, 3);
            int64_t channels = config.decoder_hidden;
            for (size_t i = 0; i < config.upsample_ratios.size(); i++) {
                blocks["decoder.model." + std::to_string(i + 1)] =
                    std::make_shared<Music3DecoderBlock>(channels, channels / 2, config.upsample_ratios[i]);
                channels /= 2;
            }
            blocks["decoder.model.5"] = std::make_shared<Music3Snake1d>(channels);
            blocks["decoder.model.6"] = std::make_shared<Music3WNConv1d>(channels, 1, 7, 3);
        }

        // one stereo channel: [frames, latent_channels/2, 1] -> [samples, 1, 1]
        ggml_tensor* decode_channel(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            auto x = std::dynamic_pointer_cast<Music3PlainConv1d>(blocks["dec_in_proj"])->forward(ctx, latent);
            x      = std::dynamic_pointer_cast<Music3WNConv1d>(blocks["decoder.model.0"])->forward(ctx, x);
            for (size_t i = 0; i < config.upsample_ratios.size(); i++) {
                x = std::dynamic_pointer_cast<Music3DecoderBlock>(blocks["decoder.model." + std::to_string(i + 1)])->forward(ctx, x);
            }
            x = std::dynamic_pointer_cast<Music3Snake1d>(blocks["decoder.model.5"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Music3WNConv1d>(blocks["decoder.model.6"])->forward(ctx, x);
            return ggml_tanh(ctx->ggml_ctx, x);
        }

        // latent: [frames, latent_channels, 1] -> [samples, 2]
        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            auto gctx            = ctx->ggml_ctx;
            const int64_t frames = latent->ne[0];
            auto stereo          = ggml_reshape_3d(gctx, latent, frames, config.latent_channels / 2, 2);

            ggml_tensor* waveform = nullptr;
            for (int64_t c = 0; c < 2; c++) {
                auto latent_c = ggml_ext_cont(gctx, ggml_ext_slice(gctx, stereo, 2, c, c + 1));  // [frames, 64, 1]
                auto wav_c    = decode_channel(ctx, latent_c);                                    // [samples, 1, 1]
                wav_c         = ggml_reshape_2d(gctx, wav_c, wav_c->ne[0], 1);
                waveform      = waveform == nullptr ? wav_c : ggml_concat(gctx, waveform, wav_c, 1);
            }
            return waveform;  // [samples, 2]
        }
    };

    struct Music3VocoderRunner : public ::AudioVAERunner {
        Music3VocoderConfig config;
        Music3Vocoder model;
        std::string weight_prefix;

        Music3VocoderRunner(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map,
                            const std::string& prefix                           = "first_stage_model",
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : ::AudioVAERunner(backend, weight_manager),
              weight_prefix(prefix),
              config(Music3VocoderConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
            model.get_param_tensors(tensors, weight_prefix);
        }

        size_t get_params_mem_size() override {
            return model.get_params_mem_size();
        }

        std::string get_desc() override {
            return "minimax_music3_vocoder";
        }

        int output_sample_rate() const override {
            return config.sample_rate;
        }

        // latent_tensor: [frames, latent_channels] -> [samples, 2]
        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& latent_tensor) override {
            int64_t t0     = ggml_time_ms();
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(16384);
                auto latent     = make_input(latent_tensor);
                auto runner_ctx = GGMLRunner::get_context();
                auto out        = model.decode(&runner_ctx, latent);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            int64_t t1  = ggml_time_ms();
            if (!result.has_value()) {
                return {};
            }
            LOG_DEBUG("minimax music3 vocoder decode: %.2fs", (t1 - t0) * 1.0f / 1000);
            return *result;
        }
    };

}  // namespace MiniMaxMusic3

#endif  // __SD_MODEL_VAE_MINIMAX_MUSIC3_VOCODER_HPP__
