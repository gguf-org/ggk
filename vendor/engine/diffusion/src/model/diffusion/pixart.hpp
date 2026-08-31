#ifndef __SD_MODEL_DIFFUSION_PIXART_HPP__
#define __SD_MODEL_DIFFUSION_PIXART_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Ref: https://github.com/PixArt-alpha/PixArt-alpha
// Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
//
// PixArt-alpha / PixArt-Sigma: a plain DiT over 4-channel SD-VAE latents, with
// T5-XXL cross attention and no self-conditioning of any kind.  Checkpoints use
// the diffusers naming (pos_embed.proj, adaln_single, caption_projection,
// transformer_blocks.N.attn1/attn2/ff, proj_out), so no name conversion runs.
//
// The model is eps-prediction on the ADM/DDPM *linear* beta schedule (not SD's
// scaled_linear one) and predicts learned sigma: proj_out emits 2 * C values per
// patch and only the first C are the noise estimate.
//
// Positions are a fixed 2D sin-cos grid rather than RoPE.  The grid is not part
// of the checkpoint (diffusers registers it as a non-persistent buffer), so it is
// rebuilt on the host whenever the latent size changes, using the same
// base_size / interpolation_scale interpolation diffusers applies when the
// requested resolution differs from the model's native one.
namespace PixArt {
    constexpr int PIXART_GRAPH_SIZE = 20480;

    struct PixArtConfig {
        int64_t hidden_size      = 1152;
        int64_t num_layers       = 28;
        int64_t num_heads        = 16;
        int64_t head_dim         = 72;
        int64_t in_channels      = 4;
        int64_t out_channels     = 8;  // learned sigma: eps and variance interleaved
        int64_t patch_size       = 2;
        int64_t caption_channels = 4096;
        int64_t caption_length   = 120;
        int64_t sample_size      = 64;  // native latent side the position grid was trained at
        float interpolation_scale = 1.0f;
        float norm_eps            = 1e-6f;
        bool use_additional_conditions = false;  // PixArt-alpha 1024 micro-conditioning

        int64_t base_size() const {
            return std::max<int64_t>(1, sample_size / patch_size);
        }

        static PixArtConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                const std::string& prefix) {
            PixArtConfig config;
            int64_t detected_layers = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "pos_embed.proj.weight") && tensor_storage.n_dims == 4) {
                    config.patch_size  = tensor_storage.ne[0];
                    config.in_channels = tensor_storage.ne[2];
                    config.hidden_size = tensor_storage.ne[3];
                } else if (ends_with(name, "caption_projection.linear_1.weight") && tensor_storage.n_dims == 2) {
                    config.caption_channels = tensor_storage.ne[0];
                    config.hidden_size      = tensor_storage.ne[1];
                } else if (ends_with(name, "caption_projection.y_embedding") && tensor_storage.n_dims == 2) {
                    config.caption_length = tensor_storage.ne[1];
                } else if (ends_with(name, "adaln_single.emb.resolution_embedder.linear_1.weight")) {
                    config.use_additional_conditions = true;
                }

                size_t pos = name.find("transformer_blocks.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int64_t idx     = atoi(items[1].c_str());
                        detected_layers = std::max<int64_t>(detected_layers, idx + 1);
                    }
                }
            }

            // proj_out folds the patch area and the doubled (learned-sigma) channel
            // count into one dimension; it needs patch_size, so resolve it last.
            auto proj_out = tensor_storage_map.find(prefix + ".proj_out.weight");
            if (proj_out != tensor_storage_map.end() && proj_out->second.n_dims == 2) {
                int64_t patch_area = config.patch_size * config.patch_size;
                if (patch_area > 0) {
                    config.out_channels = proj_out->second.ne[1] / patch_area;
                }
            }

            if (detected_layers > 0) {
                config.num_layers = detected_layers;
            }
            // PixArt XL/2 is 16 heads of 72; nothing in the checkpoint says so (q, k and
            // v are all square), so fall back to 64-wide heads for any other width.
            config.head_dim  = (config.hidden_size % 72 == 0) ? 72 : 64;
            config.num_heads = std::max<int64_t>(1, config.hidden_size / config.head_dim);

            // The position grid's native size is likewise config-only. The 1024 alpha
            // model is the one that carries micro-conditioning embedders; Sigma marks
            // itself by its 300-token caption window.
            if (config.use_additional_conditions || config.caption_length >= 300) {
                config.sample_size         = 128;
                config.interpolation_scale = 2.0f;
            }
            if (const char* env = getenv("SD_PIXART_SAMPLE_SIZE")) {
                config.sample_size = std::max<int64_t>(config.patch_size, atoi(env));
            }
            if (const char* env = getenv("SD_PIXART_INTERPOLATION_SCALE")) {
                float scale = (float)atof(env);
                if (scale > 0.f) {
                    config.interpolation_scale = scale;
                }
            }

            LOG_DEBUG("pixart: hidden_size=%" PRId64 ", layers=%" PRId64 ", heads=%" PRId64 ", head_dim=%" PRId64
                      ", patch=%" PRId64 ", in_channels=%" PRId64 ", out_channels=%" PRId64
                      ", caption=%" PRId64 "x%" PRId64 ", sample_size=%" PRId64 ", interpolation_scale=%.2f",
                      config.hidden_size,
                      config.num_layers,
                      config.num_heads,
                      config.head_dim,
                      config.patch_size,
                      config.in_channels,
                      config.out_channels,
                      config.caption_channels,
                      config.caption_length,
                      config.sample_size,
                      config.interpolation_scale);
            return config;
        }
    };

    // diffusers get_2d_sincos_pos_embed: the first half of the channels encodes the
    // column, the second half the row, each as [sin(...), cos(...)] over
    // omega_i = 10000^(-i / (embed_dim/4)).
    inline std::vector<float> make_2d_sincos_pos_embed(int64_t grid_h,
                                                       int64_t grid_w,
                                                       int64_t embed_dim,
                                                       int64_t base_size,
                                                       float interpolation_scale) {
        GGML_ASSERT(embed_dim % 4 == 0);
        const int64_t half    = embed_dim / 2;
        const int64_t quarter = half / 2;

        std::vector<double> omega(quarter);
        for (int64_t i = 0; i < quarter; ++i) {
            omega[i] = 1.0 / std::pow(10000.0, (double)i / (double)quarter);
        }

        // grid = arange(n) / (n / base_size) / interpolation_scale
        const double scale   = interpolation_scale > 0.f ? (double)interpolation_scale : 1.0;
        const double step_h  = (double)base_size / ((double)grid_h * scale);
        const double step_w  = (double)base_size / ((double)grid_w * scale);

        std::vector<float> out((size_t)grid_h * (size_t)grid_w * (size_t)embed_dim);
        for (int64_t y = 0; y < grid_h; ++y) {
            for (int64_t x = 0; x < grid_w; ++x) {
                const size_t base = ((size_t)(y * grid_w + x)) * (size_t)embed_dim;
                const double py   = (double)y * step_h;
                const double px   = (double)x * step_w;
                for (int64_t i = 0; i < quarter; ++i) {
                    const double aw = px * omega[i];
                    const double ah = py * omega[i];
                    out[base + i]                  = (float)std::sin(aw);
                    out[base + quarter + i]        = (float)std::cos(aw);
                    out[base + half + i]           = (float)std::sin(ah);
                    out[base + half + quarter + i] = (float)std::cos(ah);
                }
            }
        }
        return out;
    }

    // diffusers PatchEmbed(pos_embed_type="sincos", flatten=True, bias=True)
    struct PatchEmbed : public GGMLBlock {
        PatchEmbed(int64_t in_channels, int64_t hidden_size, int64_t patch_size) {
            blocks["proj"] = std::make_shared<Conv2d>(in_channels,
                                                      hidden_size,
                                                      std::pair<int, int>{(int)patch_size, (int)patch_size},
                                                      std::pair<int, int>{(int)patch_size, (int)patch_size});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            // x: [N, C, H, W] -> [N, h*w, hidden]
            auto proj = std::dynamic_pointer_cast<Conv2d>(blocks["proj"]);
            x         = proj->forward(ctx, x);  // [N, hidden, h, w]

            const int64_t w = x->ne[0];
            const int64_t h = x->ne[1];
            const int64_t c = x->ne[2];
            const int64_t n = x->ne[3];

            x = ggml_ext_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 2, 0, 3));  // [N, h, w, hidden]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, c, w * h, n);                            // [N, h*w, hidden]
            return x;
        }
    };

    // diffusers TimestepEmbedding
    struct TimestepEmbedder : public GGMLBlock {
        int64_t frequency_embedding_size;

        TimestepEmbedder(int64_t hidden_size, int64_t frequency_embedding_size = 256)
            : frequency_embedding_size(frequency_embedding_size) {
            blocks["linear_1"] = std::make_shared<Linear>(frequency_embedding_size, hidden_size);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            // Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)
            auto t = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, (int)frequency_embedding_size);
            t      = linear_1->forward(ctx, t);
            t      = ggml_silu(ctx->ggml_ctx, t);
            t      = linear_2->forward(ctx, t);
            return t;  // [N, hidden]
        }
    };

    // diffusers AdaLayerNormSingle over PixArtAlphaCombinedTimestepSizeEmbeddings
    struct AdaLayerNormSingle : public GGMLBlock {
        struct CombinedTimestepEmbeddings : public GGMLBlock {
            bool use_additional_conditions;

            CombinedTimestepEmbeddings(int64_t hidden_size, bool use_additional_conditions)
                : use_additional_conditions(use_additional_conditions) {
                blocks["timestep_embedder"] = std::make_shared<TimestepEmbedder>(hidden_size);
                if (use_additional_conditions) {
                    int64_t size_emb_dim           = hidden_size / 3;
                    blocks["resolution_embedder"]  = std::make_shared<TimestepEmbedder>(size_emb_dim);
                    blocks["aspect_ratio_embedder"] = std::make_shared<TimestepEmbedder>(size_emb_dim);
                }
            }

            ggml_tensor* forward(GGMLRunnerContext* ctx,
                                 ggml_tensor* timesteps,
                                 ggml_tensor* resolution,
                                 ggml_tensor* aspect_ratio) {
                auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["timestep_embedder"]);
                auto conditioning      = timestep_embedder->forward(ctx, timesteps);

                if (use_additional_conditions && resolution != nullptr && aspect_ratio != nullptr) {
                    auto resolution_embedder   = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["resolution_embedder"]);
                    auto aspect_ratio_embedder = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["aspect_ratio_embedder"]);

                    // resolution carries height and width as two timesteps; the two
                    // embeddings are concatenated, not summed.
                    auto res_emb = resolution_embedder->forward(ctx, resolution);
                    res_emb      = ggml_reshape_2d(ctx->ggml_ctx,
                                              res_emb,
                                              res_emb->ne[0] * 2,
                                              res_emb->ne[1] / 2);
                    auto ar_emb  = aspect_ratio_embedder->forward(ctx, aspect_ratio);

                    conditioning = ggml_concat(ctx->ggml_ctx, conditioning, res_emb, 0);
                    conditioning = ggml_concat(ctx->ggml_ctx, conditioning, ar_emb, 0);
                }
                return conditioning;
            }
        };

        AdaLayerNormSingle(int64_t hidden_size, bool use_additional_conditions) {
            blocks["emb"]    = std::make_shared<CombinedTimestepEmbeddings>(hidden_size, use_additional_conditions);
            blocks["linear"] = std::make_shared<Linear>(hidden_size, 6 * hidden_size);
        }

        // returns {6 * hidden modulation, embedded timestep}
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* timesteps,
                                                      ggml_tensor* resolution,
                                                      ggml_tensor* aspect_ratio) {
            auto emb    = std::dynamic_pointer_cast<CombinedTimestepEmbeddings>(blocks["emb"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto embedded_timestep = emb->forward(ctx, timesteps, resolution, aspect_ratio);
            auto modulation        = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, embedded_timestep));
            return {modulation, embedded_timestep};
        }
    };

    // diffusers PixArtAlphaTextProjection(act_fn="gelu_tanh")
    struct CaptionProjection : public GGMLBlock {
        CaptionProjection(int64_t in_features, int64_t hidden_size) {
            blocks["linear_1"] = std::make_shared<Linear>(in_features, hidden_size);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* caption) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto x = linear_1->forward(ctx, caption);
            x      = ggml_ext_gelu(ctx->ggml_ctx, x);
            x      = linear_2->forward(ctx, x);
            return x;
        }
    };

    // diffusers Attention with a plain (non-fused) to_q/to_k/to_v/to_out.0
    struct Attention : public GGMLBlock {
        int64_t num_heads;

        Attention(int64_t query_dim, int64_t context_dim, int64_t num_heads)
            : num_heads(num_heads) {
            blocks["to_q"]     = std::make_shared<Linear>(query_dim, query_dim);
            blocks["to_k"]     = std::make_shared<Linear>(context_dim, query_dim);
            blocks["to_v"]     = std::make_shared<Linear>(context_dim, query_dim);
            blocks["to_out.0"] = std::make_shared<Linear>(query_dim, query_dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context = nullptr,
                             ggml_tensor* mask    = nullptr) {
            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto kv = context != nullptr ? context : x;
            auto q  = to_q->forward(ctx, x);
            auto k  = to_k->forward(ctx, kv);
            auto v  = to_v->forward(ctx, kv);

            auto out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              num_heads,
                                              mask,
                                              false,
                                              ctx->flash_attn_enabled);
            return to_out_0->forward(ctx, out);
        }
    };

    // diffusers FeedForward(activation_fn="gelu-approximate")
    struct FeedForward : public GGMLBlock {
        FeedForward(int64_t dim, int64_t mult = 4) {
            blocks["net.0.proj"] = std::make_shared<Linear>(dim, dim * mult);
            blocks["net.2"]      = std::make_shared<Linear>(dim * mult, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["net.0.proj"]);
            auto out  = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

            x = proj->forward(ctx, x);
            x = ggml_ext_gelu(ctx->ggml_ctx, x);
            x = out->forward(ctx, x);
            return x;
        }
    };

    // diffusers BasicTransformerBlock(norm_type="ada_norm_single").
    // norm1/norm2 are affine-free LayerNorms, so neither carries weights; the
    // cross attention deliberately runs on the *unnormalized* hidden states.
    struct BasicTransformerBlock : public GGMLBlock {
        int64_t dim;
        float norm_eps;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 6);
        }

        BasicTransformerBlock(int64_t dim, int64_t num_heads, int64_t context_dim, float norm_eps)
            : dim(dim), norm_eps(norm_eps) {
            blocks["attn1"] = std::make_shared<Attention>(dim, dim, num_heads);
            blocks["attn2"] = std::make_shared<Attention>(dim, context_dim, num_heads);
            blocks["ff"]    = std::make_shared<FeedForward>(dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* modulation,
                             ggml_tensor* context_mask) {
            auto attn1 = std::dynamic_pointer_cast<Attention>(blocks["attn1"]);
            auto attn2 = std::dynamic_pointer_cast<Attention>(blocks["attn2"]);
            auto ff    = std::dynamic_pointer_cast<FeedForward>(blocks["ff"]);

            const int64_t batch = modulation->ne[1];
            auto table          = ggml_reshape_3d(ctx->ggml_ctx, params["scale_shift_table"], dim, 6, 1);
            auto t              = ggml_reshape_3d(ctx->ggml_ctx, modulation, dim, 6, batch);
            auto shape          = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, dim, 6, batch);
            auto mods           = ggml_add(ctx->ggml_ctx,
                                 ggml_repeat(ctx->ggml_ctx, table, shape),
                                 ggml_repeat(ctx->ggml_ctx, t, shape));
            auto chunks         = ggml_ext_chunk(ctx->ggml_ctx, mods, 6, 1);  // each [N, 1, dim]

            // 1. self attention
            auto h = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, norm_eps);
            h      = Flux::modulate(ctx->ggml_ctx, h, chunks[0], chunks[1], true);
            h      = attn1->forward(ctx, h, nullptr, nullptr);
            x      = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, h, chunks[2]));

            // 2. cross attention (no norm, no gate)
            h = attn2->forward(ctx, x, context, context_mask);
            x = ggml_add(ctx->ggml_ctx, x, h);

            // 3. feed forward
            h = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, norm_eps);
            h = Flux::modulate(ctx->ggml_ctx, h, chunks[3], chunks[4], true);
            h = ff->forward(ctx, h);
            x = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, h, chunks[5]));

            return x;
        }
    };

    struct PixArtTransformer2DModel : public GGMLBlock {
        PixArtConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, 2);
        }

        PixArtTransformer2DModel(const PixArtConfig& config)
            : config(config) {
            blocks["pos_embed"]          = std::make_shared<PatchEmbed>(config.in_channels, config.hidden_size, config.patch_size);
            blocks["adaln_single"]       = std::make_shared<AdaLayerNormSingle>(config.hidden_size, config.use_additional_conditions);
            blocks["caption_projection"] = std::make_shared<CaptionProjection>(config.caption_channels, config.hidden_size);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] =
                    std::make_shared<BasicTransformerBlock>(config.hidden_size,
                                                            config.num_heads,
                                                            config.hidden_size,
                                                            config.norm_eps);
            }
            blocks["proj_out"] = std::make_shared<Linear>(config.hidden_size,
                                                          config.patch_size * config.patch_size * config.out_channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timesteps,
                             ggml_tensor* context,
                             ggml_tensor* context_mask,
                             ggml_tensor* pos_embed,
                             ggml_tensor* resolution,
                             ggml_tensor* aspect_ratio) {
            auto patch_embed       = std::dynamic_pointer_cast<PatchEmbed>(blocks["pos_embed"]);
            auto adaln_single      = std::dynamic_pointer_cast<AdaLayerNormSingle>(blocks["adaln_single"]);
            auto caption_proj      = std::dynamic_pointer_cast<CaptionProjection>(blocks["caption_projection"]);
            auto proj_out          = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            const int64_t W = x->ne[0];
            const int64_t H = x->ne[1];
            const int64_t w = W / config.patch_size;
            const int64_t h = H / config.patch_size;

            auto latent = patch_embed->forward(ctx, x);              // [N, h*w, hidden]
            latent      = ggml_add(ctx->ggml_ctx, latent, pos_embed);

            auto [modulation, embedded_timestep] = adaln_single->forward(ctx, timesteps, resolution, aspect_ratio);

            auto caption = caption_proj->forward(ctx, context);  // [N, caption_len, hidden]

            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<BasicTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                latent     = block->forward(ctx, latent, caption, modulation, context_mask);
                sd::ggml_graph_cut::mark_graph_cut(latent, "pixart.transformer_blocks." + std::to_string(i), "x");
            }

            // final adaLN: scale_shift_table[None] + embedded_timestep[:, None]
            const int64_t batch = embedded_timestep->ne[1];
            auto table          = ggml_reshape_3d(ctx->ggml_ctx, params["scale_shift_table"], config.hidden_size, 2, 1);
            auto t              = ggml_reshape_3d(ctx->ggml_ctx, embedded_timestep, config.hidden_size, 1, batch);
            auto shape          = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32, config.hidden_size, 2, batch);
            auto mods           = ggml_add(ctx->ggml_ctx,
                                 ggml_repeat(ctx->ggml_ctx, table, shape),
                                 ggml_repeat(ctx->ggml_ctx, t, shape));
            auto chunks         = ggml_ext_chunk(ctx->ggml_ctx, mods, 2, 1);

            latent = ggml_ext_layer_norm(ctx->ggml_ctx, latent, nullptr, nullptr, config.norm_eps);
            latent = Flux::modulate(ctx->ggml_ctx, latent, chunks[0], chunks[1], true);
            latent = proj_out->forward(ctx, latent);  // [N, h*w, ph*pw*out_channels]

            // diffusers unpatchify is "nhwpqc->nchpwq": the channel runs fastest.
            auto out = DiT::unpatchify(ctx->ggml_ctx,
                                       latent,
                                       h,
                                       w,
                                       (int)config.patch_size,
                                       (int)config.patch_size,
                                       false);  // [N, out_channels, H, W]

            // learned sigma: keep the noise prediction, drop the variance half
            if (config.out_channels > config.in_channels) {
                out = ggml_ext_slice(ctx->ggml_ctx, out, 2, 0, config.in_channels);
            }
            return out;
        }
    };

    struct PixArtRunner : public DiffusionModelRunner {
        PixArtConfig config;
        PixArtTransformer2DModel model;

        ggml_context* position_cache_ctx            = nullptr;
        ggml_backend_buffer_t position_cache_buffer = nullptr;
        ggml_tensor* cached_pos_embed               = nullptr;
        int64_t cached_grid_h                       = -1;
        int64_t cached_grid_w                       = -1;

        PixArtRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map      = {},
                     const std::string prefix                            = "",
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(PixArtConfig::detect_from_weights(tensor_storage_map, this->prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, this->prefix);
        }

        ~PixArtRunner() override {
            free_position_cache();
        }

        std::string get_desc() override {
            return "PixArt";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        void free_position_cache() {
            if (position_cache_buffer != nullptr) {
                ggml_backend_buffer_free(position_cache_buffer);
                position_cache_buffer = nullptr;
            }
            if (position_cache_ctx != nullptr) {
                ggml_free(position_cache_ctx);
                position_cache_ctx = nullptr;
            }
            cached_pos_embed = nullptr;
            cached_grid_h    = -1;
            cached_grid_w    = -1;
        }

        void ensure_position_cache(int64_t grid_h, int64_t grid_w) {
            if (cached_pos_embed != nullptr && cached_grid_h == grid_h && cached_grid_w == grid_w) {
                return;
            }
            free_position_cache();

            auto pos_embed_vec = make_2d_sincos_pos_embed(grid_h,
                                                          grid_w,
                                                          config.hidden_size,
                                                          config.base_size(),
                                                          config.interpolation_scale);

            ggml_init_params params;
            params.mem_size    = ggml_tensor_overhead();
            params.mem_buffer  = nullptr;
            params.no_alloc    = true;
            position_cache_ctx = ggml_init(params);
            GGML_ASSERT(position_cache_ctx != nullptr);

            cached_pos_embed = ggml_new_tensor_3d(position_cache_ctx,
                                                  GGML_TYPE_F32,
                                                  config.hidden_size,
                                                  grid_h * grid_w,
                                                  1);
            ggml_set_name(cached_pos_embed, "pixart.pos_embed");

            position_cache_buffer = ggml_backend_alloc_ctx_tensors(position_cache_ctx, runtime_backend);
            GGML_ASSERT(position_cache_buffer != nullptr);
            ggml_backend_buffer_set_usage(position_cache_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(cached_pos_embed, pos_embed_vec.data(), 0, ggml_nbytes(cached_pos_embed));
            ggml_backend_synchronize(runtime_backend);

            cached_grid_h = grid_h;
            cached_grid_w = grid_w;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& context_mask_tensor,
                                 const sd::Tensor<float>& resolution_tensor,
                                 const sd::Tensor<float>& aspect_ratio_tensor) {
            ggml_cgraph* gf           = new_graph_custom(PIXART_GRAPH_SIZE);
            ggml_tensor* x            = make_input(x_tensor);
            ggml_tensor* timesteps    = make_input(timesteps_tensor);
            ggml_tensor* context      = make_input(context_tensor);
            ggml_tensor* context_mask = make_optional_input(context_mask_tensor);
            ggml_tensor* resolution   = make_optional_input(resolution_tensor);
            ggml_tensor* aspect_ratio = make_optional_input(aspect_ratio_tensor);

            if (context_mask != nullptr) {
                // ggml_ext_attention_ext adds the mask to the [L_k, L_q] scores and
                // broadcasts a singleton query dimension.
                context_mask = ggml_reshape_2d(compute_ctx, context_mask, context_mask->ne[0], 1);
            }

            ensure_position_cache(x->ne[1] / config.patch_size, x->ne[0] / config.patch_size);

            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx,
                                     x,
                                     timesteps,
                                     context,
                                     context_mask,
                                     cached_pos_embed,
                                     resolution,
                                     aspect_ratio);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);

            const sd::Tensor<float>& x         = *diffusion_params.x;
            const sd::Tensor<float>& timesteps = *diffusion_params.timesteps;
            const sd::Tensor<float>& context   = *diffusion_params.context;
            static const sd::Tensor<float> empty;
            const sd::Tensor<float>& mask = diffusion_params.y != nullptr ? *diffusion_params.y : empty;

            sd::Tensor<float> resolution;
            sd::Tensor<float> aspect_ratio;
            if (config.use_additional_conditions) {
                float height = (float)(x.shape()[1] * 8);
                float width  = (float)(x.shape()[0] * 8);
                resolution   = sd::Tensor<float>::from_vector(std::vector<float>{height, width});
                aspect_ratio = sd::Tensor<float>::from_vector(std::vector<float>{height / width});
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, mask, resolution, aspect_ratio);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }
    };
}  // namespace PixArt

#endif  // __SD_MODEL_DIFFUSION_PIXART_HPP__
