#ifndef __SD_MODEL_DIFFUSION_LUMINA2_HPP__
#define __SD_MODEL_DIFFUSION_LUMINA2_HPP__

#include <algorithm>

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/mmdit.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Ref: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
// Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_lumina2.py
//
// Lumina Image 2.0 NextDiT. Original checkpoint naming (layers.N.attention.qkv,
// feed_forward.w1/w2/w3, adaLN_modulation.1, ...). Text conditioning comes from
// Gemma2-2B hidden_states[-2]; latents are 16-channel flux-VAE.
//
// Convention: the pipeline feeds timestep = 1 - sigma (in [0, 1]) and the model
// output is negated so it matches the engine's flow prediction (noise - data).

namespace Lumina2 {
    constexpr int LUMINA2_GRAPH_SIZE = 20480;

    struct Lumina2Config {
        int patch_size             = 2;
        int64_t hidden_size        = 2304;
        int64_t in_channels        = 16;
        int64_t out_channels       = 16;
        int64_t num_layers         = 26;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 96;
        int64_t num_heads          = 24;
        int64_t num_kv_heads       = 8;
        int64_t multiple_of        = 256;
        int64_t ffn_inner_dim      = 9216;
        int64_t adaln_embed_dim    = 1024;  // min(hidden_size, 1024)
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2304;
        int theta                  = 10000;
        std::vector<int> axes_dim  = {32, 32, 32};
        int64_t axes_dim_sum       = 96;

        static Lumina2Config detect_from_weights(const String2TensorStorage& tensor_storage_map, const std::string& prefix) {
            Lumina2Config config;
            int64_t detected_layers          = 0;
            int64_t detected_refiner_layers  = 0;
            int64_t detected_context_refiner = 0;
            int64_t detected_head_dim        = 0;
            int64_t detected_qkv_dim         = 0;

            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "x_embedder.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area = config.patch_size * config.patch_size;
                    config.in_channels = tensor_storage.ne[0] / patch_area;
                    config.hidden_size = tensor_storage.ne[1];
                } else if (ends_with(name, "cap_embedder.1.weight") && tensor_storage.n_dims == 2) {
                    config.cap_feat_dim = tensor_storage.ne[0];
                    config.hidden_size  = tensor_storage.ne[1];
                } else if (ends_with(name, "layers.0.attention.q_norm.weight") && tensor_storage.n_dims == 1) {
                    detected_head_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "layers.0.attention.qkv.weight") && tensor_storage.n_dims == 2) {
                    detected_qkv_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "layers.0.feed_forward.w1.weight") && tensor_storage.n_dims == 2) {
                    config.ffn_inner_dim = tensor_storage.ne[1];
                } else if (ends_with(name, "final_layer.linear.weight") && tensor_storage.n_dims == 2) {
                    int64_t patch_area  = config.patch_size * config.patch_size;
                    config.out_channels = tensor_storage.ne[1] / patch_area;
                }

                size_t pos = name.find("layers.");
                if (pos != std::string::npos && name.find("stream_layers.") == std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        detected_layers = std::max<int64_t>(detected_layers, block_index + 1);
                    }
                }
                pos = name.find("noise_refiner.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index         = atoi(items[1].c_str());
                        detected_refiner_layers = std::max<int64_t>(detected_refiner_layers, block_index + 1);
                    }
                }
                pos = name.find("context_refiner.");
                if (pos != std::string::npos) {
                    auto items = split_string(name.substr(pos), '.');
                    if (items.size() > 1) {
                        int block_index          = atoi(items[1].c_str());
                        detected_context_refiner = std::max<int64_t>(detected_context_refiner, block_index + 1);
                    }
                }
            }
            if (detected_layers > 0) {
                config.num_layers = detected_layers;
            }
            if (detected_refiner_layers > 0 || detected_context_refiner > 0) {
                config.num_refiner_layers = std::max(detected_refiner_layers, detected_context_refiner);
            }
            if (detected_head_dim > 0) {
                config.head_dim  = detected_head_dim;
                config.num_heads = config.hidden_size / config.head_dim;
                if (detected_qkv_dim > 0) {
                    int64_t qkv_heads   = detected_qkv_dim / config.head_dim;
                    config.num_kv_heads = std::max<int64_t>(1, (qkv_heads - config.num_heads) / 2);
                }
            }
            config.adaln_embed_dim = std::min<int64_t>(config.hidden_size, 1024);
            config.axes_dim_sum    = config.head_dim;
            if (config.head_dim % 3 == 0) {
                int axis        = static_cast<int>(config.head_dim / 3);
                config.axes_dim = {axis, axis, axis};
            }
            LOG_DEBUG("lumina2: num_layers = %" PRId64 ", num_refiner_layers = %" PRId64 ", hidden_size = %" PRId64 ", num_heads = %" PRId64 ", num_kv_heads = %" PRId64 ", ffn_inner_dim = %" PRId64 ", in_channels = %" PRId64 ", out_channels = %" PRId64,
                      config.num_layers,
                      config.num_refiner_layers,
                      config.hidden_size,
                      config.num_heads,
                      config.num_kv_heads,
                      config.ffn_inner_dim,
                      config.in_channels,
                      config.out_channels);
            return config;
        }
    };

    struct JointAttention : public GGMLBlock {
    protected:
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        JointAttention(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
            : head_dim(head_dim), num_heads(num_heads), num_kv_heads(num_kv_heads), qk_norm(qk_norm) {
            blocks["qkv"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads * 2) * head_dim, false);
            float scale   = 1.f;
            blocks["out"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false, false, false, scale);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            if (sd_backend_is(ctx->backend, "ROCm")) {
                out_proj->set_scale(1.f / 16.f);
            }

            auto qkv = qkv_proj->forward(ctx, x);                                                                            // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
            qkv      = ggml_reshape_4d(ctx->ggml_ctx, qkv, head_dim, num_heads + num_kv_heads * 2, qkv->ne[1], qkv->ne[2]);  // [N, n_token, num_heads + num_kv_heads*2, head_dim]

            auto q = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  0);  // [N, n_token, num_heads, head_dim]
            auto k = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_kv_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  num_heads * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]
            auto v = ggml_view_4d(ctx->ggml_ctx,
                                  qkv,
                                  qkv->ne[0],
                                  num_kv_heads,
                                  qkv->ne[2],
                                  qkv->ne[3],
                                  qkv->nb[1],
                                  qkv->nb[2],
                                  qkv->nb[3],
                                  (num_heads + num_kv_heads) * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            x = Rope::attention(ctx, q, k, v, pe, mask, 1.f / 128.f);  // [N, n_token, num_heads * head_dim]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t inner_dim,
                    int64_t multiple_of) {
            inner_dim    = multiple_of * ((inner_dim + multiple_of - 1) / multiple_of);
            blocks["w1"] = std::make_shared<Linear>(dim, inner_dim, false);

            bool force_prec_f32 = false;
            float scale         = 1.f / 128.f;

            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example, when using CUDA but the weights are k-quants.
            blocks["w2"] = std::make_shared<Linear>(inner_dim, dim, false, false, force_prec_f32, scale);
            blocks["w3"] = std::make_shared<Linear>(dim, inner_dim, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            if (sd_backend_is(ctx->backend, "Vulkan")) {
                w2->set_force_prec_f32(true);
            }

            auto x1 = w1->forward(ctx, x);
            auto x3 = w3->forward(ctx, x);
            x       = ggml_swiglu_split(ctx->ggml_ctx, x1, x3);
            x       = w2->forward(ctx, x);

            return x;
        }
    };

    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        // return: x * (1 + scale)
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        return x;
    }

    struct JointTransformerBlock : public GGMLBlock {
    protected:
        bool modulation;

    public:
        JointTransformerBlock(int64_t hidden_size,
                              int64_t head_dim,
                              int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t multiple_of,
                              int64_t ffn_inner_dim,
                              int64_t adaln_embed_dim,
                              float norm_eps,
                              bool qk_norm,
                              bool modulation = true)
            : modulation(modulation) {
            blocks["attention"]       = std::make_shared<JointAttention>(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm);
            blocks["feed_forward"]    = std::make_shared<FeedForward>(hidden_size, ffn_inner_dim, multiple_of);
            blocks["attention_norm1"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm1"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["attention_norm2"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm2"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            if (modulation) {
                // nn.Sequential(nn.SiLU(), nn.Linear(min(dim, 1024), 4 * dim))
                blocks["adaLN_modulation.1"] = std::make_shared<Linear>(adaln_embed_dim, 4 * hidden_size);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe,
                             ggml_tensor* mask        = nullptr,
                             ggml_tensor* adaln_input = nullptr) {
            auto attention       = std::dynamic_pointer_cast<JointAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                GGML_ASSERT(adaln_input != nullptr);
                auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

                auto m         = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, adaln_input));  // [N, 4 * hidden_size]
                auto mods      = ggml_ext_chunk(ctx->ggml_ctx, m, 4, 0);
                auto scale_msa = mods[0];
                auto gate_msa  = mods[1];
                auto scale_mlp = mods[2];
                auto gate_mlp  = mods[3];

                auto residual = x;
                x             = modulate(ctx->ggml_ctx, attention_norm1->forward(ctx, x), scale_msa);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_msa));
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, x), scale_mlp);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_mlp));
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            } else {
                GGML_ASSERT(adaln_input == nullptr);

                auto residual = x;
                x             = attention_norm1->forward(ctx, x);
                x             = attention->forward(ctx, x, pe, mask);
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = ffn_norm1->forward(ctx, x);
                x        = feed_forward->forward(ctx, x);
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            }

            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    public:
        FinalLayer(int64_t hidden_size,
                   int64_t patch_size,
                   int64_t out_channels,
                   int64_t adaln_embed_dim) {
            blocks["norm_final"]         = std::make_shared<LayerNorm>(hidden_size, 1e-06f, false);
            blocks["linear"]             = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true, true);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(adaln_embed_dim, hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, adaln_embed_dim]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, hidden_size]
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);

            return x;
        }
    };

    // Positions: caption tokens at (i, 0, 0), image tokens at (cap_len, row, col).
    __STATIC_INLINE__ std::vector<float> gen_lumina2_pe(int h,
                                                        int w,
                                                        int patch_size,
                                                        int bs,
                                                        int context_len,
                                                        int theta,
                                                        const std::vector<int>& axes_dim) {
        auto txt_ids = std::vector<std::vector<float>>(static_cast<size_t>(bs) * context_len, std::vector<float>(3, 0.0f));
        for (int i = 0; i < bs * context_len; i++) {
            txt_ids[i][0] = static_cast<float>(i % context_len);
        }

        int axes_dim_num = 3;
        auto img_ids     = Rope::gen_flux_img_ids(h, w, patch_size, bs, axes_dim_num, context_len);

        auto ids = Rope::concat_ids(txt_ids, img_ids, bs);

        return Rope::embed_nd(ids, bs, static_cast<float>(theta), axes_dim);
    }

    class Lumina2Model : public GGMLBlock {
    protected:
        Lumina2Config config;

    public:
        Lumina2Model() = default;
        Lumina2Model(Lumina2Config config)
            : config(config) {
            blocks["x_embedder"]     = std::make_shared<Linear>(config.patch_size * config.patch_size * config.in_channels, config.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(config.adaln_embed_dim, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(config.cap_feat_dim, config.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(config.cap_feat_dim, config.hidden_size);
            // present in the checkpoint but never used by NextDiT.forward
            blocks["norm_final"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                blocks["noise_refiner." + std::to_string(i)] = std::make_shared<JointTransformerBlock>(config.hidden_size,
                                                                                                       config.head_dim,
                                                                                                       config.num_heads,
                                                                                                       config.num_kv_heads,
                                                                                                       config.multiple_of,
                                                                                                       config.ffn_inner_dim,
                                                                                                       config.adaln_embed_dim,
                                                                                                       config.norm_eps,
                                                                                                       config.qk_norm,
                                                                                                       true);
            }

            for (int i = 0; i < config.num_refiner_layers; i++) {
                blocks["context_refiner." + std::to_string(i)] = std::make_shared<JointTransformerBlock>(config.hidden_size,
                                                                                                         config.head_dim,
                                                                                                         config.num_heads,
                                                                                                         config.num_kv_heads,
                                                                                                         config.multiple_of,
                                                                                                         config.ffn_inner_dim,
                                                                                                         config.adaln_embed_dim,
                                                                                                         config.norm_eps,
                                                                                                         config.qk_norm,
                                                                                                         false);
            }

            for (int i = 0; i < config.num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::make_shared<JointTransformerBlock>(config.hidden_size,
                                                                                                config.head_dim,
                                                                                                config.num_heads,
                                                                                                config.num_kv_heads,
                                                                                                config.multiple_of,
                                                                                                config.ffn_inner_dim,
                                                                                                config.adaln_embed_dim,
                                                                                                config.norm_eps,
                                                                                                config.qk_norm,
                                                                                                true);
            }

            blocks["final_layer"] = std::make_shared<FinalLayer>(config.hidden_size, config.patch_size, config.out_channels, config.adaln_embed_dim);
        }

        ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* timestep,
                                  ggml_tensor* context,
                                  ggml_tensor* pe) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);  // [N, adaln_embed_dim]

            auto txt = cap_embedder_1->forward(ctx, cap_embedder_0->forward(ctx, context));  // [N, n_txt_token, hidden_size]
            auto img = x_embedder->forward(ctx, x);                                          // [N, n_img_token, hidden_size]
            sd::ggml_graph_cut::mark_graph_cut(txt, "lumina2.prelude", "txt");
            sd::ggml_graph_cut::mark_graph_cut(img, "lumina2.prelude", "img");
            sd::ggml_graph_cut::mark_graph_cut(t_emb, "lumina2.prelude", "t_emb");

            GGML_ASSERT(n_txt_token + n_img_token == pe->ne[3]);

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, n_txt_token);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, n_txt_token, pe->ne[3]);

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);

                txt = block->forward(ctx, txt, txt_pe, nullptr, nullptr);
                sd::ggml_graph_cut::mark_graph_cut(txt, "lumina2.context_refiner." + std::to_string(i), "txt");
            }

            for (int i = 0; i < config.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);

                img = block->forward(ctx, img, img_pe, nullptr, t_emb);
                sd::ggml_graph_cut::mark_graph_cut(img, "lumina2.noise_refiner." + std::to_string(i), "img");
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
            sd::ggml_graph_cut::mark_graph_cut(txt_img, "lumina2.prelude", "txt_img");

            for (int i = 0; i < config.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, pe, nullptr, t_emb);
                sd::ggml_graph_cut::mark_graph_cut(txt_img, "lumina2.layers." + std::to_string(i), "txt_img");
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);  // [N, n_txt_token + n_img_token, ph*pw*C]

            auto out = ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, n_txt_token, n_txt_token + n_img_token);  // [N, n_img_token, ph*pw*C]

            return out;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* timestep,
                             ggml_tensor* context,
                             ggml_tensor* pe) {
            // Forward pass of NextDiT.
            // x: [N, C, H, W]
            // timestep: [N,] (1 - sigma, in [0, 1])
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int patch_size = config.patch_size;

            auto img = DiT::pad_and_patchify(ctx, x, patch_size, patch_size, false);

            auto out = forward_core(ctx, img, timestep, context, pe);

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, patch_size, patch_size, false);  // [N, C, H, W]

            // the model predicts data - noise; the engine expects noise - data
            out = ggml_ext_scale(ctx->ggml_ctx, out, -1.f);

            return out;
        }
    };

    struct Lumina2Runner : public DiffusionModelRunner {
    public:
        Lumina2Config config;
        Lumina2Model lumina2;
        std::vector<float> pe_vec;

        Lumina2Runner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map      = {},
                      const std::string prefix                            = "",
                      SDVersion version                                   = VERSION_LUMINA2,
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(Lumina2Config::detect_from_weights(tensor_storage_map, prefix)) {
            SD_UNUSED(version);
            lumina2 = Lumina2Model(config);
            lumina2.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "lumina2";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            lumina2.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor) {
            ggml_cgraph* gf        = new_graph_custom(LUMINA2_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            GGML_ASSERT(x->ne[3] == 1);
            GGML_ASSERT(!context_tensor.empty());
            ggml_tensor* context = make_input(context_tensor);

            pe_vec      = gen_lumina2_pe(static_cast<int>(x->ne[1]),
                                         static_cast<int>(x->ne[0]),
                                         config.patch_size,
                                         static_cast<int>(x->ne[3]),
                                         static_cast<int>(context->ne[1]),
                                         config.theta,
                                         config.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / config.axes_dim_sum / 2);
            auto pe     = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, pos_len);
            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx = get_context();

            ggml_tensor* out = lumina2.forward(&runner_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               pe);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& x,
                                  const sd::Tensor<float>& timesteps,
                                  const sd::Tensor<float>& context) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, n_token, cap_feat_dim]
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context);
            };

            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            if (diffusion_params.ref_latents && !diffusion_params.ref_latents->empty()) {
                LOG_WARN("lumina2 does not support reference latents; ignoring them");
            }
            return compute(n_threads,
                           *diffusion_params.x,
                           *diffusion_params.timesteps,
                           tensor_or_empty(diffusion_params.context));
        }
    };

}  // namespace Lumina2

#endif  // __SD_MODEL_DIFFUSION_LUMINA2_HPP__
