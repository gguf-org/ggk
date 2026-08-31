#ifndef __SD_MODEL_DIFFUSION_ACE_STEP_HPP__
#define __SD_MODEL_DIFFUSION_ACE_STEP_HPP__

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/diffusion/flux.hpp"
#include "model/diffusion/model.hpp"
#include "model_loader.h"

// Ref: https://github.com/ace-step/ACE-Step (models/ace_step_transformer.py, models/attention.py)
// Ref: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/ace/model.py
//
// ACE-Step v1: a Sana-style DiT over mel-spectrogram DCAE latents [8, 16, T]
// (T is time; 16 mel-latent bins collapse into one token per frame via a
// (16, 1) early-conv patchify). Self attention is LiteLA linear attention
// (ReLU kernel with a ones-padded value row for normalization); cross
// attention is ordinary softmax attention over the conditioning sequence
// [speaker, genre(text), lyrics]. Both use the checkpoint's idiosyncratic
// RoPE: the cos/sin tables use the Qwen2 cat(freqs, freqs) layout while the
// rotation pairs adjacent elements (GPT-J style) - not a pure rotation, but
// it is what the model was trained with, so it is reproduced verbatim.
//
// Conditioning: the speaker embedding is all-zero (the released checkpoint
// has no speaker inputs), the genre embedder projects UMT5-base hidden
// states, and lyrics run through the conformer lyric encoder (voice-BPE
// tokens -> lyric_embs -> 6 rel-pos attention blocks -> lyric_proj) and are
// appended to the conditioning sequence. With no lyrics the reference
// pipeline feeds one token whose mask is 0, which the cross attention then
// masks out entirely, so dropping the token is numerically identical; the
// unconditional branch zeroes both the text hidden states and the lyric
// mask, so it stays [speaker, zeros-genre] only.
namespace AceStep {
    constexpr int ACE_STEP_GRAPH_SIZE = 20480;

    struct AceStepConfig {
        int64_t inner_dim          = 2560;
        int64_t num_layers         = 24;
        int64_t head_dim           = 128;
        int64_t num_heads          = 20;
        int64_t in_channels        = 8;
        int64_t out_channels       = 8;
        int64_t patch_height       = 16;   // latent frequency bins per token
        int64_t mlp_hidden         = 6400; // GLUMBConv hidden (mlp_ratio 2.5)
        int64_t text_embedding_dim = 768;
        int64_t speaker_dim        = 512;
        float rope_theta           = 1000000.0f;
        float norm_eps             = 1e-6f;

        // conformer lyric encoder (espnet rel-pos attention, no conv/macaron)
        bool has_lyric_branch    = false;
        int64_t lyric_vocab_size = 6693;
        int64_t lyric_hidden     = 1024;
        int64_t lyric_heads      = 16;
        int64_t lyric_head_dim   = 64;
        int64_t lyric_ff         = 4096;
        int64_t lyric_layers     = 6;

        static AceStepConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                 const std::string& prefix) {
            AceStepConfig config;
            int64_t detected_layers       = 0;
            int64_t detected_lyric_layers = 0;
            for (const auto& [name, tensor_storage] : tensor_storage_map) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                if (ends_with(name, "t_block.1.weight") && tensor_storage.n_dims == 2) {
                    config.inner_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "genre_embedder.weight") && tensor_storage.n_dims == 2) {
                    config.text_embedding_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "speaker_embedder.weight") && tensor_storage.n_dims == 2) {
                    config.speaker_dim = tensor_storage.ne[0];
                } else if (ends_with(name, "proj_in.early_conv_layers.0.weight") && tensor_storage.n_dims == 4) {
                    config.in_channels  = tensor_storage.ne[2];
                    config.patch_height = tensor_storage.ne[1];
                } else if (ends_with(name, "transformer_blocks.0.ff.inverted_conv.conv.weight")) {
                    config.mlp_hidden = tensor_storage.ne[2] / 2;
                } else if (ends_with(name, "lyric_embs.weight") && tensor_storage.n_dims == 2) {
                    config.has_lyric_branch = true;
                    config.lyric_hidden     = tensor_storage.ne[0];
                    config.lyric_vocab_size = tensor_storage.ne[1];
                } else if (ends_with(name, "lyric_encoder.encoders.0.feed_forward.w_1.weight") && tensor_storage.n_dims == 2) {
                    config.lyric_ff = tensor_storage.ne[1];
                } else if (ends_with(name, "lyric_encoder.encoders.0.self_attn.pos_bias_u") && tensor_storage.n_dims == 2) {
                    config.lyric_head_dim = tensor_storage.ne[0];
                    config.lyric_heads    = tensor_storage.ne[1];
                }
                size_t lyric_pos = name.find("lyric_encoder.encoders.");
                if (lyric_pos != std::string::npos) {
                    auto items = split_string(name.substr(lyric_pos + strlen("lyric_encoder.")), '.');
                    if (items.size() > 1) {
                        detected_lyric_layers = std::max<int64_t>(detected_lyric_layers, atoi(items[1].c_str()) + 1);
                    }
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
            if (detected_layers > 0) {
                config.num_layers = detected_layers;
            }
            if (detected_lyric_layers > 0) {
                config.lyric_layers = detected_lyric_layers;
            } else {
                config.has_lyric_branch = false;
            }
            config.out_channels = config.in_channels;
            config.num_heads    = std::max<int64_t>(1, config.inner_dim / config.head_dim);

            LOG_DEBUG("ace_step: inner_dim=%" PRId64 ", layers=%" PRId64 ", heads=%" PRId64 "x%" PRId64
                      ", channels=%" PRId64 ", patch_height=%" PRId64 ", mlp_hidden=%" PRId64 ", text_dim=%" PRId64,
                      config.inner_dim,
                      config.num_layers,
                      config.num_heads,
                      config.head_dim,
                      config.in_channels,
                      config.patch_height,
                      config.mlp_hidden,
                      config.text_embedding_dim);
            return config;
        }
    };

    // The reference apply_rotary_emb: cos/sin tables are cat(freqs, freqs)
    // over the head dim while the rotated companion pairs adjacent elements
    // ((-x1, x0, -x3, x2, ...)). cos/sin arrive as [head_dim, 1, S] inputs.
    // x: [head_dim, num_heads, S, N]
    __STATIC_INLINE__ ggml_tensor* ace_apply_rope(ggml_context* ctx,
                                                  ggml_tensor* x,
                                                  ggml_tensor* cos_tab,
                                                  ggml_tensor* sin_tab) {
        const int64_t d = x->ne[0];
        const int64_t h = x->ne[1];
        const int64_t s = x->ne[2];
        const int64_t n = x->ne[3];

        auto pairs  = ggml_reshape_4d(ctx, x, 2, d / 2, h, s * n);
        auto x_even = ggml_ext_slice(ctx, pairs, 0, 0, 1);  // [1, d/2, h, s*n]
        auto x_odd  = ggml_ext_slice(ctx, pairs, 0, 1, 2);
        auto rot    = ggml_concat(ctx, ggml_neg(ctx, x_odd), x_even, 0);  // (-x1, x0)
        rot         = ggml_reshape_4d(ctx, rot, d, h, s, n);

        auto out = ggml_add(ctx,
                            ggml_mul(ctx, x, cos_tab),
                            ggml_mul(ctx, rot, sin_tab));
        return out;
    }

    // Rotate a fused [inner_dim, S] projection head-by-head.
    __STATIC_INLINE__ ggml_tensor* ace_apply_rope_fused(ggml_context* ctx,
                                                        ggml_tensor* x,
                                                        ggml_tensor* cos_tab,
                                                        ggml_tensor* sin_tab,
                                                        int64_t head_dim) {
        const int64_t inner = x->ne[0];
        const int64_t s     = x->ne[1];
        auto split          = ggml_reshape_3d(ctx, x, head_dim, inner / head_dim, s);
        split               = ace_apply_rope(ctx, split, cos_tab, sin_tab);
        return ggml_reshape_2d(ctx, split, inner, s);
    }

    // diffusers-style Attention module holding to_q/to_k/to_v/to_out.0; the
    // checkpoint's unused add_{q,k,v}_proj / to_add_out tensors are skipped.
    struct AceAttention : public GGMLBlock {
        int64_t inner_dim;

        AceAttention(int64_t query_dim, int64_t context_dim, int64_t inner_dim)
            : inner_dim(inner_dim) {
            blocks["to_q"]     = std::make_shared<Linear>(query_dim, inner_dim, true);
            blocks["to_k"]     = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["to_v"]     = std::make_shared<Linear>(context_dim, inner_dim, true);
            blocks["to_out.0"] = std::make_shared<Linear>(inner_dim, query_dim, true);
        }
    };

    // LiteLA: out = (V_pad K^T) q / last_row, everything in f32.
    struct LinearSelfAttention : public AceAttention {
        int64_t num_heads;
        int64_t head_dim;

        LinearSelfAttention(int64_t dim, int64_t num_heads, int64_t head_dim)
            : AceAttention(dim, dim, num_heads * head_dim),
              num_heads(num_heads),
              head_dim(head_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,  // [dim, S, 1]
                             ggml_tensor* cos_tab,
                             ggml_tensor* sin_tab) {
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            const int64_t S = x->ne[1];
            auto gctx       = ctx->ggml_ctx;

            auto q = to_q->forward(ctx, x);  // [inner, S]
            auto k = to_k->forward(ctx, x);
            auto v = to_v->forward(ctx, x);

            q = ggml_reshape_3d(gctx, q, head_dim, num_heads, S);  // [d, h, S]
            k = ggml_reshape_3d(gctx, k, head_dim, num_heads, S);
            v = ggml_reshape_3d(gctx, v, head_dim, num_heads, S);

            q = ace_apply_rope(gctx, q, cos_tab, sin_tab);
            k = ace_apply_rope(gctx, k, cos_tab, sin_tab);

            q = ggml_relu(gctx, q);
            k = ggml_relu(gctx, k);

            // q: [d, S, h], k/v: [S, d, h]
            q = ggml_ext_cont(gctx, ggml_permute(gctx, q, 0, 2, 1, 3));
            k = ggml_ext_cont(gctx, ggml_permute(gctx, k, 1, 2, 0, 3));
            v = ggml_ext_cont(gctx, ggml_permute(gctx, v, 1, 2, 0, 3));

            auto ones  = ggml_ext_ones(gctx, S, 1, num_heads, 1);
            auto v_pad = ggml_concat(gctx, v, ones, 1);  // [S, d+1, h]

            auto vk = ggml_mul_mat(gctx, k, v_pad);  // [d, d+1, h]
            auto hs = ggml_mul_mat(gctx, vk, q);     // [d+1, S, h]

            auto num = ggml_ext_slice(gctx, hs, 0, 0, head_dim);            // [d, S, h]
            auto den = ggml_ext_slice(gctx, hs, 0, head_dim, head_dim + 1); // [1, S, h]
            auto eps = ggml_ext_scale(gctx, ggml_ext_ones(gctx, 1, 1, 1, 1), 1e-15f);
            den      = ggml_add(gctx, den, eps);
            auto out = ggml_div(gctx, num, den);  // [d, S, h]

            out = ggml_ext_cont(gctx, ggml_permute(gctx, out, 0, 2, 1, 3));  // [d, h, S]
            out = ggml_reshape_2d(gctx, out, inner_dim, S);

            return to_out->forward(ctx, out);
        }
    };

    struct CrossAttention : public AceAttention {
        int64_t num_heads;
        int64_t head_dim;

        CrossAttention(int64_t dim, int64_t num_heads, int64_t head_dim)
            : AceAttention(dim, dim, num_heads * head_dim),
              num_heads(num_heads),
              head_dim(head_dim) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,        // [dim, S, 1]
                             ggml_tensor* context,  // [dim, L, 1]
                             ggml_tensor* cos_self,
                             ggml_tensor* sin_self,
                             ggml_tensor* cos_cross,
                             ggml_tensor* sin_cross) {
            auto to_q   = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k   = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v   = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            auto gctx   = ctx->ggml_ctx;

            auto q = to_q->forward(ctx, x);        // [inner, S]
            auto k = to_k->forward(ctx, context);  // [inner, L]
            auto v = to_v->forward(ctx, context);

            q = ace_apply_rope_fused(gctx, q, cos_self, sin_self, head_dim);
            k = ace_apply_rope_fused(gctx, k, cos_cross, sin_cross, head_dim);

            auto out = ggml_ext_attention_ext(gctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              num_heads,
                                              nullptr,
                                              false,
                                              ctx->flash_attn_enabled);
            return to_out->forward(ctx, out);
        }
    };

    // GLUMBConv over the token sequence: 1x1 expand, depthwise conv along
    // time, SiLU-gated GLU, 1x1 project. The 1x1 convs are stored as conv
    // weights but applied as linears on the channel-major token layout.
    struct GLUMBConv : public GGMLBlock {
        int64_t in_dim;
        int64_t hidden;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["inverted_conv.conv.weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, in_dim, hidden * 2, 1);
            params["inverted_conv.conv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden * 2);
            params["depth_conv.conv.weight"]    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 1, hidden * 2, 1);
            params["depth_conv.conv.bias"]      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden * 2);
            params["point_conv.conv.weight"]    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, hidden, in_dim, 1);
        }

        GLUMBConv(int64_t in_dim, int64_t hidden)
            : in_dim(in_dim), hidden(hidden) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gctx       = ctx->ggml_ctx;
            const int64_t S = x->ne[1];

            auto w_inv = ggml_reshape_2d(gctx, params["inverted_conv.conv.weight"], in_dim, hidden * 2);
            x          = ggml_ext_linear(gctx, x, w_inv, params["inverted_conv.conv.bias"]);  // [2*hidden, S]
            x          = ggml_silu(gctx, x);  // inverted_conv act

            // depthwise conv along time: [S, 2*hidden]
            x           = ggml_ext_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));
            auto w_dw   = ggml_reshape_3d(gctx, params["depth_conv.conv.weight"], 3, 1, hidden * 2);
            x           = ggml_conv_1d_dw(gctx, w_dw, ggml_reshape_3d(gctx, x, S, hidden * 2, 1), 1, 1, 1);
            auto b_dw   = ggml_reshape_2d(gctx, params["depth_conv.conv.bias"], 1, hidden * 2);
            x           = ggml_add(gctx, ggml_reshape_2d(gctx, x, S, hidden * 2), b_dw);

            auto halves = ggml_ext_chunk(gctx, x, 2, 1);
            auto gate   = ggml_silu(gctx, halves[1]);
            x           = ggml_mul(gctx, halves[0], gate);  // [S, hidden]

            x           = ggml_ext_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [hidden, S]
            auto w_pt   = ggml_reshape_2d(gctx, params["point_conv.conv.weight"], hidden, in_dim);
            x           = ggml_ext_linear(gctx, x, w_pt, nullptr);
            return x;
        }
    };

    struct LinearTransformerBlock : public GGMLBlock {
        int64_t dim;
        float norm_eps;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 6);
        }

        LinearTransformerBlock(const AceStepConfig& config)
            : dim(config.inner_dim), norm_eps(config.norm_eps) {
            blocks["attn"]       = std::make_shared<LinearSelfAttention>(config.inner_dim, config.num_heads, config.head_dim);
            blocks["cross_attn"] = std::make_shared<CrossAttention>(config.inner_dim, config.num_heads, config.head_dim);
            blocks["ff"]         = std::make_shared<GLUMBConv>(config.inner_dim, config.mlp_hidden);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* context,
                             ggml_tensor* temb,  // [dim*6, 1]
                             ggml_tensor* cos_self,
                             ggml_tensor* sin_self,
                             ggml_tensor* cos_cross,
                             ggml_tensor* sin_cross) {
            auto attn       = std::dynamic_pointer_cast<LinearSelfAttention>(blocks["attn"]);
            auto cross_attn = std::dynamic_pointer_cast<CrossAttention>(blocks["cross_attn"]);
            auto ff         = std::dynamic_pointer_cast<GLUMBConv>(blocks["ff"]);
            auto gctx       = ctx->ggml_ctx;

            auto table  = ggml_reshape_3d(gctx, params["scale_shift_table"], dim, 6, 1);
            auto t      = ggml_reshape_3d(gctx, temb, dim, 6, 1);
            auto mods   = ggml_add(gctx, table, t);
            auto chunks = ggml_ext_chunk(gctx, mods, 6, 1);  // each [dim, 1, 1]

            // 1. LiteLA self attention (RMS-norm + adaLN modulation)
            auto h = ggml_rms_norm(gctx, x, norm_eps);
            h      = Flux::modulate(gctx, h, chunks[0], chunks[1], true);
            h      = attn->forward(ctx, h, cos_self, sin_self);
            x      = ggml_add(gctx, x, ggml_mul(gctx, h, chunks[2]));

            // 2. cross attention on the unnormalized residual stream
            h = cross_attn->forward(ctx, x, context, cos_self, sin_self, cos_cross, sin_cross);
            x = ggml_add(gctx, x, h);

            // 3. GLUMBConv feed forward
            h = ggml_rms_norm(gctx, x, norm_eps);
            h = Flux::modulate(gctx, h, chunks[3], chunks[4], true);
            h = ff->forward(ctx, h);
            x = ggml_add(gctx, x, ggml_mul(gctx, h, chunks[5]));

            return x;
        }
    };

    // Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0) + TimestepEmbedding
    struct TimestepEmbedder : public GGMLBlock {
        TimestepEmbedder(int64_t hidden_size) {
            blocks["linear_1"] = std::make_shared<Linear>(256, hidden_size);
            blocks["linear_2"] = std::make_shared<Linear>(hidden_size, hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* timesteps) {
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            auto t = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256);
            t      = linear_1->forward(ctx, t);
            t      = ggml_silu(ctx->ggml_ctx, t);
            t      = linear_2->forward(ctx, t);
            return t;  // [hidden, N]
        }
    };

    struct AceStepTransformer : public GGMLBlock {
        AceStepConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["final_layer.scale_shift_table"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.inner_dim, 2);
            if (config.has_lyric_branch) {
                for (int64_t i = 0; i < config.lyric_layers; ++i) {
                    const std::string p = "lyric_encoder.encoders." + std::to_string(i) + ".self_attn.";
                    params[p + "pos_bias_u"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.lyric_head_dim, config.lyric_heads);
                    params[p + "pos_bias_v"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.lyric_head_dim, config.lyric_heads);
                }
            }
        }

        AceStepTransformer(const AceStepConfig& config)
            : config(config) {
            blocks["proj_in.early_conv_layers.0"] = std::make_shared<Conv2d>(config.in_channels,
                                                                             config.in_channels * 256,
                                                                             std::pair<int, int>{(int)config.patch_height, 1},
                                                                             std::pair<int, int>{(int)config.patch_height, 1});
            blocks["proj_in.early_conv_layers.1"] = std::make_shared<GroupNorm>(32, config.in_channels * 256, 1e-6f);
            blocks["proj_in.early_conv_layers.2"] = std::make_shared<Conv2d>(config.in_channels * 256,
                                                                             config.inner_dim,
                                                                             std::pair<int, int>{1, 1});

            blocks["timestep_embedder"] = std::make_shared<TimestepEmbedder>(config.inner_dim);
            blocks["t_block.1"]         = std::make_shared<Linear>(config.inner_dim, 6 * config.inner_dim);

            blocks["speaker_embedder"] = std::make_shared<Linear>(config.speaker_dim, config.inner_dim);
            blocks["genre_embedder"]   = std::make_shared<Linear>(config.text_embedding_dim, config.inner_dim);

            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<LinearTransformerBlock>(config);
            }

            blocks["final_layer.linear"] = std::make_shared<Linear>(config.inner_dim,
                                                                    config.patch_height * config.out_channels);

            if (config.has_lyric_branch) {
                blocks["lyric_embs"]                = std::make_shared<Embedding>(config.lyric_vocab_size, config.lyric_hidden);
                blocks["lyric_encoder.embed.out.0"] = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden);
                blocks["lyric_encoder.embed.out.1"] = std::make_shared<LayerNorm>(config.lyric_hidden);
                for (int64_t i = 0; i < config.lyric_layers; ++i) {
                    const std::string p = "lyric_encoder.encoders." + std::to_string(i) + ".";
                    blocks[p + "self_attn.linear_q"]   = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden);
                    blocks[p + "self_attn.linear_k"]   = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden);
                    blocks[p + "self_attn.linear_v"]   = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden);
                    blocks[p + "self_attn.linear_out"] = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden);
                    blocks[p + "self_attn.linear_pos"] = std::make_shared<Linear>(config.lyric_hidden, config.lyric_hidden, false);
                    blocks[p + "norm_mha"]             = std::make_shared<LayerNorm>(config.lyric_hidden);
                    blocks[p + "norm_ff"]              = std::make_shared<LayerNorm>(config.lyric_hidden);
                    blocks[p + "feed_forward.w_1"]     = std::make_shared<Linear>(config.lyric_hidden, config.lyric_ff);
                    blocks[p + "feed_forward.w_2"]     = std::make_shared<Linear>(config.lyric_ff, config.lyric_hidden);
                }
                blocks["lyric_encoder.after_norm"] = std::make_shared<LayerNorm>(config.lyric_hidden);
                blocks["lyric_proj"]               = std::make_shared<Linear>(config.lyric_hidden, config.inner_dim);
            }
        }

        // The conformer lyric encoder (wenet/espnet lineage): LinearEmbed
        // (Linear + LayerNorm, then x*sqrt(d) with an espnet rel-pos table of
        // 2L-1 positions), 6 pre-norm blocks of rel-pos attention + swish FF
        // (no conv module, no macaron), after_norm, and lyric_proj into the
        // DiT conditioning width. Full bidirectional attention, batch of one,
        // no padding - so no mask anywhere.
        //
        // ids: I32 [L]; pos_emb: [lyric_hidden, 2L-1] espnet table where row k
        // encodes relative position (L-1-k). Returns [inner_dim, L].
        ggml_tensor* forward_lyrics(GGMLRunnerContext* ctx,
                                    ggml_tensor* ids,
                                    ggml_tensor* pos_emb) {
            auto gctx       = ctx->ggml_ctx;
            const int64_t L = ids->ne[0];
            const int64_t H = config.lyric_heads;
            const int64_t D = config.lyric_head_dim;

            auto embs = std::dynamic_pointer_cast<Embedding>(blocks["lyric_embs"]);
            auto emb0 = std::dynamic_pointer_cast<Linear>(blocks["lyric_encoder.embed.out.0"]);
            auto emb1 = std::dynamic_pointer_cast<LayerNorm>(blocks["lyric_encoder.embed.out.1"]);

            auto x = embs->forward(ctx, ids);  // [lyric_hidden, L, 1]
            x      = ggml_reshape_2d(gctx, x, config.lyric_hidden, L);
            x      = emb1->forward(ctx, emb0->forward(ctx, x));
            x      = ggml_scale(gctx, x, std::sqrt((float)config.lyric_hidden));  // espnet xscale

            // [lyric_hidden, S] -> [D, S, H]
            auto split_heads = [&](ggml_tensor* t) {
                t = ggml_reshape_3d(gctx, t, D, H, t->ne[1]);
                return ggml_ext_cont(gctx, ggml_permute(gctx, t, 0, 2, 1, 3));
            };

            for (int64_t i = 0; i < config.lyric_layers; ++i) {
                const std::string p = "lyric_encoder.encoders." + std::to_string(i) + ".";
                auto linear_q       = std::dynamic_pointer_cast<Linear>(blocks[p + "self_attn.linear_q"]);
                auto linear_k       = std::dynamic_pointer_cast<Linear>(blocks[p + "self_attn.linear_k"]);
                auto linear_v       = std::dynamic_pointer_cast<Linear>(blocks[p + "self_attn.linear_v"]);
                auto linear_out     = std::dynamic_pointer_cast<Linear>(blocks[p + "self_attn.linear_out"]);
                auto linear_pos     = std::dynamic_pointer_cast<Linear>(blocks[p + "self_attn.linear_pos"]);
                auto norm_mha       = std::dynamic_pointer_cast<LayerNorm>(blocks[p + "norm_mha"]);
                auto norm_ff        = std::dynamic_pointer_cast<LayerNorm>(blocks[p + "norm_ff"]);
                auto ff_w1          = std::dynamic_pointer_cast<Linear>(blocks[p + "feed_forward.w_1"]);
                auto ff_w2          = std::dynamic_pointer_cast<Linear>(blocks[p + "feed_forward.w_2"]);

                auto xn = norm_mha->forward(ctx, x);
                auto qh = split_heads(linear_q->forward(ctx, xn));       // [D, L, H]
                auto kh = split_heads(linear_k->forward(ctx, xn));       // [D, L, H]
                auto vh = split_heads(linear_v->forward(ctx, xn));       // [D, L, H]
                auto ph = split_heads(linear_pos->forward(ctx, pos_emb));  // [D, 2L-1, H]

                // per-head biases broadcast over positions
                auto bias_u = ggml_reshape_3d(gctx, params[p + "self_attn.pos_bias_u"], D, 1, H);
                auto bias_v = ggml_reshape_3d(gctx, params[p + "self_attn.pos_bias_v"], D, 1, H);
                auto q_u    = ggml_add(gctx, qh, bias_u);
                auto q_v    = ggml_add(gctx, qh, bias_v);

                auto ac      = ggml_mul_mat(gctx, kh, q_u);  // [L(key), L(query), H]
                auto bd_full = ggml_ext_cont(gctx, ggml_mul_mat(gctx, ph, q_v));  // [2L-1, L, H]

                // rel_shift as a strided view: bd[j, i, h] = bd_full[L-1-i+j, i, h]
                ggml_tensor* bd = bd_full;
                if (L > 1) {
                    const size_t es = ggml_element_size(bd_full);
                    bd              = ggml_view_3d(gctx, bd_full, L, L, H,
                                                   (size_t)(2 * L - 2) * es,
                                                   bd_full->nb[2],
                                                   (size_t)(L - 1) * es);
                    bd              = ggml_ext_cont(gctx, bd);
                }

                auto scores = ggml_scale(gctx, ggml_add(gctx, ac, bd), 1.0f / std::sqrt((float)D));
                auto attn   = ggml_soft_max(gctx, scores);  // over keys (ne0)

                auto vt  = ggml_ext_cont(gctx, ggml_permute(gctx, vh, 1, 0, 2, 3));  // [L, D, H]
                auto out = ggml_mul_mat(gctx, vt, attn);                             // [D, L, H]
                out      = ggml_ext_cont(gctx, ggml_permute(gctx, out, 0, 2, 1, 3));  // [D, H, L]
                out      = ggml_reshape_2d(gctx, out, H * D, L);
                out      = linear_out->forward(ctx, out);
                x        = ggml_add(gctx, x, out);

                auto xf = norm_ff->forward(ctx, x);
                xf      = ff_w2->forward(ctx, ggml_silu(gctx, ff_w1->forward(ctx, xf)));
                x       = ggml_add(gctx, x, xf);
            }

            auto after_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["lyric_encoder.after_norm"]);
            auto lyric_proj = std::dynamic_pointer_cast<Linear>(blocks["lyric_proj"]);
            x               = after_norm->forward(ctx, x);
            return lyric_proj->forward(ctx, x);  // [inner_dim, L]
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,          // [T, 16, 8, 1]
                             ggml_tensor* timesteps,  // [1]
                             ggml_tensor* context,    // [768, L, 1]
                             ggml_tensor* speaker,    // [512, 1]
                             ggml_tensor* lyric,      // [inner, L_lyr] pre-encoded lyric hidden states, or nullptr
                             ggml_tensor* cos_self,   // [head_dim, 1, T]
                             ggml_tensor* sin_self,
                             ggml_tensor* cos_cross,  // [head_dim, 1, L+1(+L_lyr)]
                             ggml_tensor* sin_cross) {
            auto conv0    = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in.early_conv_layers.0"]);
            auto norm0    = std::dynamic_pointer_cast<GroupNorm>(blocks["proj_in.early_conv_layers.1"]);
            auto conv2    = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in.early_conv_layers.2"]);
            auto t_embed  = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["timestep_embedder"]);
            auto t_block  = std::dynamic_pointer_cast<Linear>(blocks["t_block.1"]);
            auto spk_emb  = std::dynamic_pointer_cast<Linear>(blocks["speaker_embedder"]);
            auto gen_emb  = std::dynamic_pointer_cast<Linear>(blocks["genre_embedder"]);
            auto fl_lin   = std::dynamic_pointer_cast<Linear>(blocks["final_layer.linear"]);
            auto gctx     = ctx->ggml_ctx;

            const int64_t T = x->ne[0];

            // patchify: [T, 16, 8] -> [inner_dim, T]
            auto h = conv0->forward(ctx, x);   // [T, 1, 2048, 1]
            h      = norm0->forward(ctx, h);
            h      = conv2->forward(ctx, h);   // [T, 1, inner, 1]
            h      = ggml_reshape_2d(gctx, h, T, config.inner_dim);
            h      = ggml_ext_cont(gctx, ggml_permute(gctx, h, 1, 0, 2, 3));  // [inner, T]
            h      = ggml_reshape_3d(gctx, h, config.inner_dim, T, 1);

            // conditioning sequence: [speaker, genre(text), lyric?]
            auto spk = spk_emb->forward(ctx, speaker);                       // [inner, 1]
            auto txt = gen_emb->forward(ctx, context);                       // [inner, L]
            auto enc = ggml_concat(gctx,
                                   ggml_reshape_3d(gctx, spk, config.inner_dim, 1, 1),
                                   ggml_reshape_3d(gctx, txt, config.inner_dim, txt->ne[1], 1),
                                   1);  // [inner, L+1, 1]
            if (lyric != nullptr) {
                enc = ggml_concat(gctx,
                                  enc,
                                  ggml_reshape_3d(gctx, lyric, config.inner_dim, lyric->ne[1], 1),
                                  1);  // [inner, L+1+L_lyr, 1]
            }

            auto embedded_timestep = t_embed->forward(ctx, timesteps);  // [inner, 1]
            auto temb              = t_block->forward(ctx, ggml_silu(gctx, embedded_timestep));  // [6*inner, 1]

            for (int64_t i = 0; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<LinearTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);
                h          = block->forward(ctx, h, enc, temb, cos_self, sin_self, cos_cross, sin_cross);
                sd::ggml_graph_cut::mark_graph_cut(h, "ace_step.transformer_blocks." + std::to_string(i), "x");
            }

            // final layer: RMS norm (affine-free) + adaLN(shift, scale) + linear
            auto table  = ggml_reshape_3d(gctx, params["final_layer.scale_shift_table"], config.inner_dim, 2, 1);
            auto t2     = ggml_reshape_3d(gctx, embedded_timestep, config.inner_dim, 1, 1);
            auto shape  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, config.inner_dim, 2, 1);
            auto mods   = ggml_add(gctx,
                                 ggml_repeat(gctx, table, shape),
                                 ggml_repeat(gctx, t2, shape));
            auto chunks = ggml_ext_chunk(gctx, mods, 2, 1);

            h = ggml_rms_norm(gctx, h, config.norm_eps);
            h = Flux::modulate(gctx, h, chunks[0], chunks[1], true);
            h = fl_lin->forward(ctx, h);  // [16*8, T]

            // unpatchify: feature f = p * out_channels + c -> [T, 16, 8]
            h = ggml_reshape_4d(gctx, h, config.out_channels, config.patch_height, T, 1);
            h = ggml_ext_cont(gctx, ggml_permute(gctx, h, 2, 1, 0, 3));  // [T, 16, 8, 1]
            return h;
        }
    };

    struct AceStepRunner : public DiffusionModelRunner {
        AceStepConfig config;
        AceStepTransformer model;

        AceStepRunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map      = {},
                      const std::string prefix                            = "",
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(AceStepConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, this->prefix);
        }

        std::string get_desc() override {
            return "ACE-Step";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        // cos/sin in the reference layout: cat(freqs, freqs) per position.
        static void build_rope_table(int64_t length,
                                     int64_t head_dim,
                                     float theta,
                                     std::vector<float>* cos_out,
                                     std::vector<float>* sin_out) {
            const int64_t half = head_dim / 2;
            cos_out->resize(length * head_dim);
            sin_out->resize(length * head_dim);
            for (int64_t pos = 0; pos < length; ++pos) {
                for (int64_t j = 0; j < head_dim; ++j) {
                    const int64_t idx = j < half ? j : j - half;
                    const double freq = 1.0 / std::pow((double)theta, (double)(2 * idx) / (double)head_dim);
                    const double a    = (double)pos * freq;
                    (*cos_out)[pos * head_dim + j] = (float)std::cos(a);
                    (*sin_out)[pos * head_dim + j] = (float)std::sin(a);
                }
            }
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor,
                                 const sd::Tensor<float>& timesteps_tensor,
                                 const sd::Tensor<float>& context_tensor,
                                 const sd::Tensor<float>& speaker_tensor,
                                 const sd::Tensor<float>& lyric_tensor,
                                 const sd::Tensor<float>& cos_self_tensor,
                                 const sd::Tensor<float>& sin_self_tensor,
                                 const sd::Tensor<float>& cos_cross_tensor,
                                 const sd::Tensor<float>& sin_cross_tensor) {
            ggml_cgraph* gf        = new_graph_custom(ACE_STEP_GRAPH_SIZE);
            ggml_tensor* x         = make_input(x_tensor);
            ggml_tensor* timesteps = make_input(timesteps_tensor);
            ggml_tensor* context   = make_input(context_tensor);
            ggml_tensor* speaker   = make_input(speaker_tensor);
            ggml_tensor* lyric     = lyric_tensor.empty() ? nullptr : make_input(lyric_tensor);
            ggml_tensor* cos_self  = make_input(cos_self_tensor);
            ggml_tensor* sin_self  = make_input(sin_self_tensor);
            ggml_tensor* cos_cross = make_input(cos_cross_tensor);
            ggml_tensor* sin_cross = make_input(sin_cross_tensor);

            auto as_rope_input = [&](ggml_tensor* t) {
                return ggml_reshape_3d(compute_ctx, t, config.head_dim, 1, t->ne[1]);
            };
            cos_self  = as_rope_input(cos_self);
            sin_self  = as_rope_input(sin_self);
            cos_cross = as_rope_input(cos_cross);
            sin_cross = as_rope_input(sin_cross);

            auto runner_ctx = get_context();
            auto out        = model.forward(&runner_ctx,
                                     x,
                                     timesteps,
                                     context,
                                     speaker,
                                     lyric,
                                     cos_self,
                                     sin_self,
                                     cos_cross,
                                     sin_cross);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // Encode tokenized lyrics through the conformer branch once per
        // generation; the result rides into build_graph as the tail of the
        // cross-attention conditioning sequence (conditional branch only -
        // the reference zeroes the lyric mask on the unconditional branch,
        // which is equivalent to dropping the tokens).
        sd::Tensor<float> encode_lyrics(int n_threads, const std::vector<int32_t>& ids) {
            GGML_ASSERT(config.has_lyric_branch);
            GGML_ASSERT(!ids.empty());
            const int64_t L = (int64_t)ids.size();
            const int64_t P = 2 * L - 1;

            // espnet rel-pos table: row k encodes relative position L-1-k
            std::vector<float> pe((size_t)(P * config.lyric_hidden));
            for (int64_t k = 0; k < P; ++k) {
                const double rel = (double)(L - 1 - k);
                for (int64_t i = 0; i * 2 < config.lyric_hidden; ++i) {
                    const double freq = std::exp(-(double)(2 * i) * std::log(10000.0) / (double)config.lyric_hidden);
                    pe[(size_t)(k * config.lyric_hidden + 2 * i)]     = (float)std::sin(rel * freq);
                    pe[(size_t)(k * config.lyric_hidden + 2 * i + 1)] = (float)std::cos(rel * freq);
                }
            }

            sd::Tensor<int32_t> ids_tensor({L}, ids);
            sd::Tensor<float> pos_tensor({config.lyric_hidden, P}, pe);

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf     = new_graph_custom(ACE_STEP_GRAPH_SIZE);
                ggml_tensor* ids_in = make_input(ids_tensor);
                ggml_tensor* pos_in = make_input(pos_tensor);
                auto runner_ctx     = get_context();
                ggml_build_forward_expand(gf, model.forward_lyrics(&runner_ctx, ids_in, pos_in));
                return gf;
            };
            auto out = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            return out.has_value() ? std::move(*out) : sd::Tensor<float>();
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& diffusion_params) override {
            GGML_ASSERT(diffusion_params.x != nullptr);
            GGML_ASSERT(diffusion_params.timesteps != nullptr);
            GGML_ASSERT(diffusion_params.context != nullptr);

            const sd::Tensor<float>& x         = *diffusion_params.x;
            const sd::Tensor<float>& timesteps = *diffusion_params.timesteps;
            const sd::Tensor<float>& context   = *diffusion_params.context;
            // pre-encoded lyric hidden states ([inner, L_lyr], conditional branch only)
            const sd::Tensor<float>& lyric     = tensor_or_empty(diffusion_params.y);

            const int64_t T     = x.shape()[0];
            const int64_t L_enc = context.shape()[1] + 1 + (lyric.empty() ? 0 : lyric.shape()[1]);

            std::vector<float> cos_self_vec, sin_self_vec, cos_cross_vec, sin_cross_vec;
            build_rope_table(T, config.head_dim, config.rope_theta, &cos_self_vec, &sin_self_vec);
            build_rope_table(L_enc, config.head_dim, config.rope_theta, &cos_cross_vec, &sin_cross_vec);

            sd::Tensor<float> cos_self({config.head_dim, T}, cos_self_vec);
            sd::Tensor<float> sin_self({config.head_dim, T}, sin_self_vec);
            sd::Tensor<float> cos_cross({config.head_dim, L_enc}, cos_cross_vec);
            sd::Tensor<float> sin_cross({config.head_dim, L_enc}, sin_cross_vec);
            sd::Tensor<float> speaker = sd::Tensor<float>::zeros({config.speaker_dim, 1});

            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(x, timesteps, context, speaker, lyric,
                                   cos_self, sin_self, cos_cross, sin_cross);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), x.dim());
        }
    };
}  // namespace AceStep

#endif  // __SD_MODEL_DIFFUSION_ACE_STEP_HPP__
