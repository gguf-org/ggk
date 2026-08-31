#ifndef __SD_MODEL_VAE_ACE_VAE_HPP__
#define __SD_MODEL_VAE_ACE_VAE_HPP__

#include <cmath>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/vae/audio_vae.hpp"
#include "model/vae/ltx_audio_vae.hpp"
#include "model_loader.h"
#include "model_manager.h"

// Ref: https://github.com/ace-step/ACE-Step (music_dcae/, music_vocoder.py)
// Ref: https://github.com/comfyanonymous/ComfyUI/tree/master/comfy/ldm/ace/vae
//
// ACE-Step's music VAE: a SANA DCAE over stereo log-mel spectrograms plus an
// ADaMoS HiFi-GAN vocoder (ConvNeXt backbone + weight-normed HiFi-GAN head).
// Decode-only here: DiT latents [8, 16, T] -> mel [2, 128, 8T] -> 44.1 kHz
// stereo waveform (each stereo channel runs through the vocoder as a batch
// element). Weight-norm stays in the checkpoint's weight_g/weight_v form and
// is folded inside the graph.
namespace AceStep {

    constexpr float ACE_LATENT_SCALE  = 0.1786f;
    constexpr float ACE_LATENT_SHIFT  = -1.9091f;
    constexpr float ACE_MIN_MEL_VALUE = -11.0f;
    constexpr float ACE_MAX_MEL_VALUE = 3.0f;

    struct AceVAEConfig {
        int64_t latent_channels = 8;
        int64_t latent_bins     = 16;
        int64_t mel_bins        = 128;
        int64_t audio_channels  = 2;
        int sample_rate         = 44100;
        int hop_length          = 512;

        // DCAE decoder
        std::vector<int64_t> block_out_channels = {128, 256, 512, 1024};
        int layers_per_block                    = 3;
        int64_t attention_head_dim              = 32;

        // vocoder backbone (ConvNeXt)
        std::vector<int64_t> backbone_dims = {128, 256, 384, 512};
        std::vector<int> backbone_depths   = {3, 3, 9, 3};
        int64_t backbone_mlp               = 3;  // hidden = mlp * dim, detected

        // vocoder head (HiFi-GAN)
        int64_t head_initial_channels        = 1024;
        std::vector<int> upsample_rates      = {4, 4, 2, 2, 2, 2, 2};
        std::vector<int> upsample_kernels    = {8, 8, 4, 4, 4, 4, 4};
        std::vector<int> resblock_kernels    = {3, 7, 11, 13};
        std::vector<std::vector<int>> resblock_dilations = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
        int pre_conv_kernel  = 13;
        int post_conv_kernel = 13;

        static AceVAEConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                const std::string& prefix) {
            AceVAEConfig config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = tensor_storage_map.find(prefix + "." + suffix);
                return it != tensor_storage_map.end() ? &it->second : nullptr;
            };

            if (const TensorStorage* conv_in = find("dcae.decoder.conv_in.weight")) {
                config.latent_channels = conv_in->ne[2];
                config.block_out_channels.back() = conv_in->ne[3];
            }
            if (const TensorStorage* conv_out = find("dcae.decoder.conv_out.weight")) {
                config.audio_channels = conv_out->ne[3];
                config.block_out_channels.front() = conv_out->ne[2];
            }
            for (int i = 0; i < 4; ++i) {
                // stem (i==0) is a k7 conv, the mid layers are 1x1 convs; both
                // are Conv1d weights [k, in, out] -> out channels in ne[2]
                std::string name = "vocoder.backbone.channel_layers." + std::to_string(i) + "." + (i == 0 ? "0" : "1") + ".weight";
                if (const TensorStorage* w = find(name)) {
                    config.backbone_dims[i] = w->ne[2];
                }
            }
            if (const TensorStorage* pw = find("vocoder.backbone.stages.0.0.pwconv1.weight")) {
                config.backbone_mlp = pw->ne[1] / std::max<int64_t>(1, pw->ne[0]);
            }
            if (const TensorStorage* pre = find("vocoder.head.conv_pre.weight_v")) {
                config.head_initial_channels = pre->ne[2];
                config.pre_conv_kernel       = (int)pre->ne[0];
            }
            if (const TensorStorage* post = find("vocoder.head.conv_post.weight_v")) {
                config.post_conv_kernel = (int)post->ne[0];
            }
            config.upsample_kernels.clear();
            for (int i = 0;; ++i) {
                const TensorStorage* up = find("vocoder.head.ups." + std::to_string(i) + ".weight_v");
                if (up == nullptr) {
                    break;
                }
                config.upsample_kernels.push_back((int)up->ne[0]);
            }
            if (config.upsample_kernels.size() != config.upsample_rates.size()) {
                // rates are kernel/2 for this family (8->4, 4->2)
                config.upsample_rates.clear();
                for (int k : config.upsample_kernels) {
                    config.upsample_rates.push_back(k / 2);
                }
            }
            config.resblock_kernels.clear();
            for (int i = 0; i < 4; ++i) {
                const TensorStorage* rb = find("vocoder.head.resblocks." + std::to_string(i) + ".convs1.0.weight_v");
                if (rb == nullptr) {
                    break;
                }
                config.resblock_kernels.push_back((int)rb->ne[0]);
            }
            if (config.resblock_kernels.empty()) {
                config.resblock_kernels = {3, 7, 11, 13};
            }
            config.resblock_dilations.assign(config.resblock_kernels.size(), {1, 3, 5});
            return config;
        }
    };

    // channels-last RMSNorm (weight + bias over the channel dim of [W, H, C, N])
    struct RMSNorm2d : public UnaryBlock {
        int64_t channels;
        float eps;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        }

        RMSNorm2d(int64_t channels, float eps = 1e-5f)
            : channels(channels), eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto h    = ggml_ext_cont(gctx, ggml_ext_torch_permute(gctx, x, 2, 0, 1, 3));  // [C, W, H, N] in torch view
            h         = ggml_rms_norm(gctx, h, eps);
            h         = ggml_mul(gctx, h, params["weight"]);
            h         = ggml_add(gctx, h, params["bias"]);
            h         = ggml_ext_cont(gctx, ggml_ext_torch_permute(gctx, h, 1, 2, 0, 3));
            return h;
        }
    };

    // out(2x+dx, 2y+dy, c) = in(x, y, 4c + 2dy + dx)
    __STATIC_INLINE__ ggml_tensor* ace_pixel_shuffle_2x(ggml_context* ctx, ggml_tensor* x) {
        const int64_t W = x->ne[0];
        const int64_t H = x->ne[1];
        const int64_t C = x->ne[2] / 4;
        GGML_ASSERT(x->ne[3] == 1);

        // split channel 4C into (dx=2, dy*C=2C)
        x = ggml_reshape_4d(ctx, x, W, H, 2, 2 * C);
        // interleave dx into W
        x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [2(dx), W, H, 2C]
        x = ggml_reshape_4d(ctx, x, 2 * W, H, 2, C);               // split 2C into (dy=2, C)
        // interleave dy into H
        x = ggml_ext_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [2W, 2(dy), H, C]
        x = ggml_reshape_4d(ctx, x, 2 * W, 2 * H, C, 1);
        return x;
    }

    // channel c' of the result reads channel c'/repeats of x
    __STATIC_INLINE__ ggml_tensor* ace_repeat_interleave_channels(ggml_context* ctx,
                                                                  ggml_tensor* x,
                                                                  int64_t repeats) {
        const int64_t W = x->ne[0];
        const int64_t H = x->ne[1];
        const int64_t C = x->ne[2];
        GGML_ASSERT(x->ne[3] == 1);
        x = ggml_reshape_3d(ctx, x, W * H, 1, C);
        x = ggml_repeat_4d(ctx, x, W * H, repeats, C, 1);
        x = ggml_reshape_4d(ctx, x, W, H, repeats * C, 1);
        return x;
    }

    struct AceResBlock2D : public GGMLBlock {
        AceResBlock2D(int64_t channels) {
            blocks["conv1"] = std::make_shared<Conv2d>(channels, channels, std::pair<int, int>{3, 3},
                                                       std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
            blocks["conv2"] = std::make_shared<Conv2d>(channels, channels, std::pair<int, int>{3, 3},
                                                       std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1},
                                                       std::pair<int, int>{1, 1}, false);
            blocks["norm"]  = std::make_shared<RMSNorm2d>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
            auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);
            auto norm  = std::dynamic_pointer_cast<RMSNorm2d>(blocks["norm"]);

            auto h = conv1->forward(ctx, x);
            h      = ggml_silu(ctx->ggml_ctx, h);
            h      = conv2->forward(ctx, h);
            h      = norm->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, h, x);
        }
    };

    struct AceGLUMBConv2D : public GGMLBlock {
        AceGLUMBConv2D(int64_t channels, int64_t expand = 4) {
            int64_t hidden          = channels * expand;
            blocks["conv_inverted"] = std::make_shared<Conv2d>(channels, hidden * 2, std::pair<int, int>{1, 1});
            blocks["conv_depth"]    = std::make_shared<Conv2d_grouped>(hidden * 2, hidden * 2, (int)(hidden * 2),
                                                                       std::pair<int, int>{3, 3},
                                                                       std::pair<int, int>{1, 1},
                                                                       std::pair<int, int>{1, 1});
            blocks["conv_point"]    = std::make_shared<Conv2d>(hidden, channels, std::pair<int, int>{1, 1},
                                                               std::pair<int, int>{1, 1}, std::pair<int, int>{0, 0},
                                                               std::pair<int, int>{1, 1}, false);
            blocks["norm"]          = std::make_shared<RMSNorm2d>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv_inverted = std::dynamic_pointer_cast<Conv2d>(blocks["conv_inverted"]);
            auto conv_depth    = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["conv_depth"]);
            auto conv_point    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_point"]);
            auto norm          = std::dynamic_pointer_cast<RMSNorm2d>(blocks["norm"]);
            auto gctx          = ctx->ggml_ctx;

            auto h = conv_inverted->forward(ctx, x);
            h      = ggml_silu(gctx, h);
            h      = conv_depth->forward(ctx, h);

            auto halves = ggml_ext_chunk(gctx, h, 2, 2);
            h           = ggml_mul(gctx, halves[0], ggml_silu(gctx, halves[1]));

            h = conv_point->forward(ctx, h);
            h = norm->forward(ctx, h);
            return ggml_add(gctx, h, x);
        }
    };

    // SanaMultiscaleLinearAttention with one 5x5 multiscale set, exactly as
    // diffusers runs it (including the q/k/v channel regrouping quirk).
    struct AceSanaAttention : public GGMLBlock {
        int64_t channels;
        int64_t head_dim;

        AceSanaAttention(int64_t channels, int64_t head_dim)
            : channels(channels), head_dim(head_dim) {
            blocks["to_q"] = std::make_shared<Linear>(channels, channels, false);
            blocks["to_k"] = std::make_shared<Linear>(channels, channels, false);
            blocks["to_v"] = std::make_shared<Linear>(channels, channels, false);
            blocks["to_qkv_multiscale.0.proj_in"]  = std::make_shared<Conv2d_grouped>(channels * 3, channels * 3, (int)(channels * 3),
                                                                                      std::pair<int, int>{5, 5},
                                                                                      std::pair<int, int>{1, 1},
                                                                                      std::pair<int, int>{2, 2},
                                                                                      std::pair<int, int>{1, 1},
                                                                                      false);
            blocks["to_out"]   = std::make_shared<Linear>(channels * 2, channels, false);
            blocks["norm_out"] = std::make_shared<RMSNorm2d>(channels);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            // grouped 1x1 conv (groups = 3 * num_heads), applied as batched matmul
            params["to_qkv_multiscale.0.proj_out.weight"] =
                ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, head_dim, channels * 3);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto proj_in  = std::dynamic_pointer_cast<Conv2d_grouped>(blocks["to_qkv_multiscale.0.proj_in"]);
            auto to_out   = std::dynamic_pointer_cast<Linear>(blocks["to_out"]);
            auto norm_out = std::dynamic_pointer_cast<RMSNorm2d>(blocks["norm_out"]);
            auto gctx     = ctx->ggml_ctx;

            const int64_t W  = x->ne[0];
            const int64_t H  = x->ne[1];
            const int64_t C  = channels;
            const int64_t HW = W * H;
            const int64_t d  = head_dim;

            auto residual = x;

            // tokens [C, HW]
            auto tokens = ggml_reshape_3d(gctx, x, HW, C, 1);
            tokens      = ggml_ext_cont(gctx, ggml_permute(gctx, tokens, 1, 0, 2, 3));
            tokens      = ggml_reshape_2d(gctx, tokens, C, HW);

            auto q = to_q->forward(ctx, tokens);
            auto k = to_k->forward(ctx, tokens);
            auto v = to_v->forward(ctx, tokens);

            auto qkv = ggml_concat(gctx, ggml_concat(gctx, q, k, 0), v, 0);  // [3C, HW]

            // spatial [W, H, 3C] for the multiscale depthwise conv
            auto qkv_sp = ggml_ext_cont(gctx, ggml_permute(gctx, qkv, 1, 0, 2, 3));  // [HW, 3C]
            qkv_sp      = ggml_reshape_4d(gctx, qkv_sp, W, H, 3 * C, 1);
            auto ms     = proj_in->forward(ctx, qkv_sp);  // depthwise 5x5

            // grouped 1x1 proj_out as batched matmul over 3*num_heads groups
            {
                const int64_t groups = 3 * (C / d);
                auto w  = ggml_reshape_3d(gctx, params["to_qkv_multiscale.0.proj_out.weight"], d, d, groups);
                auto t  = ggml_reshape_3d(gctx, ms, HW, 3 * C, 1);
                t       = ggml_ext_cont(gctx, ggml_permute(gctx, t, 1, 0, 2, 3));  // [3C, HW]
                t       = ggml_reshape_3d(gctx, t, d, groups, HW);
                t       = ggml_ext_cont(gctx, ggml_permute(gctx, t, 0, 2, 1, 3));  // [d, HW, groups]
                t       = ggml_mul_mat(gctx, w, t);                                // [d, HW, groups]
                t       = ggml_ext_cont(gctx, ggml_permute(gctx, t, 0, 2, 1, 3));  // [d, groups, HW]
                ms      = ggml_reshape_2d(gctx, t, 3 * C, HW);
            }

            // concat the base and multiscale sets, regroup into (q, k, v) of
            // head_dim each per group - the diffusers channel-order quirk.
            auto all = ggml_concat(gctx, qkv, ms, 0);  // [6C, HW]
            const int64_t n_groups = (6 * C) / (3 * d);
            all = ggml_reshape_3d(gctx, all, 3 * d, n_groups, HW);

            auto qg = ggml_ext_slice(gctx, all, 0, 0, d);          // [d, G, HW]
            auto kg = ggml_ext_slice(gctx, all, 0, d, 2 * d);
            auto vg = ggml_ext_slice(gctx, all, 0, 2 * d, 3 * d);

            qg = ggml_relu(gctx, qg);
            kg = ggml_relu(gctx, kg);

            // q: [d, HW, G], k/v: [HW, d, G]
            qg = ggml_ext_cont(gctx, ggml_permute(gctx, qg, 0, 2, 1, 3));
            kg = ggml_ext_cont(gctx, ggml_permute(gctx, kg, 1, 2, 0, 3));
            vg = ggml_ext_cont(gctx, ggml_permute(gctx, vg, 1, 2, 0, 3));

            auto ones  = ggml_ext_ones(gctx, HW, 1, n_groups, 1);
            auto v_pad = ggml_concat(gctx, vg, ones, 1);  // [HW, d+1, G]

            auto vk = ggml_mul_mat(gctx, kg, v_pad);  // [d, d+1, G]
            auto hs = ggml_mul_mat(gctx, vk, qg);     // [d+1, HW, G]

            auto num = ggml_ext_slice(gctx, hs, 0, 0, d);
            auto den = ggml_ext_slice(gctx, hs, 0, d, d + 1);
            auto eps = ggml_ext_scale(gctx, ggml_ext_ones(gctx, 1, 1, 1, 1), 1e-15f);
            den      = ggml_add(gctx, den, eps);
            auto out = ggml_div(gctx, num, den);  // [d, HW, G]

            out = ggml_ext_cont(gctx, ggml_permute(gctx, out, 0, 2, 1, 3));  // [d, G, HW]
            out = ggml_reshape_2d(gctx, out, 2 * C, HW);

            out = to_out->forward(ctx, out);  // [C, HW]

            out = ggml_ext_cont(gctx, ggml_permute(gctx, out, 1, 0, 2, 3));  // [HW, C]
            out = ggml_reshape_4d(gctx, out, W, H, C, 1);
            out = norm_out->forward(ctx, out);
            return ggml_add(gctx, out, residual);
        }
    };

    struct AceEViTBlock : public GGMLBlock {
        AceEViTBlock(int64_t channels, int64_t head_dim) {
            blocks["attn"]     = std::make_shared<AceSanaAttention>(channels, head_dim);
            blocks["conv_out"] = std::make_shared<AceGLUMBConv2D>(channels);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto attn     = std::dynamic_pointer_cast<AceSanaAttention>(blocks["attn"]);
            auto conv_out = std::dynamic_pointer_cast<AceGLUMBConv2D>(blocks["conv_out"]);
            x             = attn->forward(ctx, x);
            x             = conv_out->forward(ctx, x);
            return x;
        }
    };

    // DCUpBlock2d, interpolate variant with the repeat/pixel-shuffle shortcut
    struct AceUpBlock : public GGMLBlock {
        int64_t in_channels;
        int64_t out_channels;

        AceUpBlock(int64_t in_channels, int64_t out_channels)
            : in_channels(in_channels), out_channels(out_channels) {
            blocks["conv"] = std::make_shared<Conv2d>(in_channels, out_channels, std::pair<int, int>{3, 3},
                                                      std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto conv = std::dynamic_pointer_cast<Conv2d>(blocks["conv"]);
            auto gctx = ctx->ggml_ctx;

            auto up  = ggml_upscale(gctx, x, 2, GGML_SCALE_MODE_NEAREST);
            auto out = conv->forward(ctx, up);

            const int64_t repeats = out_channels * 4 / in_channels;
            auto shortcut         = ace_repeat_interleave_channels(gctx, x, repeats);
            shortcut              = ace_pixel_shuffle_2x(gctx, shortcut);
            return ggml_add(gctx, out, shortcut);
        }
    };

    struct AceDCAEDecoder : public GGMLBlock {
        AceVAEConfig config;

        AceDCAEDecoder(const AceVAEConfig& config)
            : config(config) {
            const auto& ch = config.block_out_channels;

            blocks["conv_in"] = std::make_shared<Conv2d>(config.latent_channels, ch.back(), std::pair<int, int>{3, 3},
                                                         std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});

            const int num_stages = (int)ch.size();
            for (int i = 0; i < num_stages; ++i) {
                int item = 0;
                if (i < num_stages - 1) {
                    blocks["up_blocks." + std::to_string(i) + "." + std::to_string(item)] =
                        std::make_shared<AceUpBlock>(ch[i + 1], ch[i]);
                    ++item;
                }
                for (int j = 0; j < config.layers_per_block; ++j, ++item) {
                    std::string name = "up_blocks." + std::to_string(i) + "." + std::to_string(item);
                    if (i == num_stages - 1) {
                        blocks[name] = std::make_shared<AceEViTBlock>(ch[i], config.attention_head_dim);
                    } else {
                        blocks[name] = std::make_shared<AceResBlock2D>(ch[i]);
                    }
                }
            }

            blocks["norm_out"] = std::make_shared<RMSNorm2d>(ch.front());
            blocks["conv_out"] = std::make_shared<Conv2d>(ch.front(), config.audio_channels, std::pair<int, int>{3, 3},
                                                          std::pair<int, int>{1, 1}, std::pair<int, int>{1, 1});
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* z) {
            auto conv_in  = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
            auto norm_out = std::dynamic_pointer_cast<RMSNorm2d>(blocks["norm_out"]);
            auto conv_out = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);
            auto gctx     = ctx->ggml_ctx;

            const auto& ch = config.block_out_channels;

            // conv_in + in_shortcut (repeat_interleave over channels)
            auto x = conv_in->forward(ctx, z);
            x      = ggml_add(gctx, x, ace_repeat_interleave_channels(gctx, z, ch.back() / config.latent_channels));

            const int num_stages = (int)ch.size();
            for (int i = num_stages - 1; i >= 0; --i) {
                int item = 0;
                if (i < num_stages - 1) {
                    auto up = std::dynamic_pointer_cast<AceUpBlock>(blocks["up_blocks." + std::to_string(i) + ".0"]);
                    x       = up->forward(ctx, x);
                    ++item;
                }
                for (int j = 0; j < config.layers_per_block; ++j, ++item) {
                    std::string name = "up_blocks." + std::to_string(i) + "." + std::to_string(item);
                    if (i == num_stages - 1) {
                        x = std::dynamic_pointer_cast<AceEViTBlock>(blocks[name])->forward(ctx, x);
                    } else {
                        x = std::dynamic_pointer_cast<AceResBlock2D>(blocks[name])->forward(ctx, x);
                    }
                }
            }

            x = norm_out->forward(ctx, x);
            x = ggml_relu(gctx, x);
            x = conv_out->forward(ctx, x);
            return x;  // [8T, 128, 2, 1]
        }
    };

    /* ------------------------------- vocoder ------------------------------- */

    // LayerNorm over the channel dim of [L, C, N]
    struct ChannelsFirstLayerNorm : public UnaryBlock {
        int64_t channels;
        float eps;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        }

        ChannelsFirstLayerNorm(int64_t channels, float eps = 1e-6f)
            : channels(channels), eps(eps) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto h    = ggml_ext_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [C, L, N]
            h         = ggml_ext_layer_norm(gctx, h, params["weight"], params["bias"], eps);
            h         = ggml_ext_cont(gctx, ggml_permute(gctx, h, 1, 0, 2, 3));
            return h;
        }
    };

    // gk's conv_1d_dw takes no batch dimension; run each batch element on its own
    __STATIC_INLINE__ ggml_tensor* ace_conv_1d_dw_batched(ggml_context* ctx,
                                                          ggml_tensor* w,
                                                          ggml_tensor* x,  // [L, C, N]
                                                          int stride,
                                                          int padding,
                                                          int dilation) {
        const int64_t n = x->ne[2];
        if (n == 1) {
            return ggml_conv_1d_dw(ctx, w, x, stride, padding, dilation);
        }
        ggml_tensor* out = nullptr;
        for (int64_t i = 0; i < n; ++i) {
            auto xi = ggml_ext_slice(ctx, x, 2, i, i + 1);
            auto oi = ggml_conv_1d_dw(ctx, w, xi, stride, padding, dilation);
            oi      = ggml_reshape_3d(ctx, oi, oi->ne[0], oi->ne[1], 1);
            out     = out == nullptr ? oi : ggml_concat(ctx, out, oi, 2);
        }
        return out;
    }

    struct AceConvNeXtBlock : public GGMLBlock {
        int64_t dim;
        int64_t hidden;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["dwconv.weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 1, dim);
            params["dwconv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["gamma"]         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
        }

        AceConvNeXtBlock(int64_t dim, int64_t hidden)
            : dim(dim), hidden(hidden) {
            blocks["norm"]    = std::make_shared<LayerNorm>(dim, 1e-6f);
            blocks["pwconv1"] = std::make_shared<Linear>(dim, hidden);
            blocks["pwconv2"] = std::make_shared<Linear>(hidden, dim);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto norm    = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto pwconv1 = std::dynamic_pointer_cast<Linear>(blocks["pwconv1"]);
            auto pwconv2 = std::dynamic_pointer_cast<Linear>(blocks["pwconv2"]);
            auto gctx    = ctx->ggml_ctx;

            const int64_t L = x->ne[0];
            const int64_t N = x->ne[2];

            auto h = ace_conv_1d_dw_batched(gctx, params["dwconv.weight"], x, 1, 3, 1);
            h      = ggml_add(gctx, h, ggml_reshape_3d(gctx, params["dwconv.bias"], 1, dim, 1));

            h = ggml_ext_cont(gctx, ggml_permute(gctx, h, 1, 0, 2, 3));  // [C, L, N]
            h = norm->forward(ctx, h);
            h = pwconv1->forward(ctx, h);
            h = ggml_gelu_erf(gctx, h);
            h = pwconv2->forward(ctx, h);
            h = ggml_mul(gctx, h, params["gamma"]);
            h = ggml_ext_cont(gctx, ggml_permute(gctx, h, 1, 0, 2, 3));  // [L, C, N]

            return ggml_add(gctx, h, x);
        }
    };

    // weight-norm fold: w = v * g / ||v|| with the norm over (ne0, ne1) per ne2
    __STATIC_INLINE__ ggml_tensor* ace_fold_weight_norm(ggml_context* ctx,
                                                        ggml_tensor* weight_v,
                                                        ggml_tensor* weight_g) {
        const int64_t k   = weight_v->ne[0];
        const int64_t c1  = weight_v->ne[1];
        const int64_t c2  = weight_v->ne[2];
        auto v            = weight_v;
        if (v->type != GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F32);
        }
        auto flat  = ggml_reshape_2d(ctx, v, k * c1, c2);
        auto norm  = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, flat)));  // [1, c2]
        auto g     = weight_g;
        if (g->type != GGML_TYPE_F32) {
            g = ggml_cast(ctx, g, GGML_TYPE_F32);
        }
        g          = ggml_reshape_2d(ctx, g, 1, c2);
        auto scale = ggml_div(ctx, g, norm);           // [1, c2]
        auto w     = ggml_mul(ctx, flat, scale);       // [k*c1, c2]
        return ggml_reshape_3d(ctx, w, k, c1, c2);
    }

    struct WNConv1d : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int padding;
        int dilation;
        bool bias;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight_v"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
            params["weight_g"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, out_channels);
            if (bias) {
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }
        }

        WNConv1d(int64_t in_channels, int64_t out_channels, int kernel_size, int padding, int dilation = 1, bool bias = true)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              padding(padding),
              dilation(dilation),
              bias(bias) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto w    = ace_fold_weight_norm(gctx, params["weight_v"], params["weight_g"]);
            x         = ggml_conv_1d(gctx, w, x, 1, padding, dilation);
            if (bias) {
                x = ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
            }
            return x;
        }
    };

    struct WNConvTranspose1d : public UnaryBlock {
        int64_t in_channels;
        int64_t out_channels;
        int kernel_size;
        int stride;
        int padding;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            params["weight_v"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, out_channels, in_channels);
            params["weight_g"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, in_channels);
            params["bias"]     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }

        WNConvTranspose1d(int64_t in_channels, int64_t out_channels, int kernel_size, int stride, int padding)
            : in_channels(in_channels),
              out_channels(out_channels),
              kernel_size(kernel_size),
              stride(stride),
              padding(padding) {}

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto gctx = ctx->ggml_ctx;
            auto w    = ace_fold_weight_norm(gctx, params["weight_v"], params["weight_g"]);
            x         = ggml_conv_transpose_1d(gctx, w, x, stride, 0, 1);
            if (padding > 0) {
                x = ggml_ext_slice(gctx, x, 0, padding, x->ne[0] - padding);
            }
            x = ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
            return x;
        }
    };

    struct AceHiFiGANResBlock : public GGMLBlock {
        AceHiFiGANResBlock(int64_t channels, int kernel_size, const std::vector<int>& dilation) {
            for (int i = 0; i < 3; ++i) {
                blocks["convs1." + std::to_string(i)] = std::make_shared<WNConv1d>(channels, channels, kernel_size,
                                                                                   (kernel_size * dilation[i] - dilation[i]) / 2,
                                                                                   dilation[i]);
                blocks["convs2." + std::to_string(i)] = std::make_shared<WNConv1d>(channels, channels, kernel_size,
                                                                                   (kernel_size - 1) / 2,
                                                                                   1);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto gctx = ctx->ggml_ctx;
            for (int i = 0; i < 3; ++i) {
                auto c1 = std::dynamic_pointer_cast<WNConv1d>(blocks["convs1." + std::to_string(i)]);
                auto c2 = std::dynamic_pointer_cast<WNConv1d>(blocks["convs2." + std::to_string(i)]);
                auto h  = ggml_silu(gctx, x);
                h       = c1->forward(ctx, h);
                h       = ggml_silu(gctx, h);
                h       = c2->forward(ctx, h);
                x       = ggml_add(gctx, x, h);
            }
            return x;
        }
    };

    struct AceVocoder : public GGMLBlock {
        AceVAEConfig config;

        AceVocoder(const AceVAEConfig& config)
            : config(config) {
            // ConvNeXt backbone
            const auto& dims = config.backbone_dims;
            blocks["backbone.channel_layers.0.0"] = std::make_shared<Conv1D_Replicate>(config.mel_bins, dims[0], 7, 3);
            blocks["backbone.channel_layers.0.1"] = std::make_shared<ChannelsFirstLayerNorm>(dims[0]);
            for (int i = 1; i < 4; ++i) {
                blocks["backbone.channel_layers." + std::to_string(i) + ".0"] = std::make_shared<ChannelsFirstLayerNorm>(dims[i - 1]);
                blocks["backbone.channel_layers." + std::to_string(i) + ".1"] = std::make_shared<Conv1DPlain>(dims[i - 1], dims[i], 1, 0);
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < config.backbone_depths[i]; ++j) {
                    blocks["backbone.stages." + std::to_string(i) + "." + std::to_string(j)] =
                        std::make_shared<AceConvNeXtBlock>(dims[i], dims[i] * config.backbone_mlp);
                }
            }
            blocks["backbone.norm"] = std::make_shared<ChannelsFirstLayerNorm>(dims[3]);

            // HiFi-GAN head
            blocks["head.conv_pre"] = std::make_shared<WNConv1d>(dims[3], config.head_initial_channels,
                                                                 config.pre_conv_kernel,
                                                                 (config.pre_conv_kernel - 1) / 2);
            int64_t ch = config.head_initial_channels;
            int rb     = 0;
            for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
                int64_t next = config.head_initial_channels / (int64_t)(1u << (i + 1));
                blocks["head.ups." + std::to_string(i)] =
                    std::make_shared<WNConvTranspose1d>(ch, next,
                                                        config.upsample_kernels[i],
                                                        config.upsample_rates[i],
                                                        (config.upsample_kernels[i] - config.upsample_rates[i]) / 2);
                for (size_t j = 0; j < config.resblock_kernels.size(); ++j, ++rb) {
                    blocks["head.resblocks." + std::to_string(rb)] =
                        std::make_shared<AceHiFiGANResBlock>(next, config.resblock_kernels[j], config.resblock_dilations[j]);
                }
                ch = next;
            }
            blocks["head.conv_post"] = std::make_shared<WNConv1d>(ch, 1,
                                                                  config.post_conv_kernel,
                                                                  (config.post_conv_kernel - 1) / 2);
        }

        // stem conv with replicate padding
        struct Conv1D_Replicate : public UnaryBlock {
            int64_t in_channels;
            int64_t out_channels;
            int kernel_size;
            int padding;

            void init_params(ggml_context* ctx,
                             const String2TensorStorage& tensor_storage_map = {},
                             const std::string prefix                       = "") override {
                params["weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
                params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }

            Conv1D_Replicate(int64_t in_channels, int64_t out_channels, int kernel_size, int padding)
                : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), padding(padding) {}

            ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
                auto gctx = ctx->ggml_ctx;
                x         = LTXV::replicate_pad_1d(ctx, x, padding, padding);
                x         = ggml_conv_1d(gctx, params["weight"], x, 1, 0, 1);
                x         = ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
                return x;
            }
        };

        struct Conv1DPlain : public UnaryBlock {
            int64_t in_channels;
            int64_t out_channels;
            int kernel_size;
            int padding;

            void init_params(ggml_context* ctx,
                             const String2TensorStorage& tensor_storage_map = {},
                             const std::string prefix                       = "") override {
                params["weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel_size, in_channels, out_channels);
                params["bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
            }

            Conv1DPlain(int64_t in_channels, int64_t out_channels, int kernel_size, int padding)
                : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), padding(padding) {}

            ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
                auto gctx = ctx->ggml_ctx;
                x         = ggml_conv_1d(gctx, params["weight"], x, 1, padding, 1);
                x         = ggml_add(gctx, x, ggml_reshape_3d(gctx, params["bias"], 1, out_channels, 1));
                return x;
            }
        };

        // mel: [L, mel_bins, N]
        ggml_tensor* forward_backbone(GGMLRunnerContext* ctx, ggml_tensor* mel) {
            auto x = std::dynamic_pointer_cast<Conv1D_Replicate>(blocks["backbone.channel_layers.0.0"])->forward(ctx, mel);
            x      = std::dynamic_pointer_cast<ChannelsFirstLayerNorm>(blocks["backbone.channel_layers.0.1"])->forward(ctx, x);
            for (int i = 0; i < 4; ++i) {
                if (i > 0) {
                    x = std::dynamic_pointer_cast<ChannelsFirstLayerNorm>(blocks["backbone.channel_layers." + std::to_string(i) + ".0"])->forward(ctx, x);
                    x = std::dynamic_pointer_cast<Conv1DPlain>(blocks["backbone.channel_layers." + std::to_string(i) + ".1"])->forward(ctx, x);
                }
                for (int j = 0; j < config.backbone_depths[i]; ++j) {
                    x = std::dynamic_pointer_cast<AceConvNeXtBlock>(blocks["backbone.stages." + std::to_string(i) + "." + std::to_string(j)])->forward(ctx, x);
                }
            }
            x = std::dynamic_pointer_cast<ChannelsFirstLayerNorm>(blocks["backbone.norm"])->forward(ctx, x);
            return x;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* mel) {
            auto gctx = ctx->ggml_ctx;

            auto x = forward_backbone(ctx, mel);

            // head
            x      = std::dynamic_pointer_cast<WNConv1d>(blocks["head.conv_pre"])->forward(ctx, x);
            int rb = 0;
            for (size_t i = 0; i < config.upsample_rates.size(); ++i) {
                x = ggml_silu(gctx, x);
                x = std::dynamic_pointer_cast<WNConvTranspose1d>(blocks["head.ups." + std::to_string(i)])->forward(ctx, x);

                ggml_tensor* sum = nullptr;
                for (size_t j = 0; j < config.resblock_kernels.size(); ++j, ++rb) {
                    auto block_out = std::dynamic_pointer_cast<AceHiFiGANResBlock>(blocks["head.resblocks." + std::to_string(rb)])->forward(ctx, x);
                    sum            = sum == nullptr ? block_out : ggml_add(gctx, sum, block_out);
                }
                x = ggml_ext_scale(gctx, sum, 1.0f / (float)config.resblock_kernels.size());
            }
            x = ggml_silu(gctx, x);
            x = std::dynamic_pointer_cast<WNConv1d>(blocks["head.conv_post"])->forward(ctx, x);
            x = ggml_tanh(gctx, x);
            return x;  // [samples, 1, N]
        }
    };

    struct AceMusicVAE : public GGMLBlock {
        AceVAEConfig config;

        explicit AceMusicVAE(const AceVAEConfig& config)
            : config(config) {
            blocks["dcae.decoder"] = std::make_shared<AceDCAEDecoder>(config);
            blocks["vocoder"]      = std::make_shared<AceVocoder>(config);
        }

        ggml_tensor* decode_mel(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            auto decoder = std::dynamic_pointer_cast<AceDCAEDecoder>(blocks["dcae.decoder"]);
            auto gctx    = ctx->ggml_ctx;

            // undo the diffusion-space normalization
            auto shift = ggml_ext_scale(gctx, ggml_ext_ones(gctx, 1, 1, 1, 1), ACE_LATENT_SHIFT);
            auto z     = ggml_add(gctx, ggml_ext_scale(gctx, latent, 1.0f / ACE_LATENT_SCALE), shift);

            auto mel = decoder->forward(ctx, z);  // [8T, 128, 2, 1], in [-1, 1]

            // [0,1] -> log-mel range
            const float mel_scale = 0.5f * (ACE_MAX_MEL_VALUE - ACE_MIN_MEL_VALUE);
            const float mel_shift = 0.5f * (ACE_MAX_MEL_VALUE - ACE_MIN_MEL_VALUE) + ACE_MIN_MEL_VALUE;
            auto mel_bias         = ggml_ext_scale(gctx, ggml_ext_ones(gctx, 1, 1, 1, 1), mel_shift);
            mel                   = ggml_add(gctx, ggml_ext_scale(gctx, mel, mel_scale), mel_bias);

            // stereo channels ride the batch dim through the vocoder
            return ggml_reshape_3d(gctx, mel, mel->ne[0], mel->ne[1], mel->ne[2]);
        }

        ggml_tensor* decode_backbone(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            auto vocoder = std::dynamic_pointer_cast<AceVocoder>(blocks["vocoder"]);
            return vocoder->forward_backbone(ctx, decode_mel(ctx, latent));
        }

        ggml_tensor* decode(GGMLRunnerContext* ctx, ggml_tensor* latent) {
            auto vocoder = std::dynamic_pointer_cast<AceVocoder>(blocks["vocoder"]);
            auto gctx    = ctx->ggml_ctx;

            auto mel = decode_mel(ctx, latent);  // [L, mel_bins, channels]

            // gk's conv_1d family scrambles a batched (ne2 > 1) input, so each
            // stereo channel takes its own pass through the vocoder.
            ggml_tensor* waveform = nullptr;
            for (int64_t c = 0; c < mel->ne[2]; ++c) {
                auto mel_c = ggml_ext_slice(gctx, mel, 2, c, c + 1);       // [L, mel_bins, 1]
                auto wav_c = vocoder->forward(ctx, mel_c);                 // [samples, 1, 1]
                wav_c      = ggml_reshape_2d(gctx, wav_c, wav_c->ne[0], 1);
                waveform   = waveform == nullptr ? wav_c : ggml_concat(gctx, waveform, wav_c, 1);
            }
            return waveform;  // [samples, channels]
        }
    };

    struct AceVAERunner : public ::AudioVAERunner {
        AceVAEConfig config;
        AceMusicVAE model;
        std::string weight_prefix;

        AceVAERunner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map,
                     const std::string& prefix                           = "vae",
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : ::AudioVAERunner(backend, weight_manager),
              weight_prefix(prefix),
              config(AceVAEConfig::detect_from_weights(tensor_storage_map, prefix)),
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
            return "ace_music_vae";
        }

        int output_sample_rate() const override {
            return config.sample_rate;
        }

        sd::Tensor<float> decode(int n_threads,
                                 const sd::Tensor<float>& latent_tensor) override {
            int64_t t0 = ggml_time_ms();
            // SD_ACE_DECODE_STAGE=mel: stop the graph at the mel spectrogram
            // (debugging aid for telling a DCAE problem from a vocoder one)
            const char* stage = getenv("SD_ACE_DECODE_STAGE");
            auto get_graph    = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(65536);
                auto latent     = make_input(latent_tensor);
                auto runner_ctx = GGMLRunner::get_context();
                ggml_tensor* out;
                if (stage != nullptr && std::string(stage) == "mel") {
                    out = model.decode_mel(&runner_ctx, latent);
                } else if (stage != nullptr && std::string(stage) == "backbone") {
                    out = model.decode_backbone(&runner_ctx, latent);
                } else {
                    out = model.decode(&runner_ctx, latent);
                }
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, false, false, false), 4);
            int64_t t1  = ggml_time_ms();
            LOG_INFO("ace music vae decode completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
            if (const char* dump = getenv("SD_ACE_DECODE_DUMP")) {
                if (FILE* f = fopen(dump, "wb")) {
                    fwrite(result.data(), sizeof(float), (size_t)result.numel(), f);
                    fclose(f);
                    LOG_INFO("SD_ACE_DECODE_DUMP: wrote %s (%" PRId64 " floats, shape %" PRId64 "x%" PRId64 "x%" PRId64 ")",
                             dump, result.numel(), result.shape()[0], result.shape()[1],
                             result.dim() > 2 ? result.shape()[2] : 1);
                }
            }
            return result;
        }
    };

}  // namespace AceStep

#endif  // __SD_MODEL_VAE_ACE_VAE_HPP__
