#ifndef __SD_MODEL_TE_LLM_ADAPTER_HPP__
#define __SD_MODEL_TE_LLM_ADAPTER_HPP__

#include <cinttypes>
#include <string>

#include "core/ggml_extend.hpp"
#include "core/util.h"
#include "model_loader.h"

// Qwen3 -> T5-XXL bridge adapter (trainer/adapter.py).
//
// A Perceiver-style resampler: `num_queries` learned queries, each seeded with
// the T5 sentencepiece token embedding of its slot, self-attend, cross-attend
// over the Qwen3 hidden states (taken AFTER the LLM's final RMSNorm) and are
// projected to T5-XXL's embedding space.  The output is a drop-in replacement
// for the T5 encoder output PixArt's cross attention was trained on.
//
// gguf contract (trainer/export_gguf.py): every tensor is the PyTorch
// state_dict name under an "adapter." prefix; head_dim is fixed at 64 so the
// head count follows from `width`; LayerNorm eps is torch's 1e-5; the MLP
// activation is exact (erf) GELU.
//
// The query-seed embedding comes in two layouts (trainer3/export_gguf.py):
//   full:       t5_embed.weight [width, t5_vocab]        seed = E[id]
//   factorized: t5_embed_a.weight [rank, t5_vocab] +
//               t5_embed_b.weight [rank, width]          seed = B(A[id])
// The presence of t5_embed_a.weight selects the factorized path; both encode
// the same function, factorized is ~2.6x smaller on disk.
//
// TOKEN-ALIGNED variant (trainer4/adapter.py, arch qwen3_qwen3_adapter):
// student and teacher share the Qwen tokenizer, so there are no learned
// queries and no seed embedding -- the adapter maps the student LLM's hidden
// state at every position to the teacher LLM's hidden state at the SAME
// position (Qwen3-0.6B final-norm 1024 -> Qwen3-4B layer-35 2560 for
// Z-Image).  Layout: in_proj, blocks.N.{ln_self,self_attn,ln_mlp,mlp}, ln_out,
// out_proj, plus a direct linear `skip` path:
//     out = out_proj(ln_out(blocks(in_proj(h)))) + skip(h)
// The ABSENCE of the `query` tensor selects this variant; output length
// follows the input length.
//
// VISION extension of the token-aligned variant (trainer5/adapter.py, for
// MageFlow-Edit): the teacher is a VL model whose mmproj embeds (2560) are
// spliced into the input sequence.  Two extra bias-free linears:
//   vision_proj [vis_dim -> in_dim]  applied by the ENGINE to every mmproj
//                                    image embed before the student LLM
//                                    (frozen least-squares vocab map)
//   vis_in      [vis_dim -> width]   inside the adapter: the RAW mmproj
//                                    embeds, laid out densely per position
//                                    (zeros at text positions), are added
//                                    to the trunk after in_proj
// Selected by the presence of `vision_proj.weight`.
//
// LAYERWISE extension of the token-aligned variant (trainer6/adapter.py, for
// krea2-style models that condition on a STACK of intermediate teacher
// layers): out_proj and skip map to n_out_layers * teacher_dim, laid out as
// the ascending-tap-order concatenation the conditioner produces for the
// full teacher, so the adapter output IS the stacked context.  Selected by
// the presence of `layer_map` (f32 [n_out_layers], the tap indices, kept for
// provenance); the conditioner refuses an adapter whose layer count does not
// match the model's out_layers.
namespace LLMAdapter {

    constexpr int64_t HEAD_DIM = 64;
    constexpr size_t ADAPTER_GRAPH_SIZE = 8192;

    struct AdapterConfig {
        int64_t in_dim      = 1024;
        int64_t out_dim     = 4096;
        int64_t width       = 1024;
        int64_t depth       = 6;
        int64_t heads       = 16;
        int64_t num_queries = 120;
        int64_t t5_vocab    = 32128;
        int64_t embed_rank  = 0;  // 0 = full-width t5_embed table
        int64_t mlp_hidden  = 4096;
        float norm_eps      = 1e-5f;
        bool token_aligned  = false;  // no queries/seed: out[i] = f(hidden[i])
        bool has_skip       = false;  // direct linear in_dim -> out_dim path
        int64_t vis_dim     = 0;      // raw mmproj embed dim (vision ext)
        bool has_vision_proj = false;  // vis_dim -> in_dim student input map
        bool has_vis_in      = false;  // vis_dim -> width trunk injection
        int64_t n_out_layers = 1;      // >1: out_dim stacks this many teacher layers

        static AdapterConfig detect_from_weights(const String2TensorStorage& tensor_storage_map,
                                                 const std::string& prefix) {
            AdapterConfig config;
            const std::string p  = prefix + ".";
            int64_t max_block    = -1;
            bool has_query       = false;
            for (const auto& [name, ts] : tensor_storage_map) {
                if (!starts_with(name, p)) {
                    continue;
                }
                const std::string rest = name.substr(p.size());
                if (rest == "query" && ts.n_dims == 2) {
                    has_query          = true;
                    config.width       = ts.ne[0];
                    config.num_queries = ts.ne[1];
                } else if (rest == "in_proj.weight" && ts.n_dims == 2) {
                    // [in_dim, width] in both variants; `query` (seen later in
                    // the sorted map) also carries width and agrees with this
                    config.in_dim = ts.ne[0];
                    config.width  = ts.ne[1];
                } else if (rest == "skip.weight" && ts.n_dims == 2) {
                    config.has_skip = true;
                } else if (rest == "vision_proj.weight" && ts.n_dims == 2) {
                    config.has_vision_proj = true;
                    config.vis_dim         = ts.ne[0];
                } else if (rest == "vis_in.weight" && ts.n_dims == 2) {
                    config.has_vis_in = true;
                    config.vis_dim    = ts.ne[0];
                } else if (rest == "layer_map" && ts.n_dims == 1) {
                    config.n_out_layers = ts.ne[0];
                } else if (rest == "out_proj.weight" && ts.n_dims == 2) {
                    config.out_dim = ts.ne[1];
                } else if (rest == "t5_embed.weight" && ts.n_dims == 2) {
                    config.t5_vocab = ts.ne[1];
                } else if (rest == "t5_embed_a.weight" && ts.n_dims == 2) {
                    config.embed_rank = ts.ne[0];
                    config.t5_vocab   = ts.ne[1];
                } else if (rest == "blocks.0.mlp.0.weight" && ts.n_dims == 2) {
                    config.mlp_hidden = ts.ne[1];
                } else if (starts_with(rest, "blocks.")) {
                    size_t dot = rest.find('.', 7);
                    if (dot != std::string::npos) {
                        try {
                            max_block = std::max<int64_t>(max_block, std::stoll(rest.substr(7, dot - 7)));
                        } catch (...) {
                        }
                    }
                }
            }
            config.depth         = max_block + 1;
            config.heads         = config.width / HEAD_DIM;
            config.token_aligned = !has_query;
            if (config.token_aligned) {
                config.num_queries = 0;
                config.t5_vocab    = 0;
            }
            LOG_INFO("llm adapter: %s in=%" PRId64 " out=%" PRId64 " width=%" PRId64 " depth=%" PRId64
                     " heads=%" PRId64 " queries=%" PRId64 " t5_vocab=%" PRId64 " embed_rank=%" PRId64
                     " mlp=%" PRId64 " skip=%d",
                     config.token_aligned ? "token-aligned" : "resampler",
                     config.in_dim, config.out_dim, config.width, config.depth, config.heads,
                     config.num_queries, config.t5_vocab, config.embed_rank, config.mlp_hidden,
                     config.has_skip ? 1 : 0);
            if (config.has_vision_proj || config.has_vis_in) {
                LOG_INFO("llm adapter: vision ext (vis_dim=%" PRId64 " vision_proj=%d vis_in=%d)",
                         config.vis_dim, config.has_vision_proj ? 1 : 0, config.has_vis_in ? 1 : 0);
            }
            if (config.n_out_layers > 1) {
                LOG_INFO("llm adapter: layerwise ext (%" PRId64 " stacked layers, %" PRId64 " dims each)",
                         config.n_out_layers, config.out_dim / config.n_out_layers);
            }
            return config;
        }
    };

    // trainer/adapter.py Attention: explicit q/k/v/o linears, no packed in_proj.
    struct Attention : public GGMLBlock {
        int64_t heads;

        Attention(int64_t width, int64_t heads)
            : heads(heads) {
            blocks["q"] = std::make_shared<Linear>(width, width);
            blocks["k"] = std::make_shared<Linear>(width, width);
            blocks["v"] = std::make_shared<Linear>(width, width);
            blocks["o"] = std::make_shared<Linear>(width, width);
        }

        // xq: [width, L_q, N], xkv: [width, L_k, N], mask: additive [L_k, 1] or nullptr
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* xq, ggml_tensor* xkv, ggml_tensor* mask = nullptr) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v"]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o"]);

            auto q   = q_proj->forward(ctx, xq);
            auto k   = k_proj->forward(ctx, xkv);
            auto v   = v_proj->forward(ctx, xkv);
            auto out = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, heads, mask, false, ctx->flash_attn_enabled);
            return o_proj->forward(ctx, out);
        }
    };

    // trainer/adapter.py Block: pre-LN self attention, cross attention, MLP.
    struct Block : public GGMLBlock {
        Block(int64_t width, int64_t heads, int64_t mlp_hidden, float eps) {
            blocks["ln_self"]    = std::make_shared<LayerNorm>(width, eps);
            blocks["self_attn"]  = std::make_shared<Attention>(width, heads);
            blocks["ln_q"]       = std::make_shared<LayerNorm>(width, eps);
            blocks["ln_kv"]      = std::make_shared<LayerNorm>(width, eps);
            blocks["cross_attn"] = std::make_shared<Attention>(width, heads);
            blocks["ln_mlp"]     = std::make_shared<LayerNorm>(width, eps);
            blocks["mlp.0"]      = std::make_shared<Linear>(width, mlp_hidden);
            blocks["mlp.2"]      = std::make_shared<Linear>(mlp_hidden, width);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* q, ggml_tensor* kv, ggml_tensor* kv_mask) {
            auto ln_self    = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_self"]);
            auto self_attn  = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto ln_q       = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_q"]);
            auto ln_kv      = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_kv"]);
            auto cross_attn = std::dynamic_pointer_cast<Attention>(blocks["cross_attn"]);
            auto ln_mlp     = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_mlp"]);
            auto mlp_0      = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2      = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            auto x = ln_self->forward(ctx, q);
            q      = ggml_add(ctx->ggml_ctx, q, self_attn->forward(ctx, x, x));
            q      = ggml_add(ctx->ggml_ctx, q, cross_attn->forward(ctx, ln_q->forward(ctx, q), ln_kv->forward(ctx, kv), kv_mask));
            auto h = mlp_0->forward(ctx, ln_mlp->forward(ctx, q));
            h      = ggml_gelu_erf(ctx->ggml_ctx, h);  // torch nn.GELU() default: exact
            q      = ggml_add(ctx->ggml_ctx, q, mlp_2->forward(ctx, h));
            return q;
        }
    };

    // trainer4/adapter.py Block: pre-LN self attention + MLP, no cross path.
    struct TokenBlock : public GGMLBlock {
        TokenBlock(int64_t width, int64_t heads, int64_t mlp_hidden, float eps) {
            blocks["ln_self"]   = std::make_shared<LayerNorm>(width, eps);
            blocks["self_attn"] = std::make_shared<Attention>(width, heads);
            blocks["ln_mlp"]    = std::make_shared<LayerNorm>(width, eps);
            blocks["mlp.0"]     = std::make_shared<Linear>(width, mlp_hidden);
            blocks["mlp.2"]     = std::make_shared<Linear>(mlp_hidden, width);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* mask) {
            auto ln_self   = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_self"]);
            auto self_attn = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto ln_mlp    = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_mlp"]);
            auto mlp_0     = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
            auto mlp_2     = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);

            auto h = ln_self->forward(ctx, x);
            x      = ggml_add(ctx->ggml_ctx, x, self_attn->forward(ctx, h, h, mask));
            h      = mlp_0->forward(ctx, ln_mlp->forward(ctx, x));
            h      = ggml_gelu_erf(ctx->ggml_ctx, h);  // torch nn.GELU() default: exact
            x      = ggml_add(ctx->ggml_ctx, x, mlp_2->forward(ctx, h));
            return x;
        }
    };

    struct Adapter : public GGMLBlock {
        AdapterConfig config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            if (!config.token_aligned) {
                params["query"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.width, config.num_queries);
            }
            if (config.n_out_layers > 1) {
                // tap indices, provenance only (the forward never reads it)
                params["layer_map"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.n_out_layers);
            }
        }

        Adapter(const AdapterConfig& config)
            : config(config) {
            if (!config.token_aligned) {
                if (config.embed_rank > 0) {
                    blocks["t5_embed_a"] = std::make_shared<Embedding>(config.t5_vocab, config.embed_rank);
                    blocks["t5_embed_b"] = std::make_shared<Linear>(config.embed_rank, config.width, /*bias=*/false);
                } else {
                    blocks["t5_embed"] = std::make_shared<Embedding>(config.t5_vocab, config.width);
                }
            }
            blocks["in_proj"] = std::make_shared<Linear>(config.in_dim, config.width);
            for (int64_t i = 0; i < config.depth; ++i) {
                if (config.token_aligned) {
                    blocks["blocks." + std::to_string(i)] =
                        std::make_shared<TokenBlock>(config.width, config.heads, config.mlp_hidden, config.norm_eps);
                } else {
                    blocks["blocks." + std::to_string(i)] =
                        std::make_shared<Block>(config.width, config.heads, config.mlp_hidden, config.norm_eps);
                }
            }
            blocks["ln_out"]   = std::make_shared<LayerNorm>(config.width, config.norm_eps);
            blocks["out_proj"] = std::make_shared<Linear>(config.width, config.out_dim);
            if (config.has_skip) {
                blocks["skip"] = std::make_shared<Linear>(config.in_dim, config.out_dim);
            }
            if (config.has_vision_proj) {
                blocks["vision_proj"] = std::make_shared<Linear>(config.vis_dim, config.in_dim, /*bias=*/false);
            }
            if (config.has_vis_in) {
                blocks["vis_in"] = std::make_shared<Linear>(config.vis_dim, config.width, /*bias=*/false);
            }
        }

        // vision ext: raw mmproj image embeds [vis_dim, n, N] -> the student's
        // input embedding space [in_dim, n, N]
        ggml_tensor* project_vision(GGMLRunnerContext* ctx, ggml_tensor* embeds) {
            GGML_ASSERT(config.has_vision_proj);
            auto vision_proj = std::dynamic_pointer_cast<Linear>(blocks["vision_proj"]);
            return vision_proj->forward(ctx, embeds);
        }

        // token-aligned path: qwen_hidden [in_dim, L, N] (student final-norm
        // hidden states) -> [out_dim, L, N]; kv_mask: additive [L, 1] or null;
        // vis: dense raw mmproj embeds [vis_dim, L, N] (zeros at text
        // positions) or null.
        ggml_tensor* forward_token_aligned(GGMLRunnerContext* ctx, ggml_tensor* qwen_hidden, ggml_tensor* kv_mask, ggml_tensor* vis = nullptr) {
            auto in_proj  = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto ln_out   = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_out"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            auto x = in_proj->forward(ctx, qwen_hidden);  // [width, L, N]
            if (vis != nullptr && config.has_vis_in) {
                auto vis_in = std::dynamic_pointer_cast<Linear>(blocks["vis_in"]);
                x           = ggml_add(ctx->ggml_ctx, x, vis_in->forward(ctx, vis));
            }
            for (int64_t i = 0; i < config.depth; ++i) {
                auto block = std::dynamic_pointer_cast<TokenBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, kv_mask);
            }
            x = out_proj->forward(ctx, ln_out->forward(ctx, x));
            if (config.has_skip) {
                auto skip = std::dynamic_pointer_cast<Linear>(blocks["skip"]);
                x         = ggml_add(ctx->ggml_ctx, x, skip->forward(ctx, qwen_hidden));
            }
            return x;
        }

        // qwen_hidden: [in_dim, L_k, N] (post final-norm LLM hidden states)
        // t5_ids:      [num_queries, N] i32, EOS-appended and pad(0)-padded
        //              (ignored in the token-aligned variant, may be null)
        // kv_mask:     additive [L_k, 1] over the Qwen tokens, or nullptr
        // returns:     [out_dim, num_queries, N]  (token-aligned: [out_dim, L_k, N])
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* qwen_hidden, ggml_tensor* t5_ids, ggml_tensor* kv_mask, ggml_tensor* vis = nullptr) {
            if (config.token_aligned) {
                return forward_token_aligned(ctx, qwen_hidden, kv_mask, vis);
            }
            auto in_proj  = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto ln_out   = std::dynamic_pointer_cast<LayerNorm>(blocks["ln_out"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            auto kv = in_proj->forward(ctx, qwen_hidden);  // [width, L_k, N]
            ggml_tensor* q;
            if (config.embed_rank > 0) {
                auto t5_embed_a = std::dynamic_pointer_cast<Embedding>(blocks["t5_embed_a"]);
                auto t5_embed_b = std::dynamic_pointer_cast<Linear>(blocks["t5_embed_b"]);
                q = t5_embed_a->forward(ctx, t5_ids);  // [rank, num_queries, N]
                q = t5_embed_b->forward(ctx, q);       // [width, num_queries, N]
            } else {
                auto t5_embed = std::dynamic_pointer_cast<Embedding>(blocks["t5_embed"]);
                q = t5_embed->forward(ctx, t5_ids);  // [width, num_queries, N]
            }
            q = ggml_add(ctx->ggml_ctx, q, params["query"]);
            for (int64_t i = 0; i < config.depth; ++i) {
                auto block = std::dynamic_pointer_cast<Block>(blocks["blocks." + std::to_string(i)]);
                q          = block->forward(ctx, q, kv, kv_mask);
            }
            return out_proj->forward(ctx, ln_out->forward(ctx, q));
        }
    };

    struct AdapterRunner : public GGMLRunner {
        AdapterConfig config;
        Adapter model;

        AdapterRunner(ggml_backend_t backend,
                      const String2TensorStorage& tensor_storage_map,
                      const std::string prefix,
                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              config(AdapterConfig::detect_from_weights(tensor_storage_map, prefix)),
              model(config) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "llm_adapter";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& qwen_hidden_tensor,
                                 const sd::Tensor<int32_t>& t5_ids_tensor,
                                 const sd::Tensor<float>& kv_mask_tensor,
                                 const sd::Tensor<float>& vis_tensor) {
            ggml_cgraph* gf           = new_graph_custom(ADAPTER_GRAPH_SIZE);
            ggml_tensor* qwen_hidden  = make_input(qwen_hidden_tensor);
            ggml_tensor* t5_ids       = config.token_aligned ? nullptr : make_input(t5_ids_tensor);
            ggml_tensor* kv_mask      = make_optional_input(kv_mask_tensor);
            ggml_tensor* vis          = make_optional_input(vis_tensor);
            if (kv_mask != nullptr) {
                // ggml_ext_attention_ext adds the mask to the [L_k, L_q] scores and
                // broadcasts a singleton query dimension.
                kv_mask = ggml_reshape_2d(compute_ctx, kv_mask, kv_mask->ne[0], 1);
            }

            auto runner_ctx  = get_context();
            ggml_tensor* out = model.forward(&runner_ctx, qwen_hidden, t5_ids, kv_mask, vis);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(const int n_threads,
                                  const sd::Tensor<float>& qwen_hidden,
                                  const sd::Tensor<int32_t>& t5_ids,
                                  const sd::Tensor<float>& kv_mask = {},
                                  const sd::Tensor<float>& vis     = {},
                                  bool auto_free                   = true,
                                  bool free_compute_buffer         = true,
                                  bool free_compute_params         = true) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(qwen_hidden, t5_ids, kv_mask, vis);
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, auto_free, free_compute_buffer, free_compute_params), 3);
        }

        // vision ext: map raw mmproj image embeds [vis_dim, n] (or [vis_dim,
        // n, 1]) into the student's input embedding space [in_dim, n].
        sd::Tensor<float> project_vision(const int n_threads,
                                         const sd::Tensor<float>& embeds,
                                         bool auto_free           = true,
                                         bool free_compute_buffer = true,
                                         bool free_compute_params = true) {
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf     = new_graph_custom(ADAPTER_GRAPH_SIZE);
                ggml_tensor* in     = make_input(embeds);
                auto runner_ctx     = get_context();
                ggml_tensor* out    = model.project_vision(&runner_ctx, in);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, n_threads, auto_free, free_compute_buffer, free_compute_params), embeds.dim());
        }
    };

}  // namespace LLMAdapter

#endif  // __SD_MODEL_TE_LLM_ADAPTER_HPP__
