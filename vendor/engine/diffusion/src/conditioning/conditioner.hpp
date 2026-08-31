#ifndef __SD_CONDITIONING_CONDITIONER_HPP__
#define __SD_CONDITIONING_CONDITIONER_HPP__

#include <cmath>
#include <limits>
#include <optional>

#include "core/tensor_ggml.hpp"
#include "model/diffusion/minimax_music3.hpp"
#include "model/diffusion/model.hpp"
#include "model/te/clip.hpp"
#include "model/te/llm.hpp"
#include "model/te/llm_adapter.hpp"
#include "model/te/t5.hpp"
#include "model_loader.h"

struct SDCondition {
    sd::Tensor<float> c_crossattn;
    sd::Tensor<float> c_vector;
    sd::Tensor<float> c_concat;
    sd::Tensor<int32_t> c_t5_ids;
    sd::Tensor<float> c_t5_weights;
    sd::Tensor<int32_t> c_input_ids;
    sd::Tensor<int32_t> c_position_ids;
    sd::Tensor<int32_t> c_token_types;
    sd::Tensor<int32_t> c_vinput_mask;
    std::vector<std::pair<int, sd::Tensor<float>>> c_image_embeds;
    std::vector<sd::Tensor<float>> c_ref_images;

    std::vector<sd::Tensor<float>> extra_c_crossattns;

    SDCondition() = default;

    SDCondition(sd::Tensor<float> c_crossattn,
                sd::Tensor<float> c_vector,
                sd::Tensor<float> c_concat)
        : c_crossattn(std::move(c_crossattn)), c_vector(std::move(c_vector)), c_concat(std::move(c_concat)) {}

    bool empty() const {
        if (!c_crossattn.empty() || !c_vector.empty() || !c_concat.empty() ||
            !c_t5_ids.empty() || !c_t5_weights.empty() ||
            !c_input_ids.empty() || !c_position_ids.empty() ||
            !c_token_types.empty() || !c_vinput_mask.empty()) {
            return false;
        }

        for (const auto& image_embed : c_image_embeds) {
            if (!image_embed.second.empty()) {
                return false;
            }
        }

        for (const auto& tensor : c_ref_images) {
            if (!tensor.empty()) {
                return false;
            }
        }

        for (const auto& tensor : extra_c_crossattns) {
            if (!tensor.empty()) {
                return false;
            }
        }

        return true;
    }
};

static inline sd::Tensor<float> apply_token_weights(sd::Tensor<float> hidden_states,
                                                    const std::vector<float>& weights) {
    if (hidden_states.empty()) {
        return hidden_states;
    }

    bool all_one = true;
    for (float weight : weights) {
        if (weight != 1.0f) {
            all_one = false;
            break;
        }
    }
    if (all_one) {
        return hidden_states;
    }

    if (hidden_states.dim() == 1) {
        hidden_states.unsqueeze_(1);
    }

    GGML_ASSERT(static_cast<size_t>(hidden_states.shape()[1]) == weights.size());

    float original_mean = hidden_states.mean();
    auto chunk_weights  = sd::Tensor<float>::from_vector(weights);
    chunk_weights.reshape_({1, static_cast<int64_t>(weights.size())});
    hidden_states *= chunk_weights;
    float new_mean = hidden_states.mean();
    if (std::isfinite(original_mean) && std::isfinite(new_mean) && new_mean != 0.0f) {
        hidden_states *= (original_mean / new_mean);
    }

    return hidden_states;
}

struct ConditionerParams {
    std::string text;
    int clip_skip                                    = -1;
    int width                                        = -1;
    int height                                       = -1;
    bool zero_out_masked                             = false;
    const std::vector<sd::Tensor<float>>* ref_images = nullptr;  // for qwen image edit
    RefImageParams ref_image_params;
};

struct Conditioner {
    virtual ~Conditioner() = default;

public:
    virtual SDCondition get_learned_condition(int n_threads,
                                              const ConditionerParams& conditioner_params) = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors)           = 0;
    virtual void set_max_graph_vram_bytes(size_t max_vram_bytes) {}
    virtual void set_stream_layers_enabled(bool enabled) {}
    virtual void set_flash_attention_enabled(bool enabled) = 0;
    virtual void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) {}
    virtual void runner_done() {}
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public Conditioner {
    SDVersion version = VERSION_SD1;
    CLIPTokenizer tokenizer;
    std::shared_ptr<CLIPTextModelRunner> text_model;
    std::shared_ptr<CLIPTextModelRunner> text_model2;

    std::map<std::string, std::string> embedding_map;
    int32_t num_custom_embeddings   = 0;
    int32_t num_custom_embeddings_2 = 0;
    std::vector<uint8_t> token_embed_custom;
    std::map<std::string, std::pair<int, int>> embedding_pos_map;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend,
                                      const String2TensorStorage& tensor_storage_map,
                                      const std::map<std::string, std::string>& orig_embedding_map,
                                      SDVersion version                                   = VERSION_SD1,
                                      std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : version(version), tokenizer(sd_version_is_sd2(version) ? 0 : 49407) {
        for (const auto& kv : orig_embedding_map) {
            std::string name    = normalize_embedding_name(kv.first);
            embedding_map[name] = kv.second;
            tokenizer.add_special_token(name);
        }
        bool force_clip_f32 = !embedding_map.empty();
        if (sd_version_is_sd1(version)) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, true, force_clip_f32, weight_manager);
        } else if (sd_version_is_sd2(version)) {
            text_model = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "cond_stage_model.transformer.text_model", OPEN_CLIP_VIT_H_14, true, force_clip_f32, weight_manager);
        } else if (sd_version_is_sdxl(version)) {
            text_model  = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "cond_stage_model.transformer.text_model", OPENAI_CLIP_VIT_L_14, false, force_clip_f32, weight_manager);
            text_model2 = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "cond_stage_model.1.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false, force_clip_f32, weight_manager);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        text_model->get_param_tensors(tensors, "cond_stage_model.transformer.text_model");
        if (sd_version_is_sdxl(version)) {
            text_model2->get_param_tensors(tensors, "cond_stage_model.1.transformer.text_model");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        text_model->set_max_graph_vram_bytes(max_vram_bytes);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        text_model->set_stream_layers_enabled(enabled);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        text_model->set_flash_attention_enabled(enabled);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        text_model->set_weight_adapter(adapter);
        if (sd_version_is_sdxl(version)) {
            text_model2->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        text_model->runner_done();
        if (sd_version_is_sdxl(version)) {
            text_model2->runner_done();
        }
    }

    bool load_embedding(std::string embd_name, std::string embd_path, std::vector<int32_t>& bpe_tokens) {
        ModelLoader model_loader;
        if (!model_loader.init_from_file_and_convert_name(embd_path)) {
            LOG_ERROR("embedding '%s' failed", embd_name.c_str());
            return false;
        }
        auto iter = embedding_pos_map.find(embd_name);
        if (iter != embedding_pos_map.end()) {
            LOG_DEBUG("embedding already read in: %s", embd_name.c_str());
            for (int i = iter->second.first; i < iter->second.second; i++) {
                bpe_tokens.push_back(text_model->model.vocab_size + i);
            }
            return true;
        }
        ggml_init_params params;
        params.mem_size        = 100 * 1024 * 1024;  // max for custom embeddings 100 MB
        params.mem_buffer      = nullptr;
        params.no_alloc        = false;
        ggml_context* embd_ctx = ggml_init(params);
        ggml_tensor* embd      = nullptr;
        ggml_tensor* embd2     = nullptr;
        auto on_load           = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != text_model->model.hidden_size) {
                if (text_model2) {
                    if (tensor_storage.ne[0] == text_model2->model.hidden_size) {
                        embd2       = ggml_new_tensor_2d(embd_ctx, tensor_storage.type, text_model2->model.hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                        *dst_tensor = embd2;
                    } else {
                        LOG_DEBUG("embedding wrong hidden size, got %i, expected %i or %i", tensor_storage.ne[0], text_model->model.hidden_size, text_model2->model.hidden_size);
                        return false;
                    }
                } else {
                    LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], text_model->model.hidden_size);
                    return false;
                }
            } else {
                embd        = ggml_new_tensor_2d(embd_ctx, tensor_storage.type, text_model->model.hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
                *dst_tensor = embd;
            }
            return true;
        };
        model_loader.set_n_threads(1);
        model_loader.load_tensors(on_load);
        int pos_start = num_custom_embeddings;
        if (embd) {
            int64_t hidden_size = text_model->model.hidden_size;
            token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd));
            memcpy((void*)(token_embed_custom.data() + num_custom_embeddings * hidden_size * ggml_type_size(embd->type)),
                   embd->data,
                   ggml_nbytes(embd));
            for (int i = 0; i < embd->ne[1]; i++) {
                bpe_tokens.push_back(text_model->model.vocab_size + num_custom_embeddings);
                // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
                num_custom_embeddings++;
            }
            LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        }
        if (embd2) {
            int64_t hidden_size = text_model2->model.hidden_size;
            token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd2));
            memcpy((void*)(token_embed_custom.data() + num_custom_embeddings_2 * hidden_size * ggml_type_size(embd2->type)),
                   embd2->data,
                   ggml_nbytes(embd2));
            for (int i = 0; i < embd2->ne[1]; i++) {
                bpe_tokens.push_back(text_model2->model.vocab_size + num_custom_embeddings_2);
                // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
                num_custom_embeddings_2++;
            }
            LOG_DEBUG("embedding '%s' applied, custom embeddings: %i (text model 2)", embd_name.c_str(), num_custom_embeddings_2);
        }
        int pos_end = num_custom_embeddings;
        if (pos_end == pos_start) {
            return false;
        }
        embedding_pos_map[embd_name] = std::pair{pos_start, pos_end};
        return true;
    }

    static std::string normalize_embedding_name(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        return name;
    }

    bool append_embedding_tokens(std::string str, std::vector<int32_t>& bpe_tokens) {
        std::string name = normalize_embedding_name(std::move(str));
        auto iter        = embedding_map.find(name);
        if (iter == embedding_map.end()) {
            return false;
        }
        return load_embedding(name, iter->second, bpe_tokens);
    }

    std::vector<int> convert_token_to_id(std::string text) {
        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return append_embedding_tokens(str, bpe_tokens);
        };
        std::vector<int> curr_tokens = tokenizer.encode(text, on_new_token_cb);
        return curr_tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        return tokenizer.decode(tokens);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t min_length          = 0,
                                                             size_t max_length          = 0,
                                                             bool allow_overflow_expand = true) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return append_embedding_tokens(str, bpe_tokens);
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;

            if (curr_text == "BREAK" && curr_weight == -1.0f) {
                // Pad token array up to chunk size at this point.
                // TODO: This is a hardcoded chunk_len, like in stable-diffusion.cpp, make it a parameter for the future?
                // Also, this is 75 instead of 77 to leave room for BOS and EOS tokens.
                size_t current_size = tokens.size();
                size_t padding_size = (75 - (current_size % 75)) % 75;  // Ensure no negative padding

                if (padding_size > 0) {
                    LOG_DEBUG("BREAK token encountered, padding current chunk by %zu tokens.", padding_size);
                    tokens.insert(tokens.end(), padding_size, tokenizer.EOS_TOKEN_ID);
                    weights.insert(weights.end(), padding_size, 1.0f);
                }
                continue;  // Skip to the next item after handling BREAK
            }

            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer.pad_tokens(tokens, &weights, nullptr, min_length, max_length, allow_overflow_expand);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {tokens, weights};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<int>& tokens,
                                             std::vector<float>& weights,
                                             int clip_skip,
                                             int width,
                                             int height,
                                             bool zero_out_masked = false) {
        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> hidden_states;  // [n_token, hidden_size] or [n_token, hidden_size + hidden_size2]
        sd::Tensor<float> pooled;

        if (clip_skip <= 0) {
            clip_skip = (sd_version_is_sd2(version) || sd_version_is_sdxl(version)) ? 2 : 1;
        }

        size_t chunk_len   = 77;
        size_t chunk_count = tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            std::vector<int> chunk_tokens(tokens.begin() + chunk_idx * chunk_len,
                                          tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(weights.begin() + chunk_idx * chunk_len,
                                             weights.begin() + (chunk_idx + 1) * chunk_len);

            sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
            sd::Tensor<int32_t> input_ids2;
            size_t max_token_idx = 0;
            if (sd_version_is_sdxl(version)) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), tokenizer.EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = sd::Tensor<int32_t>({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            {
                auto chunk_hidden_states = text_model->compute(n_threads,
                                                               input_ids,
                                                               num_custom_embeddings,
                                                               token_embed_custom.data(),
                                                               max_token_idx,
                                                               false,
                                                               clip_skip,
                                                               false,
                                                               true,
                                                               true);
                GGML_ASSERT(!chunk_hidden_states.empty());
                if (sd_version_is_sdxl(version)) {
                    auto chunk_hidden_states2 = text_model2->compute(n_threads,
                                                                     input_ids2,
                                                                     num_custom_embeddings,
                                                                     token_embed_custom.data(),
                                                                     max_token_idx,
                                                                     false,
                                                                     clip_skip,
                                                                     false,
                                                                     true,
                                                                     true);
                    GGML_ASSERT(!chunk_hidden_states2.empty());
                    chunk_hidden_states = sd::ops::concat(chunk_hidden_states, chunk_hidden_states2, 0);

                    if (chunk_idx == 0) {
                        pooled = text_model2->compute(n_threads,
                                                      input_ids2,
                                                      num_custom_embeddings,
                                                      token_embed_custom.data(),
                                                      max_token_idx,
                                                      true,
                                                      clip_skip,
                                                      false,
                                                      true,
                                                      true);
                        GGML_ASSERT(!pooled.empty());
                    }
                }
                int64_t t1 = ggml_time_ms();
                LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

                chunk_hidden_states = apply_token_weights(std::move(chunk_hidden_states), chunk_weights);

                if (zero_out_masked) {
                    chunk_hidden_states.fill_(0.0f);
                }
                if (!hidden_states.empty()) {
                    hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
                } else {
                    hidden_states = std::move(chunk_hidden_states);
                }
            }
        }

        sd::Tensor<float> vec;
        if (sd_version_is_sdxl(version)) {
            int out_dim         = 256;
            int adm_in_channels = 2816;
            GGML_ASSERT(!pooled.empty());
            vec = sd::Tensor<float>({adm_in_channels});
            vec.fill_(0.0f);
            size_t offset = 0;
            std::copy(pooled.values().begin(), pooled.values().end(), vec.values().begin());
            offset += pooled.values().size();

            auto append_embedding = [&](const std::vector<float>& timesteps) {
                sd::Tensor<float> embedding;
                set_timestep_embedding(timesteps, &embedding, out_dim);
                std::copy(embedding.values().begin(), embedding.values().end(), vec.values().begin() + static_cast<int64_t>(offset));
                offset += embedding.values().size();
            };

            append_embedding({static_cast<float>(height), static_cast<float>(width)});
            append_embedding({0.0f, 0.0f});
            append_embedding({static_cast<float>(height), static_cast<float>(width)});
            GGML_ASSERT(offset == vec.values().size());
        }
        SDCondition result;
        if (!hidden_states.empty()) {
            result.c_crossattn = std::move(hidden_states);
        }

        if (!vec.empty()) {
            result.c_vector = std::move(vec);
        }
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights     = tokenize(conditioner_params.text, text_model->model.n_token, text_model->model.n_token, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(n_threads,
                                            tokens,
                                            weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.width,
                                            conditioner_params.height,
                                            conditioner_params.zero_out_masked);
    }
};

struct FrozenCLIPVisionEmbedder : public GGMLRunner {
    CLIPVisionModelProjection vision_model;
    std::string weight_prefix = "cond_stage_model.transformer";

    FrozenCLIPVisionEmbedder(ggml_backend_t backend,
                             const String2TensorStorage& tensor_storage_map      = {},
                             std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager) {
        bool proj_in = false;
        for (const auto& [name, tensor_storage] : tensor_storage_map) {
            if (!starts_with(name, weight_prefix)) {
                continue;
            }
            if (contains(name, "self_attn.in_proj")) {
                proj_in = true;
                break;
            }
        }
        vision_model = CLIPVisionModelProjection(OPEN_CLIP_VIT_H_14, false, proj_in);
        vision_model.init(params_ctx, tensor_storage_map, weight_prefix);
    }

    std::string get_desc() override {
        return "clip_vision";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        vision_model.get_param_tensors(tensors, weight_prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& pixel_values_tensor, bool return_pooled, int clip_skip) {
        ggml_cgraph* gf           = ggml_new_graph(compute_ctx);
        ggml_tensor* pixel_values = make_input(pixel_values_tensor);

        auto runner_ctx = get_context();

        ggml_tensor* hidden_states = vision_model.forward(&runner_ctx, pixel_values, return_pooled, clip_skip);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    sd::Tensor<float> compute(const int n_threads,
                              const sd::Tensor<float>& pixel_values,
                              bool return_pooled,
                              int clip_skip) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(pixel_values, return_pooled, clip_skip);
        };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, true, true, true));
    }
};

struct SD3CLIPEmbedder : public Conditioner {
    CLIPTokenizer clip_l_tokenizer;
    CLIPTokenizer clip_g_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<CLIPTextModelRunner> clip_l;
    std::shared_ptr<CLIPTextModelRunner> clip_g;
    std::shared_ptr<T5Runner> t5;

    SD3CLIPEmbedder(ggml_backend_t backend,
                    const String2TensorStorage& tensor_storage_map      = {},
                    std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : clip_g_tokenizer(0) {
        bool use_clip_l = false;
        bool use_clip_g = false;
        bool use_t5     = false;
        for (auto pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.clip_l") != std::string::npos) {
                use_clip_l = true;
            } else if (pair.first.find("text_encoders.clip_g") != std::string::npos) {
                use_clip_g = true;
            } else if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            }
        }
        if (!use_clip_l && !use_clip_g && !use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        }
        if (use_clip_l) {
            clip_l = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "text_encoders.clip_l.transformer.text_model", OPENAI_CLIP_VIT_L_14, false, false, weight_manager);
        }
        if (use_clip_g) {
            clip_g = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "text_encoders.clip_g.transformer.text_model", OPEN_CLIP_VIT_BIGG_14, false, false, weight_manager);
        }
        if (use_t5) {
            t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", false, weight_manager);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (clip_l) {
            clip_l->get_param_tensors(tensors, "text_encoders.clip_l.transformer.text_model");
        }
        if (clip_g) {
            clip_g->get_param_tensors(tensors, "text_encoders.clip_g.transformer.text_model");
        }
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (clip_l) {
            clip_l->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (clip_g) {
            clip_g->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_stream_layers_enabled(enabled);
        }
        if (clip_g) {
            clip_g->set_stream_layers_enabled(enabled);
        }
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_flash_attention_enabled(enabled);
        }
        if (clip_g) {
            clip_g->set_flash_attention_enabled(enabled);
        }
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (clip_l) {
            clip_l->set_weight_adapter(adapter);
        }
        if (clip_g) {
            clip_g->set_weight_adapter(adapter);
        }
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (clip_l) {
            clip_l->runner_done();
        }
        if (clip_g) {
            clip_g->runner_done();
        }
        if (t5) {
            t5->runner_done();
        }
    }

    std::vector<std::pair<std::vector<int>, std::vector<float>>> tokenize(std::string text,
                                                                          size_t min_length          = 0,
                                                                          size_t max_length          = 0,
                                                                          bool allow_overflow_expand = true) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> clip_l_tokens;
        std::vector<float> clip_l_weights;
        std::vector<int> clip_g_tokens;
        std::vector<float> clip_g_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            if (clip_l) {
                std::vector<int> curr_tokens = clip_l_tokenizer.encode(curr_text, on_new_token_cb);
                clip_l_tokens.insert(clip_l_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_l_weights.insert(clip_l_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (clip_g) {
                std::vector<int> curr_tokens = clip_g_tokenizer.encode(curr_text, on_new_token_cb);
                clip_g_tokens.insert(clip_g_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_g_weights.insert(clip_g_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (t5) {
                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, &clip_l_weights, nullptr, min_length, max_length, allow_overflow_expand);
        }
        if (clip_g) {
            clip_g_tokenizer.pad_tokens(clip_g_tokens, &clip_g_weights, nullptr, min_length, max_length, allow_overflow_expand);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, nullptr, min_length, max_length, true);
        }

        // for (int i = 0; i < clip_l_tokens.size(); i++) {
        //     std::cout << clip_l_tokens[i] << ":" << clip_l_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < clip_g_tokens.size(); i++) {
        //     std::cout << clip_g_tokens[i] << ":" << clip_g_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < t5_tokens.size(); i++) {
        //     std::cout << t5_tokens[i] << ":" << t5_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {{clip_l_tokens, clip_l_weights}, {clip_g_tokens, clip_g_weights}, {t5_tokens, t5_weights}};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<std::pair<std::vector<int>, std::vector<float>>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        auto& clip_l_tokens  = token_and_weights[0].first;
        auto& clip_l_weights = token_and_weights[0].second;
        auto& clip_g_tokens  = token_and_weights[1].first;
        auto& clip_g_weights = token_and_weights[1].second;
        auto& t5_tokens      = token_and_weights[2].first;
        auto& t5_weights     = token_and_weights[2].second;

        if (clip_skip <= 0) {
            clip_skip = 2;
        }

        size_t chunk_len = 77;
        int64_t t0       = ggml_time_ms();
        sd::Tensor<float> hidden_states;
        sd::Tensor<float> pooled;

        size_t chunk_count = std::max(std::max(clip_l_tokens.size(), clip_g_tokens.size()), t5_tokens.size()) / chunk_len;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            sd::Tensor<float> chunk_hidden_states_l;
            sd::Tensor<float> pooled_l;
            if (clip_l) {
                std::vector<int> chunk_tokens(clip_l_tokens.begin() + chunk_idx * chunk_len,
                                              clip_l_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_l_weights.begin() + chunk_idx * chunk_len,
                                                 clip_l_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                size_t max_token_idx = 0;

                chunk_hidden_states_l = clip_l->compute(n_threads,
                                                        input_ids,
                                                        0,
                                                        nullptr,
                                                        max_token_idx,
                                                        false,
                                                        clip_skip,
                                                        false,
                                                        true,
                                                        true);
                GGML_ASSERT(!chunk_hidden_states_l.empty());
                chunk_hidden_states_l = ::apply_token_weights(std::move(chunk_hidden_states_l), chunk_weights);

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    pooled_l      = clip_l->compute(n_threads,
                                                    input_ids,
                                                    0,
                                                    nullptr,
                                                    max_token_idx,
                                                    true,
                                                    clip_skip,
                                                    false,
                                                    true,
                                                    true);
                    GGML_ASSERT(!pooled_l.empty());
                }
            } else {
                chunk_hidden_states_l = sd::Tensor<float>::zeros({768, static_cast<int64_t>(chunk_len), 1});
                if (chunk_idx == 0) {
                    pooled_l = sd::Tensor<float>::zeros({768, 1});
                }
            }

            // clip_g
            sd::Tensor<float> chunk_hidden_states_g;
            sd::Tensor<float> pooled_g;
            if (clip_g) {
                std::vector<int> chunk_tokens(clip_g_tokens.begin() + chunk_idx * chunk_len,
                                              clip_g_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(clip_g_weights.begin() + chunk_idx * chunk_len,
                                                 clip_g_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                size_t max_token_idx = 0;

                chunk_hidden_states_g = clip_g->compute(n_threads,
                                                        input_ids,
                                                        0,
                                                        nullptr,
                                                        max_token_idx,
                                                        false,
                                                        clip_skip,
                                                        false,
                                                        true,
                                                        true);
                GGML_ASSERT(!chunk_hidden_states_g.empty());
                chunk_hidden_states_g = ::apply_token_weights(std::move(chunk_hidden_states_g), chunk_weights);

                if (chunk_idx == 0) {
                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_g_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);
                    pooled_g      = clip_g->compute(n_threads,
                                                    input_ids,
                                                    0,
                                                    nullptr,
                                                    max_token_idx,
                                                    true,
                                                    clip_skip,
                                                    false,
                                                    true,
                                                    true);
                    GGML_ASSERT(!pooled_g.empty());
                }
            } else {
                chunk_hidden_states_g = sd::Tensor<float>::zeros({1280, static_cast<int64_t>(chunk_len), 1});
                if (chunk_idx == 0) {
                    pooled_g = sd::Tensor<float>::zeros({1280, 1});
                }
            }

            // t5
            sd::Tensor<float> chunk_hidden_states_t5;
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);

                chunk_hidden_states_t5 = t5->compute(n_threads,
                                                     input_ids,
                                                     sd::Tensor<float>(),
                                                     false,
                                                     true,
                                                     true);
                GGML_ASSERT(!chunk_hidden_states_t5.empty());
                chunk_hidden_states_t5 = ::apply_token_weights(std::move(chunk_hidden_states_t5), chunk_weights);
            } else {
                chunk_hidden_states_t5 = sd::Tensor<float>::zeros({4096, static_cast<int64_t>(chunk_len), 1});
            }

            sd::Tensor<float> chunk_hidden_states_lg = sd::ops::concat(chunk_hidden_states_l, chunk_hidden_states_g, 0);
            if (chunk_hidden_states_lg.shape()[0] < 4096) {
                auto pad_shape         = chunk_hidden_states_lg.shape();
                pad_shape[0]           = 4096 - chunk_hidden_states_lg.shape()[0];
                chunk_hidden_states_lg = sd::ops::concat(chunk_hidden_states_lg,
                                                         sd::Tensor<float>::zeros(pad_shape),
                                                         0);
            }

            sd::Tensor<float> chunk_hidden_states = sd::ops::concat(chunk_hidden_states_lg,
                                                                    chunk_hidden_states_t5,
                                                                    1);  // [n_token*2, 4096]

            if (chunk_idx == 0) {
                pooled = sd::ops::concat(pooled_l, pooled_g, 0);  // [768 + 1280]
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (zero_out_masked) {
                chunk_hidden_states.fill_(0.0f);
            }

            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(pooled);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, 77, 77, true);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct FluxCLIPEmbedder : public Conditioner {
    CLIPTokenizer clip_l_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<CLIPTextModelRunner> clip_l;
    std::shared_ptr<T5Runner> t5;
    size_t chunk_len = 256;

    FluxCLIPEmbedder(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map      = {},
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr) {
        bool use_clip_l = false;
        bool use_t5     = false;
        for (auto pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.clip_l") != std::string::npos) {
                use_clip_l = true;
            } else if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            }
        }

        if (!use_clip_l && !use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        }

        if (use_clip_l) {
            clip_l = std::make_shared<CLIPTextModelRunner>(backend, tensor_storage_map, "text_encoders.clip_l.transformer.text_model", OPENAI_CLIP_VIT_L_14, true, false, weight_manager);
        } else {
            LOG_WARN("clip_l text encoder not found! Prompt adherence might be degraded.");
        }
        if (use_t5) {
            t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", false, weight_manager);
        } else {
            LOG_WARN("t5xxl text encoder not found! Prompt adherence might be degraded.");
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (clip_l) {
            clip_l->get_param_tensors(tensors, "text_encoders.clip_l.transformer.text_model");
        }
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (clip_l) {
            clip_l->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_stream_layers_enabled(enabled);
        }
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (clip_l) {
            clip_l->set_flash_attention_enabled(enabled);
        }
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (clip_l) {
            clip_l->set_weight_adapter(adapter);
        }
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (clip_l) {
            clip_l->runner_done();
        }
        if (t5) {
            t5->runner_done();
        }
    }

    std::vector<std::pair<std::vector<int>, std::vector<float>>> tokenize(std::string text,
                                                                          size_t min_length = 0,
                                                                          size_t max_length = 0) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> clip_l_tokens;
        std::vector<float> clip_l_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            if (clip_l) {
                std::vector<int> curr_tokens = clip_l_tokenizer.encode(curr_text, on_new_token_cb);
                clip_l_tokens.insert(clip_l_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                clip_l_weights.insert(clip_l_weights.end(), curr_tokens.size(), curr_weight);
            }
            if (t5) {
                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }
        }

        if (clip_l) {
            clip_l_tokenizer.pad_tokens(clip_l_tokens, &clip_l_weights, nullptr, 77, 77, true);
        }
        if (t5) {
            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, nullptr, min_length, max_length, true);
        }

        // for (int i = 0; i < clip_l_tokens.size(); i++) {
        //     std::cout << clip_l_tokens[i] << ":" << clip_l_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < t5_tokens.size(); i++) {
        //     std::cout << t5_tokens[i] << ":" << t5_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        return {{clip_l_tokens, clip_l_weights}, {t5_tokens, t5_weights}};
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::vector<std::pair<std::vector<int>, std::vector<float>>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false) {
        auto& clip_l_tokens  = token_and_weights[0].first;
        auto& clip_l_weights = token_and_weights[0].second;
        auto& t5_tokens      = token_and_weights[1].first;
        auto& t5_weights     = token_and_weights[1].second;

        if (clip_skip <= 0) {
            clip_skip = 2;
        }

        int64_t t0 = ggml_time_ms();
        sd::Tensor<float> hidden_states;  // [N, n_token, 4096]
        sd::Tensor<float> pooled;         // [768,]

        size_t chunk_count = std::max(clip_l_tokens.size() > 0 ? chunk_len : 0, t5_tokens.size()) / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // clip_l
            if (chunk_idx == 0) {
                if (clip_l) {
                    size_t chunk_len_l = 77;
                    std::vector<int> chunk_tokens(clip_l_tokens.begin(),
                                                  clip_l_tokens.begin() + chunk_len_l);
                    std::vector<float> chunk_weights(clip_l_weights.begin(),
                                                     clip_l_weights.begin() + chunk_len_l);

                    sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                    size_t max_token_idx = 0;

                    auto it       = std::find(chunk_tokens.begin(), chunk_tokens.end(), clip_l_tokenizer.EOS_TOKEN_ID);
                    max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                    pooled = clip_l->compute(n_threads,
                                             input_ids,
                                             0,
                                             nullptr,
                                             max_token_idx,
                                             true,
                                             clip_skip,
                                             false,
                                             true,
                                             true);
                    GGML_ASSERT(!pooled.empty());
                } else {
                    pooled = sd::Tensor<float>::zeros({768});
                }
            }

            // t5
            sd::Tensor<float> chunk_hidden_states;
            if (t5) {
                std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                              t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
                std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                                 t5_weights.begin() + (chunk_idx + 1) * chunk_len);

                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
                chunk_hidden_states = t5->compute(n_threads,
                                                  input_ids,
                                                  sd::Tensor<float>(),
                                                  false,
                                                  true,
                                                  true);
                GGML_ASSERT(!chunk_hidden_states.empty());
                chunk_hidden_states = ::apply_token_weights(std::move(chunk_hidden_states), chunk_weights);
                if (zero_out_masked) {
                    chunk_hidden_states.fill_(0.0f);
                }
            } else {
                chunk_hidden_states = sd::Tensor<float>::zeros({4096, static_cast<int64_t>(chunk_len)});
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(pooled);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, chunk_len);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked);
    }
};

struct T5CLIPEmbedder : public Conditioner {
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<T5Runner> t5;
    std::shared_ptr<BPETokenizer> qwen_tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;
    std::shared_ptr<LLMAdapter::AdapterRunner> llm_adapter;
    size_t chunk_len = 512;
    bool use_mask    = false;
    int mask_pad     = 0;
    bool is_umt5     = false;

    T5CLIPEmbedder(ggml_backend_t backend,
                   const String2TensorStorage& tensor_storage_map      = {},
                   bool use_mask                                       = false,
                   int mask_pad                                        = 0,
                   bool is_umt5                                        = false,
                   std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : use_mask(use_mask), mask_pad(mask_pad), t5_tokenizer(is_umt5) {
        bool use_t5      = false;
        bool has_llm     = false;
        bool has_adapter = false;
        for (const auto& pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            } else if (starts_with(pair.first, "text_encoders.llm_adapter.")) {
                has_adapter = true;
            } else if (starts_with(pair.first, "text_encoders.llm.")) {
                has_llm = true;
            }
        }
        if (has_llm && has_adapter) {
            // trainer2/: the T5/UMT5 encoder is replaced by Qwen3 hidden
            // states resampled into its embedding space; the sentencepiece
            // tokenizer stays, because the adapter's queries are seeded with
            // its token ids and the padding/mask path is unchanged.
            LOG_INFO("T5 conditioning via Qwen3 + bridge adapter%s", use_t5 ? " (t5xxl ignored)" : "");
            qwen_tokenizer = std::make_shared<Qwen2Tokenizer>();
            llm            = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                                   backend,
                                                   tensor_storage_map,
                                                   "text_encoders.llm",
                                                   false,
                                                   weight_manager);
            llm_adapter    = std::make_shared<LLMAdapter::AdapterRunner>(backend,
                                                                      tensor_storage_map,
                                                                      "text_encoders.llm_adapter.adapter",
                                                                      weight_manager);
            if (static_cast<size_t>(llm_adapter->config.num_queries) != chunk_len) {
                LOG_WARN("llm adapter has %" PRId64 " queries but this model's text window is %zu; "
                         "the text window follows the adapter (was it trained for this model?)",
                         llm_adapter->config.num_queries, chunk_len);
                chunk_len = static_cast<size_t>(llm_adapter->config.num_queries);
            }
            return;
        }
        if (has_adapter != has_llm) {
            LOG_WARN("--llm and --llm-adapter must be given together; %s",
                     has_llm ? "no adapter loaded, falling back to t5xxl" : "adapter given without --llm");
        }
        if (!use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No text encoders provided, cannot process prompts!");
            return;
        } else {
            t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", is_umt5, weight_manager);
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
        if (llm) {
            llm->get_param_tensors(tensors, "text_encoders.llm");
        }
        if (llm_adapter) {
            llm_adapter->get_param_tensors(tensors, "text_encoders.llm_adapter.adapter");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (llm) {
            llm->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (llm_adapter) {
            llm_adapter->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
        if (llm) {
            llm->set_stream_layers_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
        if (llm) {
            llm->set_flash_attention_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
        if (llm) {
            llm->set_weight_adapter(adapter);
        }
        if (llm_adapter) {
            llm_adapter->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (t5) {
            t5->runner_done();
        }
        if (llm) {
            llm->runner_done();
        }
        if (llm_adapter) {
            llm_adapter->runner_done();
        }
    }

    // Qwen3 hidden states (after the final RMSNorm) for the raw prompt text, no
    // special tokens: exactly what trainer2/data.py fed the adapter.
    sd::Tensor<float> compute_qwen_hidden(int n_threads, const std::string& text) {
        auto parsed_attention = parse_prompt_attention(text);
        std::vector<int> qwen_tokens;
        for (const auto& item : parsed_attention) {
            std::vector<int> curr_tokens = qwen_tokenizer->tokenize(item.first, nullptr);
            qwen_tokens.insert(qwen_tokens.end(), curr_tokens.begin(), curr_tokens.end());
        }
        if (qwen_tokens.empty()) {
            qwen_tokens.push_back(151643);  // qwen3 pad token
        }
        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(qwen_tokens.size()), 1}, qwen_tokens);
        return llm->compute(n_threads,
                            input_ids,
                            sd::Tensor<float>(),
                            {},
                            {},
                            false,
                            false,
                            true,
                            true);
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  size_t min_length = 0,
                                                                                  size_t max_length = 0) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            return false;
        };

        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;
        std::vector<float> t5_mask;
        if (t5 || llm_adapter) {
            for (const auto& item : parsed_attention) {
                const std::string& curr_text = item.first;
                float curr_weight            = item.second;

                std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
                t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
                t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
            }

            t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, &t5_mask, min_length, max_length, true);
            for (auto& mask_value : t5_mask) {
                mask_value = mask_value > 0.0f ? 0.0f : -HUGE_VALF;
            }
        }
        return {t5_tokens, t5_weights, t5_mask};
    }

    void modify_mask_to_attend_padding(sd::Tensor<float>* mask, int max_seq_length, int num_extra_padding = 8) {
        GGML_ASSERT(mask != nullptr);
        float* mask_data = mask->data();
        int num_pad      = 0;
        for (int64_t i = 0; i < max_seq_length; i++) {
            if (num_pad >= num_extra_padding) {
                break;
            }
            if (std::isinf(mask_data[i])) {
                mask_data[i] = 0;
                ++num_pad;
            }
        }
        // LOG_DEBUG("PAD: %d", num_pad);
    }

    SDCondition get_learned_condition_common(int n_threads,
                                             std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> token_and_weights,
                                             int clip_skip,
                                             bool zero_out_masked = false,
                                             const std::string& text = "") {
        if (!t5 && !llm_adapter) {
            SDCondition result;
            result.c_crossattn = sd::Tensor<float>::zeros({4096, 256});
            result.c_vector    = sd::Tensor<float>::full({256}, -HUGE_VALF);
            return result;
        }
        auto& t5_tokens        = std::get<0>(token_and_weights);
        auto& t5_weights       = std::get<1>(token_and_weights);
        auto& t5_attn_mask_vec = std::get<2>(token_and_weights);

        if (llm_adapter && t5_tokens.size() > chunk_len) {
            // The adapter has a fixed query window; longer prompts lose the tail.
            LOG_WARN("prompt is %zu tokens but the llm adapter window is %zu; truncating",
                     t5_tokens.size(), chunk_len);
            t5_tokens.resize(chunk_len);
            t5_weights.resize(chunk_len);
            t5_attn_mask_vec.resize(chunk_len);
        }

        int64_t t0                     = ggml_time_ms();
        sd::Tensor<float> t5_attn_mask = sd::Tensor<float>::from_vector(t5_attn_mask_vec);
        sd::Tensor<float> hidden_states;

        size_t chunk_count = t5_tokens.size() / chunk_len;

        if (llm_adapter) {
            // trainer2/: Qwen3 reads the raw prompt, the adapter resamples its
            // hidden states into the teacher encoder's 512x4096 output space.
            // Padded positions are handled below exactly like the T5 path
            // (zero_out_masked zeroes them; Wan conditions on zero-padded
            // context).
            sd::Tensor<float> qwen_hidden = compute_qwen_hidden(n_threads, text);  // [hidden, n_qwen_tokens, 1]
            GGML_ASSERT(!qwen_hidden.empty());
            sd::Tensor<int32_t> t5_ids({static_cast<int64_t>(t5_tokens.size()), 1}, t5_tokens);
            hidden_states = llm_adapter->compute(n_threads,
                                                 qwen_hidden,
                                                 t5_ids,
                                                 sd::Tensor<float>(),
                                                 sd::Tensor<float>(),
                                                 false,
                                                 true,
                                                 true);  // [4096, chunk_len, 1]
            GGML_ASSERT(!hidden_states.empty());
            // SD_DUMP_COND=<dir>: raw f32 dumps of the adapter input/output so
            // trainer2/verify_engine.py can diff the port against PyTorch.
            if (const char* dump_dir = getenv("SD_DUMP_COND")) {
                auto dump = [&](const std::string& name, const sd::Tensor<float>& t) {
                    std::string path = std::string(dump_dir) + "/" + name;
                    FILE* f          = fopen(path.c_str(), "wb");
                    if (f == nullptr) {
                        LOG_WARN("SD_DUMP_COND: cannot write %s", path.c_str());
                        return;
                    }
                    fwrite(t.data(), sizeof(float), static_cast<size_t>(t.numel()), f);
                    fclose(f);
                    LOG_INFO("SD_DUMP_COND: wrote %s (%" PRId64 " floats)", path.c_str(), t.numel());
                };
                dump("qwen_hidden.bin", qwen_hidden);
                dump("adapter_out.bin", hidden_states);
                std::string ids_path = std::string(dump_dir) + "/t5_ids.txt";
                if (FILE* f = fopen(ids_path.c_str(), "w")) {
                    for (int id : t5_tokens) {
                        fprintf(f, "%d\n", id);
                    }
                    fclose(f);
                }
            }
            hidden_states = apply_token_weights(std::move(hidden_states), t5_weights);
            if (zero_out_masked) {
                auto mask_tensor = sd::Tensor<float>::from_vector(t5_attn_mask_vec)
                                       .reshape_({1, static_cast<int64_t>(t5_attn_mask_vec.size())});
                hidden_states.masked_fill_(mask_tensor < 0.0f, 0.0f);
            }
            chunk_count = 0;  // skip the T5 loop below
        }

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            // t5
            std::vector<int> chunk_tokens(t5_tokens.begin() + chunk_idx * chunk_len,
                                          t5_tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(t5_weights.begin() + chunk_idx * chunk_len,
                                             t5_weights.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_mask(t5_attn_mask_vec.begin() + chunk_idx * chunk_len,
                                          t5_attn_mask_vec.begin() + (chunk_idx + 1) * chunk_len);

            sd::Tensor<int32_t> input_ids({static_cast<int64_t>(chunk_tokens.size())}, chunk_tokens);
            sd::Tensor<float> t5_attn_mask_chunk;
            if (use_mask) {
                t5_attn_mask_chunk = sd::Tensor<float>({static_cast<int64_t>(chunk_mask.size())}, chunk_mask);
            }

            auto chunk_hidden_states = t5->compute(n_threads,
                                                   input_ids,
                                                   t5_attn_mask_chunk,
                                                   false,
                                                   true,
                                                   true);
            GGML_ASSERT(!chunk_hidden_states.empty());
            chunk_hidden_states = apply_token_weights(std::move(chunk_hidden_states), chunk_weights);

            if (zero_out_masked) {
                auto chunk_mask_tensor = sd::Tensor<float>::from_vector(chunk_mask)
                                             .reshape_({1, static_cast<int64_t>(chunk_mask.size())});
                chunk_hidden_states.masked_fill_(chunk_mask_tensor < 0.0f, 0.0f);
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

            if (!hidden_states.empty()) {
                hidden_states = sd::ops::concat(hidden_states, chunk_hidden_states, 1);
            } else {
                hidden_states = std::move(chunk_hidden_states);
            }
        }

        modify_mask_to_attend_padding(&t5_attn_mask, static_cast<int>(t5_attn_mask.numel()), mask_pad);

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = std::move(t5_attn_mask);
        return result;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        auto tokens_and_weights = tokenize(conditioner_params.text, chunk_len, chunk_len);
        return get_learned_condition_common(n_threads,
                                            tokens_and_weights,
                                            conditioner_params.clip_skip,
                                            conditioner_params.zero_out_masked,
                                            conditioner_params.text);
    }
};

struct MiniT2IConditioner : public Conditioner {
    T5UniGramTokenizer tokenizer;
    std::shared_ptr<T5Runner> t5;
    size_t prompt_length = 256;

    MiniT2IConditioner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map      = {},
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr) {
        bool use_t5 = false;
        for (const auto& pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
                break;
            }
        }
        if (!use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No MiniT2I T5 text encoder provided, cannot process prompts!");
            return;
        }
        t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", false, weight_manager);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (t5) {
            t5->runner_done();
        }
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SDCondition result;
        if (!t5) {
            result.c_crossattn = sd::Tensor<float>::zeros({1024, static_cast<int64_t>(prompt_length)});
            result.c_vector    = sd::Tensor<float>::zeros({static_cast<int64_t>(prompt_length)});
            return result;
        }

        std::vector<int> tokens = tokenizer.encode(conditioner_params.text);
        if (tokens.size() > prompt_length) {
            tokens.resize(prompt_length);
        }
        std::vector<float> mask(tokens.size(), 1.0f);
        while (tokens.size() < prompt_length) {
            tokens.push_back(tokenizer.PAD_TOKEN_ID);
            mask.push_back(0.0f);
        }

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        std::vector<float> t5_mask(mask.size(), 0.0f);
        for (size_t i = 0; i < mask.size(); ++i) {
            t5_mask[i] = mask[i] > 0.0f ? 0.0f : -HUGE_VALF;
        }
        sd::Tensor<float> hidden_states = t5->compute(n_threads,
                                                      input_ids,
                                                      sd::Tensor<float>::from_vector(t5_mask),
                                                      false,
                                                      true,
                                                      true);
        GGML_ASSERT(!hidden_states.empty());
        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = sd::Tensor<float>::from_vector(mask);
        return result;
    }
};

// PixArt-alpha / PixArt-Sigma conditioning: T5-XXL only, padded to the model's
// fixed caption window (120 tokens for alpha, 300 for Sigma).  The padded tail is
// masked out of both the T5 self attention and the DiT cross attention, so
// c_vector carries the additive cross-attention bias rather than a pooled vector.
// With --llm (Qwen3-0.6B) and --llm-adapter (trainer/ bridge) loaded, the T5-XXL
// encoder is replaced by Qwen3 hidden states resampled into T5's embedding space;
// the T5 *tokenizer* stays, because the adapter's queries are seeded with the T5
// token ids and the DiT's cross-attention bias still follows the T5 padding.
struct PixArtT5Embedder : public Conditioner {
    T5UniGramTokenizer tokenizer;
    std::shared_ptr<T5Runner> t5;
    std::shared_ptr<BPETokenizer> qwen_tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;
    std::shared_ptr<LLMAdapter::AdapterRunner> llm_adapter;
    size_t caption_length   = 120;
    int64_t caption_channels = 4096;

    PixArtT5Embedder(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map      = {},
                     size_t caption_length                               = 120,
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : caption_length(caption_length) {
        bool use_t5      = false;
        bool has_llm     = false;
        bool has_adapter = false;
        for (const auto& pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                use_t5 = true;
            } else if (starts_with(pair.first, "text_encoders.llm_adapter.")) {
                has_adapter = true;
            } else if (starts_with(pair.first, "text_encoders.llm.")) {
                has_llm = true;
            }
        }
        if (has_llm && has_adapter) {
            LOG_INFO("PixArt conditioning via Qwen3 + bridge adapter%s", use_t5 ? " (t5xxl ignored)" : "");
            qwen_tokenizer = std::make_shared<Qwen2Tokenizer>();
            llm            = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                                   backend,
                                                   tensor_storage_map,
                                                   "text_encoders.llm",
                                                   false,
                                                   weight_manager);
            llm_adapter    = std::make_shared<LLMAdapter::AdapterRunner>(backend,
                                                                      tensor_storage_map,
                                                                      "text_encoders.llm_adapter.adapter",
                                                                      weight_manager);
            return;
        }
        if (has_adapter != has_llm) {
            LOG_WARN("PixArt: --llm and --llm-adapter must be given together; %s",
                     has_llm ? "no adapter loaded, falling back to t5xxl" : "adapter given without --llm");
        }
        if (!use_t5) {
            LOG_WARN("IMPORTANT NOTICE: No T5 text encoder provided (nor --llm + --llm-adapter), cannot process prompts!");
            return;
        }
        t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", false, weight_manager);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
        if (llm) {
            llm->get_param_tensors(tensors, "text_encoders.llm");
        }
        if (llm_adapter) {
            llm_adapter->get_param_tensors(tensors, "text_encoders.llm_adapter.adapter");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (llm) {
            llm->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (llm_adapter) {
            llm_adapter->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
        if (llm) {
            llm->set_stream_layers_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
        if (llm) {
            llm->set_flash_attention_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
        if (llm) {
            llm->set_weight_adapter(adapter);
        }
        if (llm_adapter) {
            llm_adapter->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (t5) {
            t5->runner_done();
        }
        if (llm) {
            llm->runner_done();
        }
        if (llm_adapter) {
            llm_adapter->runner_done();
        }
    }

    // Qwen3 hidden states (after the final RMSNorm) for the raw prompt text, no
    // special tokens: exactly what trainer/data.py fed the adapter.
    sd::Tensor<float> compute_qwen_hidden(int n_threads, const std::vector<std::pair<std::string, float>>& parsed_attention) {
        std::vector<int> qwen_tokens;
        for (const auto& item : parsed_attention) {
            std::vector<int> curr_tokens = qwen_tokenizer->tokenize(item.first, nullptr);
            qwen_tokens.insert(qwen_tokens.end(), curr_tokens.begin(), curr_tokens.end());
        }
        if (qwen_tokens.empty()) {
            qwen_tokens.push_back(151643);  // qwen3 pad token
        }
        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(qwen_tokens.size()), 1}, qwen_tokens);
        return llm->compute(n_threads,
                            input_ids,
                            sd::Tensor<float>(),
                            {},
                            {},
                            false,
                            false,
                            true,
                            true);
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SDCondition result;
        const int64_t length = static_cast<int64_t>(caption_length);
        if (!t5 && !llm_adapter) {
            result.c_crossattn = sd::Tensor<float>::zeros({caption_channels, length});
            result.c_vector    = sd::Tensor<float>::zeros({length});
            return result;
        }

        auto parsed_attention = parse_prompt_attention(conditioner_params.text);

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            std::vector<int> chunk = tokenizer.encode(item.first);
            tokens.insert(tokens.end(), chunk.begin(), chunk.end());
            weights.insert(weights.end(), chunk.size(), item.second);
        }

        std::vector<float> mask;
        tokenizer.pad_tokens(tokens, &weights, &mask, caption_length, caption_length, false);

        // 0 for real tokens, -inf for padding: what T5's attention expects.
        std::vector<float> t5_mask(mask.size());
        for (size_t i = 0; i < mask.size(); ++i) {
            t5_mask[i] = mask[i] > 0.0f ? 0.0f : -HUGE_VALF;
        }

        sd::Tensor<float> hidden_states;
        if (llm_adapter) {
            sd::Tensor<float> qwen_hidden = compute_qwen_hidden(n_threads, parsed_attention);  // [hidden, n_qwen_tokens, 1]
            GGML_ASSERT(!qwen_hidden.empty());
            sd::Tensor<int32_t> t5_ids({static_cast<int64_t>(tokens.size()), 1}, tokens);
            hidden_states = llm_adapter->compute(n_threads,
                                                 qwen_hidden,
                                                 t5_ids,
                                                 sd::Tensor<float>(),
                                                 sd::Tensor<float>(),
                                                 false,
                                                 true,
                                                 true);  // [caption_channels, caption_length, 1]
            // SD_DUMP_COND=<dir>: raw f32 dumps of the adapter input/output so
            // trainer/verify_engine.py can diff the port against PyTorch.
            if (const char* dump_dir = getenv("SD_DUMP_COND")) {
                auto dump = [&](const std::string& name, const sd::Tensor<float>& t) {
                    std::string path = std::string(dump_dir) + "/" + name;
                    FILE* f          = fopen(path.c_str(), "wb");
                    if (f == nullptr) {
                        LOG_WARN("SD_DUMP_COND: cannot write %s", path.c_str());
                        return;
                    }
                    fwrite(t.data(), sizeof(float), static_cast<size_t>(t.numel()), f);
                    fclose(f);
                    LOG_INFO("SD_DUMP_COND: wrote %s (%" PRId64 " floats)", path.c_str(), t.numel());
                };
                dump("qwen_hidden.bin", qwen_hidden);
                dump("adapter_out.bin", hidden_states);
                std::string ids_path = std::string(dump_dir) + "/t5_ids.txt";
                if (FILE* f = fopen(ids_path.c_str(), "w")) {
                    for (int id : tokens) {
                        fprintf(f, "%d\n", id);
                    }
                    fclose(f);
                }
            }
        } else {
            sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
            hidden_states = t5->compute(n_threads,
                                        input_ids,
                                        sd::Tensor<float>::from_vector(t5_mask),
                                        false,
                                        true,
                                        true);
        }
        GGML_ASSERT(!hidden_states.empty());
        hidden_states = apply_token_weights(std::move(hidden_states), weights);

        // The DiT adds this straight onto the cross-attention scores; diffusers uses
        // a finite -10000 there, and a finite value also survives the f16 cast the
        // flash-attention path performs.
        std::vector<float> cross_attn_bias(mask.size());
        for (size_t i = 0; i < mask.size(); ++i) {
            cross_attn_bias[i] = mask[i] > 0.0f ? 0.0f : -10000.0f;
        }

        result.c_crossattn = std::move(hidden_states);
        result.c_vector    = sd::Tensor<float>::from_vector(cross_attn_bias);
        return result;
    }
};

// ACE-Step: UMT5-base last hidden states for the tag/genre prompt, unpadded
// (batch of one), truncated to the reference pipeline's 256-token window.
// The unconditional branch is not produced here: ACE-Step's null condition is
// zeros at the text-hidden level, which the generation path builds itself.
struct ACEStepConditioner : public Conditioner {
    T5UniGramTokenizer tokenizer;
    std::shared_ptr<T5Runner> t5;
    static constexpr size_t kMaxTextTokens = 256;

    ACEStepConditioner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map      = {},
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : tokenizer(true) {
        bool has_t5 = false;
        for (const auto& pair : tensor_storage_map) {
            if (pair.first.find("text_encoders.t5xxl") != std::string::npos) {
                has_t5 = true;
                break;
            }
        }
        if (!has_t5) {
            LOG_WARN("IMPORTANT NOTICE: ACE-Step needs a UMT5-base text encoder (--llm), cannot process prompts!");
            return;
        }
        t5 = std::make_shared<T5Runner>(backend, tensor_storage_map, "text_encoders.t5xxl.transformer", true, weight_manager);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (t5) {
            t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (t5) {
            t5->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (t5) {
            t5->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (t5) {
            t5->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (t5) {
            t5->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (t5) {
            t5->runner_done();
        }
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SDCondition result;
        if (!t5) {
            return result;
        }

        auto parsed_attention = parse_prompt_attention(conditioner_params.text);

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            std::vector<int> chunk = tokenizer.encode(item.first);
            tokens.insert(tokens.end(), chunk.begin(), chunk.end());
            weights.insert(weights.end(), chunk.size(), item.second);
        }

        // appends </s> and clamps to the window; no padding for batch of one
        tokenizer.pad_tokens(tokens, &weights, nullptr, 0, kMaxTextTokens, false);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        sd::Tensor<float> hidden_states = t5->compute(n_threads,
                                                      input_ids,
                                                      sd::Tensor<float>(),
                                                      false,
                                                      true,
                                                      true);  // [768, n_tokens]
        GGML_ASSERT(!hidden_states.empty());
        hidden_states = apply_token_weights(std::move(hidden_states), weights);

        result.c_crossattn = std::move(hidden_states);
        return result;
    }
};

// MiniMax Music 3: the --llm file is not a text encoder but the whole
// autoregressive stack (global LM + RVQ depth decoder). The generation
// pipeline drives the runner directly; get_learned_condition is a stub.
struct MiniMaxMusic3Conditioner : public Conditioner {
    std::shared_ptr<MiniMaxMusic3::Music3ArRunner> ar;
    std::shared_ptr<MiniMaxMusic3::Music3Tokenizer> tokenizer;

    MiniMaxMusic3Conditioner(ggml_backend_t backend,
                             const String2TensorStorage& tensor_storage_map      = {},
                             std::shared_ptr<RunnerWeightManager> weight_manager = nullptr) {
        bool has_lm = tensor_storage_map.find("text_encoders.llm.model.layers.0.self_attn.qkv_proj.weight") != tensor_storage_map.end() &&
                      tensor_storage_map.find("text_encoders.llm.model.audio_decoder.projection.weight") != tensor_storage_map.end();
        if (!has_lm) {
            LOG_WARN("IMPORTANT NOTICE: MiniMax Music 3 needs its language model (--llm), cannot generate!");
            return;
        }
        ar        = std::make_shared<MiniMaxMusic3::Music3ArRunner>(backend, tensor_storage_map, "text_encoders.llm", weight_manager);
        tokenizer = std::make_shared<MiniMaxMusic3::Music3Tokenizer>();
        if (!tokenizer->valid) {
            LOG_WARN("MiniMax Music 3 tokenizer could not be built from the embedded tokenizer_json");
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        if (ar) {
            ar->get_param_tensors(tensors);
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        if (ar) {
            ar->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        if (ar) {
            ar->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        if (ar) {
            ar->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (ar) {
            ar->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (ar) {
            ar->reset_kv_cache();
            ar->runner_done();
        }
    }

    // caption + lyrics -> conditional and unconditional prompt ids.
    bool build_prompt_ids(const std::string& caption,
                          const std::string& lyrics,
                          std::vector<int32_t>& cond_ids,
                          std::vector<int32_t>& uncond_ids) const {
        if (!ar || !tokenizer || !tokenizer->valid) {
            return false;
        }
        const std::string text =
            std::string("<|im_start|>") +
            "<|caption_start|>" + MiniMaxMusic3::mm3_clean_caption(caption) + "<|caption_end|>" +
            "<|lyrics_start|>" + MiniMaxMusic3::mm3_normalize_lyrics(lyrics) + "<|lyrics_end|>" +
            "<|im_end|>" + "<|audio_start|>";
        std::vector<int> tokens = tokenizer->encode(text);
        // unencodable pieces (whitespace in the pruned vocab) come back as the
        // negative sentinel; the reference tokenizer drops them
        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [](int t) { return t < 0; }), tokens.end());
        if (tokens.empty() || (int64_t)tokens.size() > ar->config.max_prompt_tokens) {
            LOG_ERROR("MiniMax Music 3 prompt has %zu tokens (max %" PRId64 ")",
                      tokens.size(), ar->config.max_prompt_tokens);
            return false;
        }
        cond_ids.assign(tokens.begin(), tokens.end());
        uncond_ids = cond_ids;
        if (uncond_ids.size() >= 3) {
            std::fill(uncond_ids.begin() + 1, uncond_ids.end() - 2, ar->config.audio_cfg_token_id);
        }
        return true;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        SD_UNUSED(n_threads);
        SD_UNUSED(conditioner_params);
        return SDCondition();
    }
};

struct AnimaConditioner : public Conditioner {
    std::shared_ptr<BPETokenizer> qwen_tokenizer;
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    AnimaConditioner(ggml_backend_t backend,
                     const String2TensorStorage& tensor_storage_map      = {},
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr) {
        qwen_tokenizer = std::make_shared<Qwen2Tokenizer>();
        llm            = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::QWEN3,
                                               backend,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               false,
                                               weight_manager);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        llm->set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_stream_layers_enabled(bool enabled) override {
        llm->set_stream_layers_enabled(enabled);
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        llm->set_weight_adapter(adapter);
    }

    void runner_done() override {
        llm->runner_done();
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<int>, std::vector<float>> tokenize(std::string text) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> qwen_tokens;
        std::vector<float> qwen_weights;
        std::vector<int> t5_tokens;
        std::vector<float> t5_weights;

        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            std::vector<int> curr_tokens = qwen_tokenizer->tokenize(curr_text, nullptr);
            qwen_tokens.insert(qwen_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            // Anima uses uniform Qwen token weights.
            qwen_weights.insert(qwen_weights.end(), curr_tokens.size(), 1.f);
        }
        if (qwen_tokens.empty()) {
            qwen_tokens.push_back(151643);  // qwen3 pad token
            qwen_weights.push_back(1.f);
        }

        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = t5_tokenizer.encode(curr_text);
            t5_tokens.insert(t5_tokens.end(), curr_tokens.begin(), curr_tokens.end());
            t5_weights.insert(t5_weights.end(), curr_tokens.size(), curr_weight);
        }
        t5_tokenizer.pad_tokens(t5_tokens, &t5_weights, nullptr);

        return {qwen_tokens, qwen_weights, t5_tokens, t5_weights};
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        int64_t t0 = ggml_time_ms();

        auto tokenized     = tokenize(conditioner_params.text);
        auto& qwen_tokens  = std::get<0>(tokenized);
        auto& qwen_weights = std::get<1>(tokenized);
        auto& t5_tokens    = std::get<2>(tokenized);
        auto& t5_weights   = std::get<3>(tokenized);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(qwen_tokens.size()), 1}, qwen_tokens);
        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          sd::Tensor<float>(),
                                          {},
                                          {},
                                          false,
                                          false,
                                          true,
                                          true);
        GGML_ASSERT(!hidden_states.empty());
        hidden_states         = apply_token_weights(std::move(hidden_states), qwen_weights);
        auto t5_ids_tensor    = sd::Tensor<int32_t>::from_vector(t5_tokens);
        auto t5_weight_tensor = sd::Tensor<float>::from_vector(t5_weights);

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

        SDCondition result;
        result.c_crossattn  = std::move(hidden_states);
        result.c_t5_ids     = std::move(t5_ids_tensor);
        result.c_t5_weights = std::move(t5_weight_tensor);
        return result;
    }
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<BPETokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;
    std::shared_ptr<T5Runner> byt5;
    // trainer4/: token-aligned bridge adapter. --llm is then the small student
    // LLM (e.g. Qwen3-0.6B); its final-norm hidden states are mapped per token
    // into the big teacher LLM's conditioning space (e.g. Qwen3-4B layer 35
    // for Z-Image), replacing the teacher entirely.
    std::shared_ptr<LLMAdapter::AdapterRunner> llm_adapter;

    LLMEmbedder(ggml_backend_t backend,
                const String2TensorStorage& tensor_storage_map      = {},
                SDVersion version                                   = VERSION_QWEN_IMAGE,
                const std::string prefix                            = "",
                bool enable_vision                                  = false,
                std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : version(version) {
        LLM::LLMArch arch = LLM::LLMArch::QWEN2_5_VL;
        if (version == VERSION_FLUX2) {
            arch = LLM::LLMArch::MISTRAL_SMALL_3_2;
        } else if (sd_version_is_ernie_image(version)) {
            arch = LLM::LLMArch::MINISTRAL_3_3B;
        } else if (sd_version_is_lens(version)) {
            arch = LLM::LLMArch::GPT_OSS_20B;
        } else if (sd_version_is_pid(version) || sd_version_is_lumina2(version)) {
            arch = LLM::LLMArch::GEMMA2_2B;
        } else if (sd_version_is_lingbot_video(version) || sd_version_is_ideogram4(version) || sd_version_is_boogu_image(version) || sd_version_is_sefi_image(version) || sd_version_is_krea2(version) || sd_version_is_mage_flow(version) || sd_version_is_minimax_h3(version)) {
            arch = LLM::LLMArch::QWEN3_VL;
        } else if (sd_version_is_z_image(version) || version == VERSION_OVIS_IMAGE || version == VERSION_FLUX2_KLEIN) {
            arch = LLM::LLMArch::QWEN3;
        }
        if (arch == LLM::LLMArch::MISTRAL_SMALL_3_2 || arch == LLM::LLMArch::MINISTRAL_3_3B) {
            tokenizer = std::make_shared<MistralTokenizer>();
        } else if (arch == LLM::LLMArch::GPT_OSS_20B) {
            tokenizer = std::make_shared<GPTOSSTokenizer>();
        } else if (arch == LLM::LLMArch::GEMMA2_2B) {
            tokenizer = std::make_shared<Gemma2Tokenizer>();
        } else {
            tokenizer = std::make_shared<Qwen2Tokenizer>();
        }
        bool have_adapter = false;
        for (const auto& [name, _] : tensor_storage_map) {
            if (starts_with(name, "text_encoders.llm_adapter.")) {
                have_adapter = true;
                break;
            }
        }
        // With a bridge adapter --llm is a small plain-Qwen3 student: rope its
        // text stack natively (QWEN3 semantics) even when the version's
        // conditioner arch is QWEN3_VL (the vision tower stays QWEN3_VL for
        // the teacher's mmproj).
        llm = std::make_shared<LLM::LLMRunner>(arch,
                                               backend,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               enable_vision,
                                               weight_manager,
                                               have_adapter);
        if (have_adapter) {
            llm_adapter = std::make_shared<LLMAdapter::AdapterRunner>(backend,
                                                                      tensor_storage_map,
                                                                      "text_encoders.llm_adapter.adapter",
                                                                      weight_manager);
            if (!llm_adapter->config.token_aligned) {
                LOG_WARN("llm adapter is a resampler (T5-bridge) adapter, but this model conditions "
                         "on LLM hidden states directly; expect garbage (train a token-aligned one)");
            }
            if (arch == LLM::LLMArch::QWEN3_VL && enable_vision && !llm_adapter->config.has_vision_proj) {
                LOG_WARN("this model splices mmproj vision embeds into the LLM, but the adapter has no "
                         "vision_proj tensor; ref-image runs will fail (train a vision-ext adapter)");
            }
            LOG_INFO("LLM conditioning via student LLM + token-aligned bridge adapter");
        }
        if (sd_version_is_hunyuan_video(version)) {
            const std::string byt5_prefix = "text_encoders.t5xxl.transformer";
            for (const auto& [name, _] : tensor_storage_map) {
                if (starts_with(name, byt5_prefix + ".")) {
                    byt5 = std::make_shared<T5Runner>(backend,
                                                      tensor_storage_map,
                                                      byt5_prefix,
                                                      false,
                                                      weight_manager);
                    break;
                }
            }
        }
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
        if (byt5) {
            byt5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
        }
        if (llm_adapter) {
            llm_adapter->get_param_tensors(tensors, "text_encoders.llm_adapter.adapter");
        }
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        llm->set_max_graph_vram_bytes(max_vram_bytes);
        if (byt5) {
            byt5->set_max_graph_vram_bytes(max_vram_bytes);
        }
        if (llm_adapter) {
            llm_adapter->set_max_graph_vram_bytes(max_vram_bytes);
        }
    }

    void set_stream_layers_enabled(bool enabled) override {
        llm->set_stream_layers_enabled(enabled);
        if (byt5) {
            byt5->set_stream_layers_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_stream_layers_enabled(enabled);
        }
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
        if (byt5) {
            byt5->set_flash_attention_enabled(enabled);
        }
        if (llm_adapter) {
            llm_adapter->set_flash_attention_enabled(enabled);
        }
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        if (llm) {
            llm->set_weight_adapter(adapter);
        }
        if (byt5) {
            byt5->set_weight_adapter(adapter);
        }
        if (llm_adapter) {
            llm_adapter->set_weight_adapter(adapter);
        }
    }

    void runner_done() override {
        if (llm) {
            llm->runner_done();
        }
        if (byt5) {
            byt5->runner_done();
        }
        if (llm_adapter) {
            llm_adapter->runner_done();
        }
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  const std::pair<int, int>& attn_range,
                                                                                  size_t min_length = 0,
                                                                                  size_t max_length = 100000000,
                                                                                  bool spell_quotes = false) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        if (attn_range.first >= 0 && attn_range.second > 0) {
            if (attn_range.first > 0) {
                parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            }
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                if (spell_quotes) {
                    new_parsed_attention = split_quotation_attention(new_parsed_attention);
                }
                parsed_attention.insert(parsed_attention.end(),
                                        new_parsed_attention.begin(),
                                        new_parsed_attention.end());
            }
            if (attn_range.second < text.size()) {
                parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            }
        } else {
            parsed_attention.emplace_back(text, 1.f);
        }

        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            LOG_DEBUG("parse '%s' to %s", text.c_str(), ss.str().c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer->encode(curr_text, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        std::vector<float> mask;
        tokenizer->pad_tokens(tokens, &weights, &mask, min_length, max_length);

        // for (int i = 0; i < tokens.size(); i++) {
        //     std::cout << tokens[i] << ":" << weights[i] << ", " << i << std::endl;
        // }
        // std::cout << std::endl;

        return {tokens, weights, mask};
    }

    sd::Tensor<float> encode_prompt(int n_threads,
                                    const std::string prompt,
                                    const std::pair<int, int>& prompt_attn_range,
                                    int min_length,
                                    int hidden_states_min_length,
                                    const std::vector<std::pair<int, sd::Tensor<float>>>& image_embeds,
                                    const std::set<int>& out_layers,
                                    int prompt_template_encode_start_idx,
                                    bool spell_quotes = false,
                                    int max_length    = 100000000) {
        auto tokens_weights_mask = tokenize(prompt, prompt_attn_range, min_length, max_length, spell_quotes);
        auto& tokens             = std::get<0>(tokens_weights_mask);
        auto& weights            = std::get<1>(tokens_weights_mask);
        auto& mask               = std::get<2>(tokens_weights_mask);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
        sd::Tensor<float> attention_mask;
        if (!mask.empty()) {
            attention_mask                     = sd::Tensor<float>({static_cast<int64_t>(mask.size()), static_cast<int64_t>(mask.size())});
            const float masked_attention_value = -std::numeric_limits<float>::max() / 4.0f;
            for (size_t i1 = 0; i1 < mask.size(); ++i1) {
                for (size_t i0 = 0; i0 < mask.size(); ++i0) {
                    float value = 0.0f;
                    if (mask[i0] == 0.0f) {
                        value += masked_attention_value;
                    }
                    if (i0 > i1) {
                        value += masked_attention_value;
                    }
                    attention_mask[static_cast<int64_t>(i0 + mask.size() * i1)] = value;
                }
            }
        }

        // With a token-aligned bridge adapter the student LLM replaces the
        // teacher: take its FINAL-NORM hidden states (out_layers cleared --
        // the teacher's layer indices don't exist in the student) and map
        // them per token into the teacher's conditioning space.  Vision-ext
        // adapter: mmproj embeds are projected into the student's input
        // space (vision_proj) before the splice, and the RAW embeds are
        // handed to the adapter densely per position (vis_in path).
        std::vector<std::pair<int, sd::Tensor<float>>> student_image_embeds;
        sd::Tensor<float> vis_dense;
        const std::vector<std::pair<int, sd::Tensor<float>>>* llm_image_embeds = &image_embeds;
        if (llm_adapter && !image_embeds.empty()) {
            GGML_ASSERT(llm_adapter->config.has_vision_proj &&
                        "ref-image conditioning needs a vision-ext adapter (vision_proj)");
            const int64_t vis_dim = llm_adapter->config.vis_dim;
            if (llm_adapter->config.has_vis_in) {
                vis_dense = sd::Tensor<float>::zeros({vis_dim, static_cast<int64_t>(tokens.size()), 1});
            }
            for (const auto& [idx, embed] : image_embeds) {
                GGML_ASSERT(embed.shape()[0] == vis_dim);
                const int64_t n = embed.numel() / vis_dim;
                if (llm_adapter->config.has_vis_in) {
                    GGML_ASSERT(idx + n <= static_cast<int64_t>(tokens.size()));
                    memcpy(vis_dense.data() + idx * vis_dim,
                           embed.data(),
                           static_cast<size_t>(n * vis_dim) * sizeof(float));
                }
                student_image_embeds.emplace_back(idx,
                                                  llm_adapter->project_vision(n_threads, embed, false, true, true));
            }
            llm_image_embeds = &student_image_embeds;
        }
        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          attention_mask,
                                          *llm_image_embeds,
                                          llm_adapter ? std::set<int>{} : out_layers,
                                          false,
                                          false,
                                          true,
                                          true);
        GGML_ASSERT(!hidden_states.empty());
        if (llm_adapter) {
            auto qwen_hidden = std::move(hidden_states);
            hidden_states    = llm_adapter->compute(n_threads,
                                                    qwen_hidden,
                                                    sd::Tensor<int32_t>(),
                                                    sd::Tensor<float>(),
                                                    vis_dense,
                                                    false,
                                                    true,
                                                    true);
            GGML_ASSERT(!hidden_states.empty());
            // SD_DUMP_COND=<dir>: raw f32 dumps of the adapter input/output so
            // trainer4/verify_engine.py can diff the port against PyTorch.
            // Every conditioning pass overwrites the dump.
            if (const char* dump_dir = getenv("SD_DUMP_COND")) {
                auto dump = [&](const std::string& name, const sd::Tensor<float>& t) {
                    std::string path = std::string(dump_dir) + "/" + name;
                    FILE* f          = fopen(path.c_str(), "wb");
                    if (f == nullptr) {
                        LOG_WARN("SD_DUMP_COND: cannot write %s", path.c_str());
                        return;
                    }
                    fwrite(t.data(), sizeof(float), static_cast<size_t>(t.numel()), f);
                    fclose(f);
                    LOG_INFO("SD_DUMP_COND: wrote %s (%" PRId64 " floats)", path.c_str(), t.numel());
                };
                dump("qwen_hidden.bin", qwen_hidden);
                dump("adapter_out.bin", hidden_states);
                if (!image_embeds.empty()) {
                    std::string path = std::string(dump_dir) + "/vis_embeds.bin";
                    if (FILE* f = fopen(path.c_str(), "wb")) {
                        for (const auto& [idx, embed] : image_embeds) {
                            fwrite(embed.data(), sizeof(float), static_cast<size_t>(embed.numel()), f);
                        }
                        fclose(f);
                    }
                    path = std::string(dump_dir) + "/vis_proj.bin";
                    if (FILE* f = fopen(path.c_str(), "wb")) {
                        for (const auto& [idx, embed] : student_image_embeds) {
                            fwrite(embed.data(), sizeof(float), static_cast<size_t>(embed.numel()), f);
                        }
                        fclose(f);
                    }
                    path = std::string(dump_dir) + "/vis_pos.txt";
                    if (FILE* f = fopen(path.c_str(), "w")) {
                        for (const auto& [idx, embed] : image_embeds) {
                            fprintf(f, "%d %" PRId64 "\n", idx, embed.numel() / llm_adapter->config.vis_dim);
                        }
                        fclose(f);
                    }
                }
                std::string ids_path = std::string(dump_dir) + "/tokens.txt";
                if (FILE* f = fopen(ids_path.c_str(), "w")) {
                    for (int id : tokens) {
                        fprintf(f, "%d\n", id);
                    }
                    fclose(f);
                }
            }

            // A layerwise model (krea2, flux2, ideogram4) stacks |out_layers|
            // *intermediate* teacher layers along the feature dim. A
            // token-aligned adapter emits only the teacher's final-norm
            // space, and no engine-side reshaping can conjure the stack out
            // of it - replicating the one signal into every slot was tried
            // and conditions the DiT into unrelated imagery, because the
            // fusion blocks that consume the stack were trained to contrast
            // layers that differ. A LAYERWISE adapter (trainer6, `layer_map`
            // tensor) emits the whole ascending-tap-order stack itself, so it
            // passes when its layer count matches the model's; every other
            // pairing is refused with the reason rather than crashing in the
            // DiT's reshape or rendering noise.
            {
                const int64_t model_stack   = static_cast<int64_t>(std::max<size_t>(out_layers.size(), 1));
                const int64_t adapter_stack = llm_adapter->config.n_out_layers;
                if (adapter_stack != model_stack) {
                    LOG_ERROR("this model conditions on %" PRId64 " teacher layer(s), but the llm "
                              "adapter emits a %" PRId64 "-layer stack. Run with the full teacher "
                              "(--llm <teacher.gguf>, no --llm-adapter), or use an adapter trained "
                              "for this model family (a layerwise model needs one output head per "
                              "conditioning layer).",
                              model_stack, adapter_stack);
                    GGML_ABORT("llm adapter layer stack does not match this model's conditioning layers");
                }
            }
        } else if (const char* dump_dir = getenv("SD_DUMP_COND")) {
            // teacher reference path (--llm <teacher> without adapter): dump
            // the raw conditioning hidden states + tokens so the trainer can
            // verify its precompute against the engine's ground truth.
            std::string path = std::string(dump_dir) + "/llm_hidden.bin";
            if (FILE* f = fopen(path.c_str(), "wb")) {
                fwrite(hidden_states.data(), sizeof(float), static_cast<size_t>(hidden_states.numel()), f);
                fclose(f);
                LOG_INFO("SD_DUMP_COND: wrote %s (%" PRId64 " floats)", path.c_str(), hidden_states.numel());
            }
            if (!image_embeds.empty()) {
                path = std::string(dump_dir) + "/vis_embeds.bin";
                if (FILE* f = fopen(path.c_str(), "wb")) {
                    for (const auto& [idx, embed] : image_embeds) {
                        fwrite(embed.data(), sizeof(float), static_cast<size_t>(embed.numel()), f);
                    }
                    fclose(f);
                }
                path = std::string(dump_dir) + "/vis_pos.txt";
                if (FILE* f = fopen(path.c_str(), "w")) {
                    for (const auto& [idx, embed] : image_embeds) {
                        fprintf(f, "%d %" PRId64 "\n", idx, embed.shape()[1]);
                    }
                    fclose(f);
                }
            }
            std::string ids_path = std::string(dump_dir) + "/tokens.txt";
            if (FILE* f = fopen(ids_path.c_str(), "w")) {
                for (int id : tokens) {
                    fprintf(f, "%d\n", id);
                }
                fclose(f);
            }
        }
        hidden_states = apply_token_weights(std::move(hidden_states), weights);
        GGML_ASSERT(hidden_states.shape()[1] > prompt_template_encode_start_idx);

        int64_t zero_pad_len = 0;
        if (hidden_states_min_length > 0) {
            if (hidden_states.shape()[1] - prompt_template_encode_start_idx < hidden_states_min_length) {
                zero_pad_len = hidden_states_min_length - hidden_states.shape()[1] + prompt_template_encode_start_idx;
            }
        }

        sd::Tensor<float> new_hidden_states = sd::ops::slice(hidden_states,
                                                             1,
                                                             prompt_template_encode_start_idx,
                                                             hidden_states.shape()[1]);
        if (zero_pad_len > 0) {
            auto pad_shape    = new_hidden_states.shape();
            pad_shape[1]      = zero_pad_len;
            new_hidden_states = sd::ops::concat(new_hidden_states,
                                                sd::Tensor<float>::zeros(std::move(pad_shape)),
                                                1);
        }

        return new_hidden_states;
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        std::vector<std::string> extra_prompts;
        std::vector<std::pair<int, int>> extra_prompts_attn_range;
        std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
        int prompt_template_encode_start_idx = 34;
        int min_length                       = 0;  // pad tokens
        int max_length                       = 100000000;
        int hidden_states_min_length         = 0;  // zero pad hidden_states
        bool spell_quotes                    = false;
        std::set<int> out_layers;

        int64_t t0 = ggml_time_ms();

        if (sd_version_is_minimax_h3(version)) {
            prompt_template_encode_start_idx = 0;
            prompt                           = conditioner_params.text;
            prompt_attn_range                = {0, 0};
            // H3 consumes the raw, unnormalized output of truncated Qwen3-VL layer 50.
            out_layers = {50};
        } else if (sd_version_is_hunyuan_video(version)) {
            prompt_template_encode_start_idx = 98;
            out_layers                       = {26};

            prompt =
                "<|im_start|>system\nYou are a helpful assistant. Describe the video by detailing the following aspects:\n"
                "1. The main content and theme of the video.\n"
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n"
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n"
                "4. background environment, light, style and atmosphere.\n"
                "5. camera angles, movements, and transitions used in the video.<|im_end|>\n"
                "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());
            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (sd_version_is_lingbot_video(version)) {
            const int pad_token = 151643;
            const std::string prompt_prefix =
                "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
                "a text prompt with an image reference, or a text prompt with a video reference "
                "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
                "visual descriptions suitable for video generation. Evaluate the level of detail "
                "in the user's input: if it is simple, enrich it by adding specifics about colors, "
                "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
                "progression, and spatial relationships to create vivid, concrete, and temporally "
                "coherent scenes to create vivid and concrete scenes. Please generate only the "
                "enhanced description for the prompt below and avoid including any additional "
                "commentary or evaluations:<|im_end|>\n<|im_start|>user\n";

            auto prefix_tokens               = tokenizer->encode(prompt_prefix, nullptr);
            prompt_template_encode_start_idx = 0;
            for (int token : prefix_tokens) {
                if (token != pad_token) {
                    prompt_template_encode_start_idx++;
                }
            }
            LOG_DEBUG("prompt_template_encode_start_idx %d", prompt_template_encode_start_idx);

            prompt = prompt_prefix;
            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("LingBotVideoI2VPipeline");
                const std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    const int factor  = llm->config.vision.patch_size * llm->config.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);

                    int min_pixels = 4 * factor * factor;
                    int max_pixels = 16384 * factor * factor;

                    int h_bar = std::max(factor, static_cast<int>(std::round(static_cast<double>(height) / factor) * factor));
                    int w_bar = std::max(factor, static_cast<int>(std::round(static_cast<double>(width) / factor) * factor));

                    if (std::max(height, width) > 200 * std::min(height, width)) {
                        LOG_WARN("LingBotVideo image aspect ratio is very large: %dx%d", width, height);
                    }

                    double current_area = static_cast<double>(h_bar) * w_bar;
                    if (current_area > max_pixels) {
                        double beta = std::sqrt((static_cast<double>(height) * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
                        w_bar       = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
                    } else if (current_area < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * factor;
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * factor;
                    }

                    LOG_DEBUG("resize LingBotVideo ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);
                    auto resized_image = clip_preprocess(image, w_bar, h_bar);
                    auto image_embed   = llm->encode_image(n_threads, resized_image, false, true, true);
                    GGML_ASSERT(!image_embed.empty());

                    std::string image_prefix = prompt + img_prompt + "<|vision_start|>";
                    int image_embed_idx      = static_cast<int>(tokenizer->encode(image_prefix, nullptr).size());
                    image_embeds.emplace_back(image_embed_idx, image_embed);

                    img_prompt += "<|vision_start|>";
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(img_prompt.size() + static_cast<size_t>(num_image_tokens) * placeholder.size() + 32);
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }
                prompt += img_prompt;
            }

            prompt += conditioner_params.text;
            prompt_attn_range = {0, 0};
            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (sd_version_is_qwen_image(version) || sd_version_is_mage_flow(version)) {
            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("%s", sd_version_is_mage_flow(version) ? "MageFlowEditPipeline" : "QwenImageEditPlusPipeline");
                prompt_template_encode_start_idx = 64;
                int image_embed_idx              = 64 + 6;

                int min_pixels          = sd_version_is_mage_flow(version) ? -1 : 384 * 384;
                int max_pixels          = sd_version_is_mage_flow(version) ? 384 : 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->config.vision.patch_size * llm->config.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    int h_bar         = std::max(static_cast<int>(factor),
                                                 static_cast<int>(std::round(height / factor) * factor));
                    int w_bar         = std::max(static_cast<int>(factor),
                                                 static_cast<int>(std::round(width / factor) * factor));

                    if (sd_version_is_mage_flow(version)) {
                        int current_max_side = std::max(height, width);
                        if (max_pixels > 0 && current_max_side > max_pixels) {
                            double beta = static_cast<double>(max_pixels) / current_max_side;
                            h_bar       = std::max(static_cast<int>(factor),
                                                   static_cast<int>(std::floor(height * beta / factor)) * static_cast<int>(factor));
                            w_bar       = std::max(static_cast<int>(factor),
                                                   static_cast<int>(std::floor(width * beta / factor)) * static_cast<int>(factor));
                        }
                    } else if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                        w_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                    }

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);

                    auto image_embed = llm->encode_image(n_threads, resized_image, false, true, true);
                    GGML_ASSERT(!image_embed.empty());
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed.shape()[1]) + 6;

                    img_prompt += (sd_version_is_mage_flow(version) ? "Image " : "Picture ") + std::to_string(i + 1) + ": <|vision_start|>";
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(num_image_tokens * placeholder.size());
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
                prompt += img_prompt;

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            } else {
                prompt_template_encode_start_idx = 34;

                prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
            if (sd_version_is_mage_flow(version)) {
                max_length = 2048 + prompt_template_encode_start_idx;
            }
        } else if (sd_version_is_boogu_image(version)) {
            prompt_template_encode_start_idx = 0;

            const std::string t2i_system_prompt =
                "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";
            const std::string edit_system_prompt =
                "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";
            const bool has_ref_images = llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty();
            const bool text_empty     = conditioner_params.text.find_first_not_of(" \t\r\n") == std::string::npos;

            if (has_ref_images) {
                LOG_INFO("BooguImageEditPipeline");
                const std::string prompt_prefix = "<|im_start|>system\n" + edit_system_prompt + "<|im_end|>\n<|im_start|>user\n";
                std::string img_prompt;
                const std::string placeholder = "<|image_pad|>";

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->config.vision.patch_size * llm->config.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    double beta       = std::sqrt((384.0 * 384.0) / (static_cast<double>(height) * static_cast<double>(width)));
                    int h_bar         = std::max(static_cast<int>(factor),
                                                 static_cast<int>(std::round(height * beta / factor)) * static_cast<int>(factor));
                    int w_bar         = std::max(static_cast<int>(factor),
                                                 static_cast<int>(std::round(width * beta / factor)) * static_cast<int>(factor));

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);
                    auto image_embed   = llm->encode_image(n_threads, resized_image, false, true, true);
                    GGML_ASSERT(!image_embed.empty());

                    std::string image_prefix = prompt_prefix + img_prompt + "<|vision_start|>";
                    int image_embed_idx      = static_cast<int>(tokenizer->encode(image_prefix, nullptr).size());
                    image_embeds.emplace_back(image_embed_idx, image_embed);

                    img_prompt += "<|vision_start|>";
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(img_prompt.size() + static_cast<size_t>(num_image_tokens) * placeholder.size() + 32);
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt                  = prompt_prefix + img_prompt;
                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());
                prompt += "<|im_end|>\n";
            } else {
                const std::string& system_prompt = text_empty ? edit_system_prompt : t2i_system_prompt;
                prompt                           = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n";
                prompt_attn_range.first          = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());
                prompt += "<|im_end|>\n";
            }
        } else if (sd_version_is_krea2(version)) {
            prompt_template_encode_start_idx = 34;
            out_layers                       = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};

            prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (sd_version_is_longcat(version)) {
            spell_quotes = true;

            if (llm->enable_vision && conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("LongCatEditPipeline");
                prompt_template_encode_start_idx = 67;
                min_length                       = 512 + prompt_template_encode_start_idx;
                int image_embed_idx              = 36 + 6;

                int min_pixels          = 384 * 384;
                int max_pixels          = 560 * 560;
                std::string placeholder = "<|image_pad|>";
                std::string img_prompt;

                for (int i = 0; i < conditioner_params.ref_images->size(); i++) {
                    const auto& image = (*conditioner_params.ref_images)[i];
                    double factor     = llm->config.vision.patch_size * llm->config.vision.spatial_merge_size;
                    int height        = static_cast<int>(image.shape()[1]);
                    int width         = static_cast<int>(image.shape()[0]);
                    int h_bar         = static_cast<int>(std::round(height / factor) * factor);
                    int w_bar         = static_cast<int>(std::round(width / factor) * factor);

                    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                        h_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                        w_bar       = std::max(static_cast<int>(factor),
                                               static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                        h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                        w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                    }

                    LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, height, width, h_bar, w_bar);

                    auto resized_image = clip_preprocess(image, w_bar, h_bar);
                    auto image_embed   = llm->encode_image(n_threads, resized_image, false, true, true);
                    GGML_ASSERT(!image_embed.empty());
                    image_embeds.emplace_back(image_embed_idx, image_embed);
                    image_embed_idx += 1 + static_cast<int>(image_embed.shape()[1]) + 6;

                    img_prompt += "<|vision_start|>";
                    int64_t num_image_tokens = image_embed.shape()[1];
                    img_prompt.reserve(num_image_tokens * placeholder.size());
                    for (int j = 0; j < num_image_tokens; j++) {
                        img_prompt += placeholder;
                    }
                    img_prompt += "<|vision_end|>";
                }

                prompt = "<|im_start|>system\nAs an image editing expert, first analyze the content and attributes of the input image(s). Then, based on the user's editing instructions, clearly and precisely determine how to modify the given image(s), ensuring that only the specified parts are altered and all other aspects remain consistent with the original(s).<|im_end|>\n<|im_start|>user\n";
                prompt += img_prompt;
            } else {
                prompt_template_encode_start_idx = 36;
                min_length                       = 512 + prompt_template_encode_start_idx;

                prompt = "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n";
            }

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (version == VERSION_FLUX2) {
            prompt_template_encode_start_idx = 0;
            hidden_states_min_length         = 512;
            out_layers                       = {10, 20, 30};

            prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "[/INST]";
        } else if (sd_version_is_ideogram4(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36};

            prompt = "<|im_start|>user\n";
            prompt += conditioner_params.text;
            prompt += "<|im_end|>\n<|im_start|>assistant\n";
            prompt_attn_range = {0, 0};
        } else if (sd_version_is_ernie_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {25};  // -2

            prompt_attn_range.first = 0;
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());
        } else if (sd_version_is_lens(version)) {
            prompt_template_encode_start_idx = 97;
            min_length                       = 0;
            max_length                       = 512;
            out_layers                       = {6, 12, 18, 24};

            prompt =
                "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                "Knowledge cutoff: 2024-06\n"
                "Current date: 2026-05-26\n"  // fix for current date
                "\n"
                "Reasoning: medium\n"
                "\n"
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n"
                "\n"
                "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background.\n"
                "\n"
                "<|end|><|start|>user<|message|>";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|end|><|start|>assistant<|channel|>analysis<|message|>Need to generate one image according to the description.<|end|><|start|>assistant<|channel|>final<|message|>";
        } else if (sd_version_is_z_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {35};  // -2

            if (conditioner_params.ref_images != nullptr && !conditioner_params.ref_images->empty()) {
                LOG_INFO("ZImageOmniPipeline");
                prompt = "<|im_start|>user\n<|vision_start|>";
                for (int i = 0; i < conditioner_params.ref_images->size() - 1; i++) {
                    extra_prompts.push_back("<|vision_end|><|vision_start|>");
                }
                extra_prompts.push_back("<|vision_end|>" + conditioner_params.text + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>");
                extra_prompts.push_back("<|vision_end|><|im_end|>");
            } else {
                prompt = "<|im_start|>user\n";

                prompt_attn_range.first = static_cast<int>(prompt.size());
                prompt += conditioner_params.text;
                prompt_attn_range.second = static_cast<int>(prompt.size());

                prompt += "<|im_end|>\n<|im_start|>assistant\n";
            }
        } else if (version == VERSION_FLUX2_KLEIN) {
            prompt_template_encode_start_idx = 0;
            min_length                       = 512;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else if (sd_version_is_sefi_image(version)) {
            prompt_template_encode_start_idx = 0;
            min_length                       = 1024;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (version == VERSION_OVIS_IMAGE) {
            prompt_template_encode_start_idx = 28;
            min_length                       = prompt_template_encode_start_idx + 256;

            prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else if (sd_version_is_lumina2(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {25};  // Gemma2-2B hidden_states[-2]

            prompt =
                "You are an assistant designed to generate superior images with the superior "
                "degree of image-text alignment based on textual prompts or user prompts. "
                "<Prompt Start> ";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());
        } else if (sd_version_is_pid(version)) {
            constexpr int pixeldit_max_length = 300;
            const std::string chi_prompt =
                "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n"
                "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
                "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n"
                "Here are examples of how to transform or refine prompts:\n"
                "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
                "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n"
                "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\n"
                "User Prompt: ";
            auto chi_tokens       = std::get<0>(tokenize(chi_prompt, {0, 0}));
            size_t num_chi_tokens = chi_tokens.size();
            max_length            = (int)num_chi_tokens + pixeldit_max_length - 2;
            min_length            = max_length;

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            auto hidden_states = encode_prompt(n_threads,
                                               prompt,
                                               prompt_attn_range,
                                               min_length,
                                               0,
                                               image_embeds,
                                               out_layers,
                                               0,
                                               false,
                                               max_length);
            GGML_ASSERT(!hidden_states.empty());

            if (hidden_states.shape()[1] > pixeldit_max_length) {
                auto bos      = sd::ops::slice(hidden_states, 1, 0, 1);
                auto tail     = sd::ops::slice(hidden_states,
                                               1,
                                               hidden_states.shape()[1] - (pixeldit_max_length - 1),
                                               hidden_states.shape()[1]);
                hidden_states = sd::ops::concat(bos, tail, 1);
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);

            SDCondition result;
            result.c_crossattn = std::move(hidden_states);
            return result;
        } else {
            GGML_ABORT("unknown version %d", version);
        }

        auto hidden_states = encode_prompt(n_threads,
                                           prompt,
                                           prompt_attn_range,
                                           min_length,
                                           hidden_states_min_length,
                                           image_embeds,
                                           out_layers,
                                           prompt_template_encode_start_idx,
                                           spell_quotes,
                                           max_length);
        std::vector<sd::Tensor<float>> extra_hidden_states_vec;
        if (sd_version_is_hunyuan_video(version) && byt5) {
            std::vector<std::string> quoted_texts;
            auto collect_quoted = [&](const std::string& open, const std::string& close) {
                size_t begin = 0;
                while ((begin = conditioner_params.text.find(open, begin)) != std::string::npos) {
                    size_t content_begin = begin + open.size();
                    size_t end           = conditioner_params.text.find(close, content_begin);
                    if (end == std::string::npos) {
                        break;
                    }
                    quoted_texts.push_back(conditioner_params.text.substr(content_begin, end - content_begin));
                    begin = end + close.size();
                }
            };
            collect_quoted("\"", "\"");
            collect_quoted("\xE2\x80\x98", "\xE2\x80\x99");
            collect_quoted("\xE2\x80\x9C", "\xE2\x80\x9D");

            if (!quoted_texts.empty()) {
                std::string byt5_text;
                for (const auto& text : quoted_texts) {
                    byt5_text += "Text \"" + text + "\". ";
                }
                std::vector<int> tokens;
                tokens.reserve(byt5_text.size() + 1);
                for (unsigned char byte : byt5_text) {
                    tokens.push_back(static_cast<int>(byte) + 3);
                }
                tokens.push_back(1);
                sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, tokens);
                auto byt5_hidden_states = byt5->compute(n_threads,
                                                        input_ids,
                                                        sd::Tensor<float>(),
                                                        false,
                                                        true,
                                                        true);
                GGML_ASSERT(!byt5_hidden_states.empty());
                extra_hidden_states_vec.push_back(std::move(byt5_hidden_states));
            }
        }
        for (int i = 0; i < extra_prompts.size(); i++) {
            auto extra_hidden_states = encode_prompt(n_threads,
                                                     extra_prompts[i],
                                                     extra_prompts_attn_range[i],
                                                     min_length,
                                                     hidden_states_min_length,
                                                     image_embeds,
                                                     out_layers,
                                                     prompt_template_encode_start_idx,
                                                     spell_quotes,
                                                     max_length);
            extra_hidden_states_vec.push_back(std::move(extra_hidden_states));
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        SDCondition result;
        result.c_crossattn        = std::move(hidden_states);
        result.extra_c_crossattns = std::move(extra_hidden_states_vec);
        return result;
    }
};

struct LTXAVTextProjection : public GGMLBlock {
    static constexpr int64_t kHiddenSize = 3840;
    static constexpr int64_t kNumStates  = 49;
    bool dual_projection                 = false;

    LTXAVTextProjection(bool dual_projection = false)
        : dual_projection(dual_projection) {
        if (dual_projection) {
            blocks["video_aggregate_embed"] = std::make_shared<Linear>(kHiddenSize * kNumStates, 4096, true);
            blocks["audio_aggregate_embed"] = std::make_shared<Linear>(kHiddenSize * kNumStates, 2048, true);
        } else {
            blocks["projection"] = std::make_shared<Linear>(kHiddenSize * kNumStates, kHiddenSize, false);
        }
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        if (!dual_projection) {
            auto projection = std::dynamic_pointer_cast<Linear>(blocks["projection"]);
            return projection->forward(ctx, x);
        }

        auto video_projection = std::dynamic_pointer_cast<Linear>(blocks["video_aggregate_embed"]);
        auto audio_projection = std::dynamic_pointer_cast<Linear>(blocks["audio_aggregate_embed"]);
        auto video_in         = ggml_ext_scale(ctx->ggml_ctx, x, std::sqrt(4096.f / static_cast<float>(kHiddenSize)));
        auto audio_in         = ggml_ext_scale(ctx->ggml_ctx, x, std::sqrt(2048.f / static_cast<float>(kHiddenSize)));
        auto video            = video_projection->forward(ctx, video_in);
        auto audio            = audio_projection->forward(ctx, audio_in);
        return ggml_concat(ctx->ggml_ctx, video, audio, 0);
    }
};

struct LTXAVTextProjectionRunner : public GGMLRunner {
    LTXAVTextProjection model;

    LTXAVTextProjectionRunner(ggml_backend_t backend,
                              const String2TensorStorage& tensor_storage_map      = {},
                              const std::string& prefix                           = "",
                              std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          model(tensor_storage_map.find(prefix + ".video_aggregate_embed.weight") != tensor_storage_map.end()) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "ltxav_text_projection";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
        model.get_param_tensors(tensors, prefix);
    }

    ggml_cgraph* build_graph(const sd::Tensor<float>& x_tensor) {
        ggml_cgraph* gf = ggml_new_graph(compute_ctx);
        auto x          = make_input(x_tensor);
        auto runner_ctx = get_context();
        auto out        = model.forward(&runner_ctx, x);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    sd::Tensor<float> compute(int n_threads,
                              const sd::Tensor<float>& x,
                              bool auto_free           = true,
                              bool free_compute_buffer = true,
                              bool free_compute_params = true) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_graph(x);
        };
        return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, auto_free, free_compute_buffer, free_compute_params));
    }
};

struct LTXAVEmbedder : public Conditioner {
    static constexpr int64_t kHiddenSize = 3840;
    static constexpr int64_t kNumStates  = 49;
    static constexpr int64_t kMinLength  = 1024;

    std::shared_ptr<GemmaTokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;
    std::shared_ptr<LTXAVTextProjectionRunner> projector;
    bool dual_projection = false;

    LTXAVEmbedder(ggml_backend_t backend,
                  const String2TensorStorage& tensor_storage_map      = {},
                  const std::string& llm_prefix                       = "text_encoders.llm",
                  const std::string& projector_prefix                 = "text_embedding_projection",
                  std::shared_ptr<RunnerWeightManager> weight_manager = nullptr) {
        tokenizer       = std::make_shared<GemmaTokenizer>();
        llm             = std::make_shared<LLM::LLMRunner>(LLM::LLMArch::GEMMA3_12B,
                                               backend,
                                               tensor_storage_map,
                                               llm_prefix,
                                               false,
                                               weight_manager);
        dual_projection = tensor_storage_map.find(projector_prefix + ".video_aggregate_embed.weight") != tensor_storage_map.end();
        projector       = std::make_shared<LTXAVTextProjectionRunner>(backend,
                                                                tensor_storage_map,
                                                                projector_prefix,
                                                                weight_manager);
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
        projector->get_param_tensors(tensors, "text_embedding_projection");
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
        projector->set_flash_attention_enabled(enabled);
    }

    void set_max_graph_vram_bytes(size_t max_vram_bytes) override {
        llm->set_max_graph_vram_bytes(max_vram_bytes);
        projector->set_max_graph_vram_bytes(max_vram_bytes);
    }

    void set_weight_adapter(const std::shared_ptr<WeightAdapter>& adapter) override {
        llm->set_weight_adapter(adapter);
        projector->set_weight_adapter(adapter);
    }

    void runner_done() override {
        llm->runner_done();
        projector->runner_done();
    }

    std::tuple<std::vector<int>, std::vector<float>, std::vector<float>> tokenize(std::string text,
                                                                                  const std::pair<int, int>& attn_range) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        if (attn_range.first >= 0 && attn_range.second > 0) {
            if (attn_range.first > 0) {
                parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
            }
            if (attn_range.second - attn_range.first > 0) {
                auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
                parsed_attention.insert(parsed_attention.end(), new_parsed_attention.begin(), new_parsed_attention.end());
            }
            if (static_cast<size_t>(attn_range.second) < text.size()) {
                parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
            }
        } else {
            parsed_attention.emplace_back(text, 1.f);
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            auto curr_tokens = tokenizer->encode(item.first, nullptr);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), item.second);
        }

        std::vector<float> mask;
        tokenizer->pad_tokens(tokens, &weights, &mask, kMinLength);
        return {tokens, weights, mask};
    }

    sd::Tensor<float> encode_prompt(int n_threads,
                                    const std::string& prompt,
                                    const std::pair<int, int>& prompt_attn_range) {
        auto tokens_weights_mask = tokenize(prompt, prompt_attn_range);
        auto& tokens             = std::get<0>(tokens_weights_mask);
        auto& weights            = std::get<1>(tokens_weights_mask);
        auto& mask               = std::get<2>(tokens_weights_mask);

        sd::Tensor<int32_t> input_ids({static_cast<int64_t>(tokens.size())}, std::vector<int32_t>(tokens.begin(), tokens.end()));
        sd::Tensor<float> attention_mask;
        if (!mask.empty()) {
            const float mask_min = std::numeric_limits<float>::lowest() / 4.0f;
            attention_mask       = sd::Tensor<float>({static_cast<int64_t>(mask.size()), static_cast<int64_t>(mask.size())});
            for (size_t i1 = 0; i1 < mask.size(); ++i1) {
                for (size_t i0 = 0; i0 < mask.size(); ++i0) {
                    float value = 0.0f;
                    if (mask[i0] == 0.0f) {
                        value += mask_min;
                    }
                    if (i0 > i1) {
                        value += mask_min;
                    }
                    attention_mask[static_cast<int64_t>(i0 + mask.size() * i1)] = value;
                }
            }
        }

        auto hidden_states = llm->compute(n_threads,
                                          input_ids,
                                          attention_mask,
                                          {},
                                          {},
                                          true,
                                          false,
                                          true,
                                          true);
        GGML_ASSERT(!hidden_states.empty());
        hidden_states = apply_token_weights(std::move(hidden_states), weights);

        int64_t valid_tokens = 0;
        for (float value : mask) {
            valid_tokens += static_cast<int64_t>(value > 0.0f);
        }
        GGML_ASSERT(valid_tokens > 0);

        hidden_states = sd::ops::slice(hidden_states,
                                       1,
                                       hidden_states.shape()[1] - valid_tokens,
                                       hidden_states.shape()[1]);
        hidden_states.reshape_({kHiddenSize, kNumStates, valid_tokens});
        hidden_states = hidden_states.permute({1, 0, 2});

        if (dual_projection) {
            for (int64_t state_idx = 0; state_idx < kNumStates; ++state_idx) {
                for (int64_t token_idx = 0; token_idx < valid_tokens; ++token_idx) {
                    double sq_sum = 0.0;
                    for (int64_t hidden_idx = 0; hidden_idx < kHiddenSize; ++hidden_idx) {
                        float value = hidden_states.index(state_idx, hidden_idx, token_idx);
                        sq_sum += static_cast<double>(value) * static_cast<double>(value);
                    }

                    float inv_rms = 1.0f / std::sqrt(static_cast<float>(sq_sum / static_cast<double>(kHiddenSize)) + 1e-6f);
                    for (int64_t hidden_idx = 0; hidden_idx < kHiddenSize; ++hidden_idx) {
                        hidden_states.index(state_idx, hidden_idx, token_idx) *= inv_rms;
                    }
                }
            }
        } else {
            for (int64_t state_idx = 0; state_idx < kNumStates; ++state_idx) {
                double sum      = 0.0;
                float min_value = std::numeric_limits<float>::infinity();
                float max_value = -std::numeric_limits<float>::infinity();
                for (int64_t token_idx = 0; token_idx < valid_tokens; ++token_idx) {
                    for (int64_t hidden_idx = 0; hidden_idx < kHiddenSize; ++hidden_idx) {
                        float value = hidden_states.index(state_idx, hidden_idx, token_idx);
                        sum += value;
                        min_value = std::min(min_value, value);
                        max_value = std::max(max_value, value);
                    }
                }

                float mean_value  = static_cast<float>(sum / static_cast<double>(kHiddenSize * valid_tokens));
                float denom       = max_value - min_value + 1e-6f;
                float scale_value = 8.0f / denom;
                for (int64_t token_idx = 0; token_idx < valid_tokens; ++token_idx) {
                    for (int64_t hidden_idx = 0; hidden_idx < kHiddenSize; ++hidden_idx) {
                        float value                                           = hidden_states.index(state_idx, hidden_idx, token_idx);
                        hidden_states.index(state_idx, hidden_idx, token_idx) = (value - mean_value) * scale_value;
                    }
                }
            }
        }

        hidden_states.reshape_({kNumStates * kHiddenSize, valid_tokens});
        return projector->compute(n_threads, hidden_states, false, true, true);
    }

    SDCondition get_learned_condition(int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        int64_t t0 = ggml_time_ms();

        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        prompt_attn_range.first = static_cast<int>(prompt.size());
        prompt += conditioner_params.text;
        prompt_attn_range.second = static_cast<int>(prompt.size());

        auto hidden_states = encode_prompt(n_threads, prompt, prompt_attn_range);
        GGML_ASSERT(!hidden_states.empty());

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing LTXAV condition graph completed, taking %" PRId64 " ms", t1 - t0);

        SDCondition result;
        result.c_crossattn = std::move(hidden_states);
        return result;
    }
};

#endif  // __SD_CONDITIONING_CONDITIONER_HPP__
