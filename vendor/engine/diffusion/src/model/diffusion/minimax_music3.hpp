#ifndef __SD_MODEL_DIFFUSION_MINIMAX_MUSIC3_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_MUSIC3_HPP__

#include <algorithm>
#include <cmath>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "core/ggml_extend.hpp"
#include "json.hpp"
#include "model/diffusion/model.hpp"
#include "tokenizers/bpe_tokenizer.h"
#include "tokenizers/vocab_provider.h"

// MiniMax Music 3 text-to-music.
//
// Three GGUF files drive the pipeline:
//   --llm             the autoregressive stack: a Qwen-style global LM
//                     (model.layers.*, fused qkv/gate_up, split
//                     embed_tokens_prefill/embed_tokens_audio, compact
//                     lm_head_pruned over the 16384 semantic codes + EOS) plus
//                     the 4-layer RVQ depth decoder (model.audio_decoder.*,
//                     residual embeddings in model.audio_extra_embedding) and
//                     an embedded tokenizer_json.
//   --diffusion-model the 36-layer flow-matching transformer
//                     (diffusion_transformer.*) together with the condition
//                     encoder (cond_layer_logits/cond_layer_scale softmax mix
//                     + latent_conditioners.0 conv).
//   --vae             the DAC-style vocoder (dec_in_proj + decoder.model.*,
//                     weight-norm convs + snake activations), decoding
//                     [frames, 128] latents to 44.1 kHz stereo.
//
// Reference: the audio.cpp minimax_music3 community model (ar_runtime /
// depth_decoder / condition_encoder / flow_transformer / flow_sampler /
// vocoder). Per-frame conditioning is the stack of the LM hidden plus the 7
// depth-decoder hiddens (8 x 4096 per frame).
namespace MiniMaxMusic3 {

    struct Music3Config {
        // global LM
        int64_t vocab_size          = 200000;  // virtual id space (prefill + audio ids)
        int64_t prefill_vocab       = 151675;  // rows of embed_tokens_prefill
        int64_t hidden_size         = 4096;
        int64_t intermediate_size   = 12288;
        int64_t layers              = 36;
        int64_t attention_heads     = 32;
        int64_t kv_heads            = 8;
        int64_t head_dim            = 128;
        float rms_norm_eps          = 1.0e-6f;
        float rope_theta            = 1000000.0f;
        int32_t audio_code_offset   = 151675;
        int32_t semantic_vocab_size = 16384;
        int32_t audio_end_token_id  = 151670;
        int32_t audio_cfg_token_id  = 151654;

        // depth decoder
        int64_t depth_audio_vocab       = 1024;
        int64_t depth_intermediate_size = 6144;
        int64_t depth_positions         = 16;
        int64_t depth_heads             = 16;
        int64_t depth_codebooks         = 8;
        int64_t depth_layers            = 4;

        // condition encoder
        int64_t condition_layers = 8;
        int64_t condition_dim    = 2048;
        // 24000 Hz * hop 960 (AR frames, 25/s) -> 44100 Hz * hop 512 latents
        double condition_ratio = (44100.0 / 24000.0) * (960.0 / 512.0);  // 3.4453125

        // flow transformer
        int64_t flow_in_channels    = 128;
        int64_t flow_layers         = 36;
        int64_t flow_heads          = 32;
        int64_t flow_head_dim       = 64;
        int64_t flow_ff_inner       = 8192;
        int64_t flow_rotary_dim     = 32;
        int64_t flow_fourier_dim    = 256;
        float flow_rope_theta       = 10000.0f;

        // pipeline
        int64_t max_prompt_tokens    = 5000;
        int64_t max_audio_frames     = 9000;
        int64_t frame_rate           = 25;
        int64_t chunk_frames         = 200;
        int64_t chunk_hop_frames     = 100;
        int sample_rate              = 44100;
        int64_t vocoder_hop_length   = 512;
    };

    /* ------------------------------- tokenizer ------------------------------ */

    // Qwen2-style BPE fed by the tokenizer.json embedded in the --llm GGUF
    // (via VocabProvider overrides "minimax_music3.vocab"/".merges"), so the
    // model's added tokens (<|caption_start|> etc.) resolve to their real ids.
    class Music3Tokenizer : public BPETokenizer {
    public:
        bool valid = false;

        Music3Tokenizer() {
            UNK_TOKEN = "<|endoftext|>";
            EOS_TOKEN = "<|endoftext|>";
            PAD_TOKEN = "<|endoftext|>";

            // The vocab is pruned (no whitespace tokens at all); pieces that
            // fail to map must be DROPPED like the reference tokenizer does,
            // not replaced. A negative sentinel is filtered out after encode.
            UNK_TOKEN_ID = -1;
            EOS_TOKEN_ID = 151643;
            PAD_TOKEN_ID = 151643;

            special_tokens = {
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|caption_start|>",
                "<|caption_end|>",
                "<|lyrics_start|>",
                "<|lyrics_end|>",
                "<|audio_start|>",
                "<|audio_end|>",
            };

            std::string merges_str;
            std::string vocab_str;
            if (!VocabProvider::instance().try_load("minimax_music3.merges", merges_str) ||
                !VocabProvider::instance().try_load("minimax_music3.vocab", vocab_str)) {
                LOG_WARN("MiniMax Music 3 tokenizer data missing (no embedded tokenizer_json in --llm?)");
                return;
            }

            auto byte_unicode_pairs = bytes_to_unicode();
            byte_encoder            = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }

            nlohmann::json vocab;
            try {
                vocab = nlohmann::json::parse(vocab_str);
            } catch (const nlohmann::json::parse_error&) {
                LOG_ERROR("MiniMax Music 3 tokenizer: invalid vocab json");
                return;
            }
            for (const auto& [key, value] : vocab.items()) {
                std::u32string token = utf8_to_utf32(key);
                int i                = value;
                encoder[token]       = i;
                decoder[i]           = token;
            }
            encoder_len = static_cast<int>(encoder.size());
            LOG_DEBUG("minimax_music3 vocab size: %d", encoder_len);

            std::vector<std::u32string> merges = split_utf32(merges_str);
            int rank                           = 0;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                if (space_pos == std::u32string::npos) {
                    continue;
                }
                bpe_ranks[{merge.substr(0, space_pos), merge.substr(space_pos + 1)}] = rank++;
            }
            bpe_len = rank;
            valid   = true;
        }
    };

    /* ---------------------------- prompt building --------------------------- */
    // Ports of the reference clean_caption / normalize_lyrics.

    static inline std::string mm3_strip_markdown_line(std::string line) {
        line = std::regex_replace(line, std::regex(R"(^\s{0,3}#{1,6}\s+)"), "");
        line = std::regex_replace(line, std::regex(R"(^\s*[*+-]\s+)"), "");
        line = std::regex_replace(line, std::regex(R"(^\s*\*\s+)"), "");
        for (;;) {
            auto updated = std::regex_replace(line, std::regex(R"(\*\*([^*]+)\*\*)"), "$1");
            if (updated == line) {
                break;
            }
            line = std::move(updated);
        }
        line = std::regex_replace(line, std::regex(R"(\*([^*\n]+)\*)"), "$1");
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        return line;
    }

    static inline std::string mm3_clean_caption(const std::string& caption) {
        const std::regex special_tag_re(R"(<\|([^|]*)\|>)");
        std::string text;
        std::sregex_iterator it(caption.begin(), caption.end(), special_tag_re);
        std::sregex_iterator end;
        size_t cursor = 0;
        for (; it != end; ++it) {
            const auto& match = *it;
            text.append(caption, cursor, static_cast<size_t>(match.position()) - cursor);
            std::string inner = match.str(1);
            while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.front()))) {
                inner.erase(inner.begin());
            }
            while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.back()))) {
                inner.pop_back();
            }
            const auto split = inner.find_first_of(" \t\n\r\f\v");
            if (split == std::string::npos) {
                text += inner;
            } else {
                size_t rest = split;
                while (rest < inner.size() && std::isspace(static_cast<unsigned char>(inner[rest]))) {
                    ++rest;
                }
                text += inner.substr(0, split) + " is " + inner.substr(rest);
            }
            cursor = static_cast<size_t>(match.position() + match.length());
        }
        text.append(caption, cursor, std::string::npos);
        std::string out;
        size_t start = 0;
        while (start <= text.size()) {
            const size_t line_end = text.find('\n', start);
            auto line             = mm3_strip_markdown_line(text.substr(start, line_end == std::string::npos ? std::string::npos : line_end - start));
            if (!std::regex_match(line, std::regex(R"(^\s*[-*_]{3,}\s*$)"))) {
                if (!out.empty()) {
                    out.push_back('\n');
                }
                out += line;
            }
            if (line_end == std::string::npos) {
                break;
            }
            start = line_end + 1;
        }
        out = std::regex_replace(out, std::regex("\xE2\x80\xA2 "), "");
        out = std::regex_replace(out, std::regex("    "), "");
        return std::regex_replace(out, std::regex(R"(\n{2,})"), "\n");
    }

    static inline std::string mm3_normalize_lyrics(const std::string& lyrics) {
        std::string out;
        size_t start = 0;
        const std::regex leading_tags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
        while (start <= lyrics.size()) {
            const size_t line_end = lyrics.find('\n', start);
            std::string line      = lyrics.substr(start, line_end == std::string::npos ? std::string::npos : line_end - start);
            std::smatch match;
            if (std::regex_search(line, match, leading_tags)) {
                line = match.str(1);
                while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
                    line.pop_back();
                }
            }
            if (!out.empty()) {
                out.push_back('\n');
            }
            out += line;
            if (line_end == std::string::npos) {
                break;
            }
            start = line_end + 1;
        }
        out = std::regex_replace(out, std::regex(R"(\] )"), "]\n");
        out = std::regex_replace(out, std::regex(R"( \[)"), "\n[");
        out = std::regex_replace(out, std::regex(R"( \^ )"), "\n");
        std::string lowered;
        lowered.reserve(out.size());
        for (size_t i = 0; i < out.size();) {
            if (out[i] == '[') {
                const size_t close = out.find(']', i + 1);
                if (close != std::string::npos) {
                    lowered.push_back('[');
                    for (size_t j = i + 1; j < close; ++j) {
                        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(out[j]))));
                    }
                    lowered.push_back(']');
                    i = close + 1;
                    continue;
                }
            }
            lowered.push_back(out[i++]);
        }
        return "[start]\n" + lowered;
    }

    /* --------------------------- global LM + depth -------------------------- */

    // The whole --llm file as one block so the LM and the depth decoder share
    // the (pruned, split) token embeddings.
    struct Music3ArModel : public GGMLBlock {
        Music3Config config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            auto embed_type = [&](const std::string& name) {
                ggml_type t = get_type(prefix + name, tensor_storage_map, GGML_TYPE_F32);
                // ggml_get_rows needs a plain type; quantized embeddings get
                // converted at load time.
                if (t != GGML_TYPE_F32 && t != GGML_TYPE_F16) {
                    t = GGML_TYPE_F16;
                }
                return t;
            };
            auto weight_type = [&](const std::string& name, int64_t in_features) {
                ggml_type t = get_type(prefix + name, tensor_storage_map, GGML_TYPE_F32);
                if (in_features % ggml_blck_size(t) != 0) {
                    t = GGML_TYPE_F32;
                }
                return t;
            };
            auto new_weight = [&](const std::string& name, int64_t in_features, int64_t out_features) {
                params[name] = ggml_new_tensor_2d(ctx, weight_type(name, in_features), in_features, out_features);
            };
            auto new_norm = [&](const std::string& name, int64_t n) {
                params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            };

            params["model.embed_tokens_prefill.weight"] =
                ggml_new_tensor_2d(ctx, embed_type("model.embed_tokens_prefill.weight"), config.hidden_size, config.prefill_vocab);
            params["model.embed_tokens_audio.weight"] =
                ggml_new_tensor_2d(ctx, embed_type("model.embed_tokens_audio.weight"), config.hidden_size, config.semantic_vocab_size);
            params["model.audio_extra_embedding.weight"] =
                ggml_new_tensor_2d(ctx, embed_type("model.audio_extra_embedding.weight"), config.hidden_size,
                                   config.depth_audio_vocab * (config.depth_codebooks - 1));

            for (int64_t i = 0; i < config.layers; i++) {
                const std::string p = "model.layers." + std::to_string(i) + ".";
                new_norm(p + "input_layernorm.weight", config.hidden_size);
                new_weight(p + "self_attn.qkv_proj.weight", config.hidden_size,
                           (config.attention_heads + 2 * config.kv_heads) * config.head_dim);
                new_norm(p + "self_attn.q_norm.weight", config.head_dim);
                new_norm(p + "self_attn.k_norm.weight", config.head_dim);
                new_weight(p + "self_attn.o_proj.weight", config.attention_heads * config.head_dim, config.hidden_size);
                new_norm(p + "post_attention_layernorm.weight", config.hidden_size);
                new_weight(p + "mlp.gate_up_proj.weight", config.hidden_size, 2 * config.intermediate_size);
                new_weight(p + "mlp.down_proj.weight", config.intermediate_size, config.hidden_size);
            }
            new_norm("model.norm.weight", config.hidden_size);
            new_weight("model.lm_head_pruned.weight", config.hidden_size, config.semantic_vocab_size + 1);

            // depth decoder
            new_weight("model.audio_decoder.projection.weight", config.hidden_size, config.hidden_size);
            params["model.audio_decoder.pos_embedding.weight"] =
                ggml_new_tensor_2d(ctx, embed_type("model.audio_decoder.pos_embedding.weight"), config.hidden_size, config.depth_positions);
            for (int64_t i = 0; i < config.depth_layers; i++) {
                const std::string p = "model.audio_decoder.layers." + std::to_string(i) + ".";
                new_norm(p + "input_layernorm.weight", config.hidden_size);
                new_weight(p + "self_attn.qkv_proj.weight", config.hidden_size, 3 * config.hidden_size);
                new_weight(p + "self_attn.o_proj.weight", config.hidden_size, config.hidden_size);
                new_norm(p + "post_attention_layernorm.weight", config.hidden_size);
                new_weight(p + "mlp.gate_up_proj.weight", config.hidden_size, 2 * config.depth_intermediate_size);
                new_weight(p + "mlp.down_proj.weight", config.depth_intermediate_size, config.hidden_size);
            }
            new_norm("model.audio_decoder.norm.weight", config.hidden_size);
            for (int64_t i = 0; i < config.depth_codebooks - 1; i++) {
                new_weight("model.audio_decoder.audio_heads." + std::to_string(i) + ".weight",
                           config.hidden_size, config.depth_audio_vocab);
            }
        }

        ggml_tensor* rms(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& weight_name) {
            x = ggml_rms_norm(ctx->ggml_ctx, x, config.rms_norm_eps);
            return ggml_mul(ctx->ggml_ctx, x, params[weight_name]);
        }

        // x: [hidden, L, B]; returns [hidden, L, B]
        ggml_tensor* lm_layer(GGMLRunnerContext* ctx,
                              ggml_tensor* x,
                              int64_t layer,
                              ggml_tensor* positions,
                              ggml_tensor* mask) {
            auto gctx           = ctx->ggml_ctx;
            const std::string p = "model.layers." + std::to_string(layer) + ".";
            const int64_t L     = x->ne[1];
            const int64_t B     = x->ne[2];
            const int64_t qd    = config.attention_heads * config.head_dim;
            const int64_t kvd   = config.kv_heads * config.head_dim;

            auto h   = rms(ctx, x, p + "input_layernorm.weight");
            auto qkv = ggml_ext_linear(gctx, h, params[p + "self_attn.qkv_proj.weight"], nullptr);
            auto q   = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, 0, qd));
            auto k   = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, qd, qd + kvd));
            auto v   = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, qd + kvd, qd + 2 * kvd));

            q = ggml_reshape_4d(gctx, q, config.head_dim, config.attention_heads, L, B);
            k = ggml_reshape_4d(gctx, k, config.head_dim, config.kv_heads, L, B);

            q = ggml_rms_norm(gctx, q, config.rms_norm_eps);
            q = ggml_mul(gctx, q, params[p + "self_attn.q_norm.weight"]);
            k = ggml_rms_norm(gctx, k, config.rms_norm_eps);
            k = ggml_mul(gctx, k, params[p + "self_attn.k_norm.weight"]);

            q = ggml_rope_ext(gctx, q, positions, nullptr, (int)config.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                              config.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(gctx, k, positions, nullptr, (int)config.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                              config.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            q = ggml_reshape_3d(gctx, q, qd, L, B);
            k = ggml_reshape_3d(gctx, k, kvd, L, B);

            // KV cache: append this step, keep the whole thing for the next one
            const std::string k_name = "mm3_k_" + std::to_string(layer);
            const std::string v_name = "mm3_v_" + std::to_string(layer);
            ggml_tensor* k_prev      = ctx->load_cache_tensor(k_name);
            ggml_tensor* v_prev      = ctx->load_cache_tensor(v_name);
            ggml_tensor* k_all       = k_prev != nullptr ? ggml_concat(gctx, k_prev, k, 1) : k;
            ggml_tensor* v_all       = v_prev != nullptr ? ggml_concat(gctx, v_prev, v, 1) : v;
            ctx->persist_cache_tensor(k_name, k_all);
            ctx->persist_cache_tensor(v_name, v_all);

            auto attn = ggml_ext_attention_ext(gctx, ctx->backend, q, k_all, v_all,
                                               config.attention_heads, mask, false, ctx->flash_attn_enabled);
            attn      = ggml_ext_linear(gctx, attn, params[p + "self_attn.o_proj.weight"], nullptr);
            x         = ggml_add(gctx, x, attn);

            h         = rms(ctx, x, p + "post_attention_layernorm.weight");
            auto gu   = ggml_ext_linear(gctx, h, params[p + "mlp.gate_up_proj.weight"], nullptr);
            auto gate = ggml_ext_slice(gctx, gu, 0, 0, config.intermediate_size);
            auto up   = ggml_ext_slice(gctx, gu, 0, config.intermediate_size, 2 * config.intermediate_size);
            auto act  = ggml_mul(gctx, ggml_ext_cont(gctx, up), ggml_silu(gctx, ggml_ext_cont(gctx, gate)));
            auto down = ggml_ext_linear(gctx, act, params[p + "mlp.down_proj.weight"], nullptr);
            return ggml_add(gctx, x, down);
        }

        // x: [hidden, L, B=2]; positions: [L]; mask: [L_kv, L] or nullptr.
        // Returns concat(hidden_last [hidden, 1, B], logits_last [compact, 1, B])
        // along dim 0 -> [hidden + compact, 1, B].
        ggml_tensor* lm_forward(GGMLRunnerContext* ctx,
                                ggml_tensor* x,
                                ggml_tensor* positions,
                                ggml_tensor* mask) {
            auto gctx = ctx->ggml_ctx;
            for (int64_t i = 0; i < config.layers; i++) {
                x = lm_layer(ctx, x, i, positions, mask);
            }
            x               = rms(ctx, x, "model.norm.weight");
            const int64_t L = x->ne[1];
            auto last       = ggml_ext_cont(gctx, ggml_ext_slice(gctx, x, 1, L - 1, L));  // [hidden, 1, B]
            auto logits     = ggml_ext_linear(gctx, last, params["model.lm_head_pruned.weight"], nullptr);
            return ggml_concat(gctx, last, logits, 0);
        }

        // ids: [n] I32 over the *virtual* id space; audio ids are looked up in
        // the audio table, everything else in the prefill table. The caller
        // guarantees each id lands in the right range for `audio`.
        ggml_tensor* embed(GGMLRunnerContext* ctx, ggml_tensor* ids, bool audio) {
            auto gctx = ctx->ggml_ctx;
            auto w    = params[audio ? "model.embed_tokens_audio.weight" : "model.embed_tokens_prefill.weight"];
            return ggml_get_rows(gctx, w, ids);  // [hidden, n]
        }

        // depth decoder step for one codebook.
        // last_hidden: [hidden, rows], semantic_id: [1] (0..16383),
        // residual_ids: [n_res] into audio_extra_embedding or nullptr,
        // positions: [steps], mask: [steps, steps].
        // Returns concat(hidden_last [hidden, 1, rows], logits [audio_vocab, 1, rows]).
        ggml_tensor* depth_forward(GGMLRunnerContext* ctx,
                                   ggml_tensor* last_hidden,
                                   ggml_tensor* semantic_id,
                                   ggml_tensor* residual_ids,
                                   ggml_tensor* positions,
                                   ggml_tensor* mask,
                                   int64_t codebook) {
            auto gctx          = ctx->ggml_ctx;
            const int64_t rows = last_hidden->ne[1];
            const int64_t hd   = config.hidden_size;
            auto proj_w        = params["model.audio_decoder.projection.weight"];

            auto hidden0 = ggml_ext_linear(gctx, last_hidden, proj_w, nullptr);      // [hidden, rows]
            hidden0      = ggml_reshape_3d(gctx, hidden0, hd, 1, rows);              // [hidden, 1, rows]

            auto semantic = embed(ctx, semantic_id, true);                           // [hidden, 1]
            semantic      = ggml_ext_linear(gctx, semantic, proj_w, nullptr);
            semantic      = ggml_reshape_3d(gctx, semantic, hd, 1, 1);
            semantic      = ggml_repeat_4d(gctx, semantic, hd, 1, rows, 1);

            auto seq = ggml_concat(gctx, hidden0, semantic, 1);                      // [hidden, 2, rows]
            if (residual_ids != nullptr) {
                auto residual = ggml_get_rows(gctx, params["model.audio_extra_embedding.weight"], residual_ids);
                residual      = ggml_ext_linear(gctx, residual, proj_w, nullptr);    // [hidden, n_res]
                residual      = ggml_reshape_3d(gctx, residual, hd, residual->ne[1], 1);
                residual      = ggml_repeat_4d(gctx, residual, hd, residual->ne[1], rows, 1);
                seq           = ggml_concat(gctx, seq, residual, 1);
            }
            const int64_t steps = seq->ne[1];

            auto pos_emb = ggml_get_rows(gctx, params["model.audio_decoder.pos_embedding.weight"], positions);  // [hidden, steps]
            pos_emb      = ggml_reshape_3d(gctx, pos_emb, hd, steps, 1);
            seq          = ggml_add(gctx, seq, pos_emb);

            for (int64_t i = 0; i < config.depth_layers; i++) {
                const std::string p = "model.audio_decoder.layers." + std::to_string(i) + ".";
                auto h              = rms(ctx, seq, p + "input_layernorm.weight");
                auto qkv            = ggml_ext_linear(gctx, h, params[p + "self_attn.qkv_proj.weight"], nullptr);
                auto q              = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, 0, hd));
                auto k              = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, hd, 2 * hd));
                auto v              = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, 2 * hd, 3 * hd));
                auto attn           = ggml_ext_attention_ext(gctx, ctx->backend, q, k, v,
                                                             config.depth_heads, mask, false, ctx->flash_attn_enabled);
                attn                = ggml_ext_linear(gctx, attn, params[p + "self_attn.o_proj.weight"], nullptr);
                seq                 = ggml_add(gctx, seq, attn);

                h         = rms(ctx, seq, p + "post_attention_layernorm.weight");
                auto gu   = ggml_ext_linear(gctx, h, params[p + "mlp.gate_up_proj.weight"], nullptr);
                auto gate = ggml_ext_slice(gctx, gu, 0, 0, config.depth_intermediate_size);
                auto up   = ggml_ext_slice(gctx, gu, 0, config.depth_intermediate_size, 2 * config.depth_intermediate_size);
                auto act  = ggml_mul(gctx, ggml_ext_cont(gctx, up), ggml_silu(gctx, ggml_ext_cont(gctx, gate)));
                auto down = ggml_ext_linear(gctx, act, params[p + "mlp.down_proj.weight"], nullptr);
                seq       = ggml_add(gctx, seq, down);
            }

            seq       = rms(ctx, seq, "model.audio_decoder.norm.weight");
            auto last = ggml_ext_cont(gctx, ggml_ext_slice(gctx, seq, 1, steps - 1, steps));  // [hidden, 1, rows]
            auto head = params["model.audio_decoder.audio_heads." + std::to_string(codebook - 1) + ".weight"];
            auto lg   = ggml_ext_linear(gctx, last, head, nullptr);                            // [audio_vocab, 1, rows]
            return ggml_concat(gctx, last, lg, 0);
        }

        // codes: semantic id [1] + residual ids [codebooks-1] -> [hidden]
        ggml_tensor* feedback_forward(GGMLRunnerContext* ctx,
                                      ggml_tensor* semantic_id,
                                      ggml_tensor* residual_ids) {
            auto gctx     = ctx->ggml_ctx;
            auto semantic = embed(ctx, semantic_id, true);  // [hidden, 1]
            auto residual = ggml_get_rows(gctx, params["model.audio_extra_embedding.weight"], residual_ids);  // [hidden, n]
            auto residual_t = ggml_ext_cont(gctx, ggml_transpose(gctx, residual));   // [n, hidden]
            auto res_sum    = ggml_sum_rows(gctx, residual_t);                        // [1, hidden]
            res_sum         = ggml_reshape_2d(gctx, res_sum, config.hidden_size, 1);  // [hidden, 1]
            auto out        = ggml_add(gctx, ggml_reshape_2d(gctx, semantic, config.hidden_size, 1), res_sum);
            return ggml_ext_scale(gctx, out, 1.0f / std::sqrt((float)config.depth_codebooks));
        }
    };

    struct Music3ArStepOutput {
        // compact logits: semantic_vocab + 1 entries per row (cond, uncond)
        std::vector<float> logits;
        std::vector<float> cond_hidden;
        std::vector<float> uncond_hidden;
    };

    struct Music3ArRunner : public GGMLRunner {
        Music3Config config;
        Music3ArModel model;
        std::string weight_prefix;
        int64_t cache_position = 0;  // next position index in the KV cache

        Music3ArRunner(ggml_backend_t backend,
                       const String2TensorStorage& tensor_storage_map,
                       const std::string& prefix                           = "text_encoders.llm",
                       std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : GGMLRunner(backend, weight_manager),
              weight_prefix(prefix) {
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "minimax_music3_ar";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
            model.get_param_tensors(tensors, weight_prefix);
        }

        void reset_kv_cache() {
            free_cache_ctx_and_buffer();
            cache_position = 0;
        }

        static void split_step_output(const sd::Tensor<float>& out,
                                      int64_t hidden,
                                      int64_t logits_size,
                                      Music3ArStepOutput& step) {
            const int64_t width = hidden + logits_size;
            GGML_ASSERT(out.numel() == 2 * width);
            const float* d = out.data();
            step.cond_hidden.assign(d, d + hidden);
            step.logits.resize(2 * logits_size);
            std::copy(d + hidden, d + width, step.logits.begin());
            step.uncond_hidden.assign(d + width, d + width + hidden);
            std::copy(d + width + hidden, d + 2 * width, step.logits.begin() + logits_size);
        }

        // prompt ids for both branches: cond_ids and uncond_ids, same length.
        bool prefill(int n_threads,
                     const std::vector<int32_t>& cond_ids,
                     const std::vector<int32_t>& uncond_ids,
                     Music3ArStepOutput& step) {
            const int64_t L = (int64_t)cond_ids.size();
            GGML_ASSERT(L > 0 && cond_ids.size() == uncond_ids.size());
            reset_kv_cache();

            std::vector<int32_t> ids(cond_ids);
            ids.insert(ids.end(), uncond_ids.begin(), uncond_ids.end());
            sd::Tensor<int32_t> ids_tensor({2 * L}, ids);

            std::vector<int32_t> pos(L);
            for (int64_t i = 0; i < L; i++) {
                pos[i] = (int32_t)i;
            }
            sd::Tensor<int32_t> pos_tensor({L}, pos);

            sd::Tensor<float> mask_tensor({L, L});
            for (int64_t q = 0; q < L; q++) {
                for (int64_t k = 0; k < L; k++) {
                    mask_tensor.data()[q * L + k] = k <= q ? 0.0f : -INFINITY;
                }
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(SD_MM3_LM_GRAPH_SIZE);
                auto ids_in     = make_input(ids_tensor);
                auto pos_in     = make_input(pos_tensor);
                auto mask_in    = make_input(mask_tensor);
                auto runner_ctx = get_context();
                auto x          = model.embed(&runner_ctx, ids_in, false);  // [hidden, 2L]
                x               = ggml_reshape_3d(compute_ctx, x, config.hidden_size, L, 2);
                auto out        = model.lm_forward(&runner_ctx, x, pos_in, mask_in);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            cache_position = L;
            split_step_output(*result, config.hidden_size, config.semantic_vocab_size + 1, step);
            return true;
        }

        // one decode step; feedback: [hidden] embedding fed to both branches.
        bool decode(int n_threads,
                    const std::vector<float>& feedback,
                    Music3ArStepOutput& step) {
            GGML_ASSERT((int64_t)feedback.size() == config.hidden_size);
            std::vector<float> x(feedback);
            x.insert(x.end(), feedback.begin(), feedback.end());
            sd::Tensor<float> x_tensor({config.hidden_size, 1, 2}, x);
            sd::Tensor<int32_t> pos_tensor({1}, std::vector<int32_t>{(int32_t)cache_position});

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(SD_MM3_LM_GRAPH_SIZE);
                auto x_in       = make_input(x_tensor);
                auto pos_in     = make_input(pos_tensor);
                auto runner_ctx = get_context();
                auto out        = model.lm_forward(&runner_ctx, x_in, pos_in, nullptr);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            cache_position += 1;
            split_step_output(*result, config.hidden_size, config.semantic_vocab_size + 1, step);
            return true;
        }

        // one depth codebook step at batch 2 (cond, uncond).
        // residual_codes are raw codes (0..1023) for codebooks 1..codebook-1.
        bool depth_step(int n_threads,
                        const std::vector<float>& cond_hidden,
                        const std::vector<float>& uncond_hidden,
                        int32_t semantic_code,
                        const std::vector<int32_t>& residual_codes,
                        int64_t codebook,
                        std::vector<float>& cond_hidden_out,
                        std::vector<float>& cond_logits,
                        std::vector<float>& uncond_logits) {
            const int64_t hd    = config.hidden_size;
            const int64_t steps = codebook + 1;

            std::vector<float> hiddens(cond_hidden);
            hiddens.insert(hiddens.end(), uncond_hidden.begin(), uncond_hidden.end());
            sd::Tensor<float> hidden_tensor({hd, 2}, hiddens);
            sd::Tensor<int32_t> semantic_tensor({1}, std::vector<int32_t>{semantic_code});

            std::vector<int32_t> res_ids;
            for (size_t i = 0; i < residual_codes.size(); i++) {
                res_ids.push_back(residual_codes[i] + (int32_t)(i * config.depth_audio_vocab));
            }
            sd::Tensor<int32_t> res_tensor;
            if (!res_ids.empty()) {
                res_tensor = sd::Tensor<int32_t>({(int64_t)res_ids.size()}, res_ids);
            }

            std::vector<int32_t> pos(steps);
            for (int64_t i = 0; i < steps; i++) {
                pos[i] = (int32_t)i;
            }
            sd::Tensor<int32_t> pos_tensor({steps}, pos);

            sd::Tensor<float> mask_tensor({steps, steps});
            for (int64_t q = 0; q < steps; q++) {
                for (int64_t k = 0; k < steps; k++) {
                    mask_tensor.data()[q * steps + k] = k <= q ? 0.0f : -INFINITY;
                }
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(SD_MM3_DEPTH_GRAPH_SIZE);
                auto hidden_in   = make_input(hidden_tensor);
                auto semantic_in = make_input(semantic_tensor);
                auto res_in      = make_optional_input(res_tensor);
                auto pos_in      = make_input(pos_tensor);
                auto mask_in     = make_input(mask_tensor);
                auto runner_ctx  = get_context();
                auto out         = model.depth_forward(&runner_ctx, hidden_in, semantic_in, res_in,
                                                       pos_in, mask_in, codebook);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            const int64_t width = hd + config.depth_audio_vocab;
            GGML_ASSERT(result->numel() == 2 * width);
            const float* d = result->data();
            cond_hidden_out.assign(d, d + hd);
            cond_logits.assign(d + hd, d + width);
            uncond_logits.assign(d + width + hd, d + 2 * width);
            return true;
        }

        bool feedback_embedding(int n_threads,
                                const std::vector<int32_t>& codes,  // [codebooks]: semantic + residuals
                                std::vector<float>& out) {
            GGML_ASSERT((int64_t)codes.size() == config.depth_codebooks);
            sd::Tensor<int32_t> semantic_tensor({1}, std::vector<int32_t>{codes[0]});
            std::vector<int32_t> res_ids;
            for (int64_t i = 1; i < config.depth_codebooks; i++) {
                res_ids.push_back(codes[i] + (int32_t)((i - 1) * config.depth_audio_vocab));
            }
            sd::Tensor<int32_t> res_tensor({(int64_t)res_ids.size()}, res_ids);

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf  = new_graph_custom(4096);
                auto semantic_in = make_input(semantic_tensor);
                auto res_in      = make_input(res_tensor);
                auto runner_ctx  = get_context();
                auto result      = model.feedback_forward(&runner_ctx, semantic_in, res_in);
                ggml_build_forward_expand(gf, result);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            out.assign(result->data(), result->data() + result->numel());
            return true;
        }

        static const int SD_MM3_LM_GRAPH_SIZE    = 8192;
        static const int SD_MM3_DEPTH_GRAPH_SIZE = 2048;
    };

    /* ----------------------------- flow transformer -------------------------- */

    struct Music3FlowModel : public GGMLBlock {
        Music3Config config;

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensor_storage_map = {},
                         const std::string prefix                       = "") override {
            const int64_t inner = config.flow_heads * config.flow_head_dim;  // 2048
            const int64_t cin   = 2 * config.flow_in_channels + config.condition_dim;  // 2304

            auto weight_type = [&](const std::string& name, int64_t in_features) {
                ggml_type t = get_type(prefix + name, tensor_storage_map, GGML_TYPE_F32);
                if (in_features % ggml_blck_size(t) != 0) {
                    t = GGML_TYPE_F32;
                }
                return t;
            };
            auto new_weight = [&](const std::string& name, int64_t in_features, int64_t out_features) {
                params[name] = ggml_new_tensor_2d(ctx, weight_type(name, in_features), in_features, out_features);
            };
            auto new_f32 = [&](const std::string& name, int64_t n) {
                params[name] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            };

            // condition encoder
            new_f32("cond_layer_logits", config.condition_layers);
            new_f32("cond_layer_scale", 1);
            params["latent_conditioners.0.weight"] =
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, config.hidden_size, config.condition_dim);
            new_f32("latent_conditioners.0.bias", config.condition_dim);

            // flow transformer
            params["diffusion_transformer.timestep_features.weight"] =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, config.flow_fourier_dim / 2);
            new_weight("diffusion_transformer.to_timestep_embed.0.weight", config.flow_fourier_dim, inner);
            new_f32("diffusion_transformer.to_timestep_embed.0.bias", inner);
            new_weight("diffusion_transformer.to_timestep_embed.2.weight", inner, inner);
            new_f32("diffusion_transformer.to_timestep_embed.2.bias", inner);
            params["diffusion_transformer.preprocess_conv.weight"] =
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, cin, cin);
            new_weight("diffusion_transformer.transformer.project_in.weight", cin, inner);
            for (int64_t i = 0; i < config.flow_layers; i++) {
                const std::string p = "diffusion_transformer.transformer.layers." + std::to_string(i) + ".";
                new_f32(p + "pre_norm.gamma", inner);
                new_f32(p + "pre_norm.beta", inner);
                new_weight(p + "self_attn.to_qkv.weight", inner, 3 * inner);
                new_weight(p + "self_attn.to_out.weight", inner, inner);
                new_f32(p + "ff_norm.gamma", inner);
                new_f32(p + "ff_norm.beta", inner);
                new_weight(p + "ff.ff.0.proj.weight", inner, 2 * config.flow_ff_inner);
                new_f32(p + "ff.ff.0.proj.bias", 2 * config.flow_ff_inner);
                new_weight(p + "ff.ff.2.weight", config.flow_ff_inner, inner);
                new_f32(p + "ff.ff.2.bias", inner);
            }
            new_weight("diffusion_transformer.transformer.project_out.weight", inner, config.flow_in_channels);
            params["diffusion_transformer.postprocess_conv.weight"] =
                ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, config.flow_in_channels, config.flow_in_channels);
            new_f32("diffusion_transformer.transformer.rotary_pos_emb.inv_freq", config.flow_rotary_dim / 2);
        }

        ggml_tensor* host_param(const std::string& name) {
            return params[name];
        }

        ggml_tensor* layer_norm(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& base) {
            return ggml_ext_layer_norm(ctx->ggml_ctx, x, params[base + ".gamma"], params[base + ".beta"], 1.0e-5f);
        }

        // conv over the frames axis with kernel 3, padding 1, on [frames, hidden, 1]
        ggml_tensor* condition_conv(GGMLRunnerContext* ctx, ggml_tensor* mixed) {
            auto gctx = ctx->ggml_ctx;
            auto out  = ggml_conv_1d(gctx, params["latent_conditioners.0.weight"], mixed, 1, 1, 1);
            out       = ggml_add(gctx, out,
                                 ggml_reshape_3d(gctx, params["latent_conditioners.0.bias"], 1, config.condition_dim, 1));
            return out;  // [frames, condition_dim, 1]
        }

        // latents: [frames, 128, B], condition: [frames, 2048, B],
        // time_features: [256, B], positions: [frames + 1] I32.
        ggml_tensor* flow_forward(GGMLRunnerContext* ctx,
                                  ggml_tensor* latents,
                                  ggml_tensor* condition,
                                  ggml_tensor* time_features,
                                  ggml_tensor* positions) {
            auto gctx            = ctx->ggml_ctx;
            const int64_t frames = latents->ne[0];
            const int64_t B      = latents->ne[2];
            const int64_t inner  = config.flow_heads * config.flow_head_dim;
            const int64_t cin    = 2 * config.flow_in_channels + config.condition_dim;
            const int64_t steps  = frames + 1;

            // concat channels: [latents, zeros(latents), condition]
            auto zeros = ggml_ext_scale(gctx, latents, 0.0f);
            auto x     = ggml_concat(gctx, latents, zeros, 1);
            x          = ggml_concat(gctx, x, condition, 1);  // [frames, 2304, B]

            // preprocess 1x1 conv + residual, as a linear over channels
            auto tokens = ggml_ext_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));  // [2304, frames, B]
            auto pre_w  = ggml_reshape_2d(gctx, params["diffusion_transformer.preprocess_conv.weight"], cin, cin);
            auto pre    = ggml_mul_mat(gctx, pre_w, tokens);
            tokens      = ggml_add(gctx, pre, tokens);

            tokens = ggml_ext_linear(gctx, tokens, params["diffusion_transformer.transformer.project_in.weight"], nullptr);  // [2048, frames, B]

            auto temb = ggml_ext_linear(gctx, time_features,
                                        params["diffusion_transformer.to_timestep_embed.0.weight"],
                                        params["diffusion_transformer.to_timestep_embed.0.bias"]);
            temb      = ggml_silu(gctx, temb);
            temb      = ggml_ext_linear(gctx, temb,
                                        params["diffusion_transformer.to_timestep_embed.2.weight"],
                                        params["diffusion_transformer.to_timestep_embed.2.bias"]);
            temb      = ggml_reshape_3d(gctx, temb, inner, 1, B);
            auto seq  = ggml_concat(gctx, temb, tokens, 1);  // [2048, steps, B]

            for (int64_t i = 0; i < config.flow_layers; i++) {
                const std::string p = "diffusion_transformer.transformer.layers." + std::to_string(i) + ".";

                auto normed = layer_norm(ctx, seq, p + "pre_norm");
                auto qkv    = ggml_ext_linear(gctx, normed, params[p + "self_attn.to_qkv.weight"], nullptr);
                auto q      = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, 0, inner));
                auto k      = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, inner, 2 * inner));
                auto v      = ggml_ext_cont(gctx, ggml_ext_slice(gctx, qkv, 0, 2 * inner, 3 * inner));

                q = ggml_reshape_4d(gctx, q, config.flow_head_dim, config.flow_heads, steps, B);
                k = ggml_reshape_4d(gctx, k, config.flow_head_dim, config.flow_heads, steps, B);
                q = ggml_rope_ext(gctx, q, positions, nullptr, (int)config.flow_rotary_dim, GGML_ROPE_TYPE_NEOX, 0,
                                  config.flow_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(gctx, k, positions, nullptr, (int)config.flow_rotary_dim, GGML_ROPE_TYPE_NEOX, 0,
                                  config.flow_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                q = ggml_reshape_3d(gctx, q, inner, steps, B);
                k = ggml_reshape_3d(gctx, k, inner, steps, B);

                auto attn = ggml_ext_attention_ext(gctx, ctx->backend, q, k, v,
                                                   config.flow_heads, nullptr, false, ctx->flash_attn_enabled);
                attn      = ggml_ext_linear(gctx, attn, params[p + "self_attn.to_out.weight"], nullptr);
                seq       = ggml_add(gctx, seq, attn);

                auto ffn    = layer_norm(ctx, seq, p + "ff_norm");
                ffn         = ggml_ext_linear(gctx, ffn, params[p + "ff.ff.0.proj.weight"], params[p + "ff.ff.0.proj.bias"]);
                auto states = ggml_ext_slice(gctx, ffn, 0, 0, config.flow_ff_inner);
                auto gate   = ggml_ext_slice(gctx, ffn, 0, config.flow_ff_inner, 2 * config.flow_ff_inner);
                auto gated  = ggml_mul(gctx, ggml_ext_cont(gctx, states), ggml_silu(gctx, ggml_ext_cont(gctx, gate)));
                gated       = ggml_ext_linear(gctx, gated, params[p + "ff.ff.2.weight"], params[p + "ff.ff.2.bias"]);
                seq         = ggml_add(gctx, seq, gated);
            }

            seq = ggml_ext_cont(gctx, ggml_ext_slice(gctx, seq, 1, 1, steps));  // drop the temb token
            seq = ggml_ext_linear(gctx, seq, params["diffusion_transformer.transformer.project_out.weight"], nullptr);  // [128, frames, B]

            auto post_w = ggml_reshape_2d(gctx, params["diffusion_transformer.postprocess_conv.weight"],
                                          config.flow_in_channels, config.flow_in_channels);
            auto post   = ggml_mul_mat(gctx, post_w, seq);
            seq         = ggml_add(gctx, post, seq);

            seq = ggml_ext_cont(gctx, ggml_permute(gctx, seq, 1, 0, 2, 3));  // [frames, 128, B]
            return seq;
        }
    };

    struct Music3FlowRunner : public DiffusionModelRunner {
        Music3Config config;
        Music3FlowModel model;

        // read back once after weights are loaded
        bool host_params_ready = false;
        std::vector<float> layer_weights;   // softmax(cond_layer_logits) * cond_layer_scale
        std::vector<float> time_proj;       // timestep_features.weight [fourier/2]

        Music3FlowRunner(ggml_backend_t backend,
                         const String2TensorStorage& tensor_storage_map      = {},
                         const std::string prefix                            = "",
                         std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager) {
            model.init(params_ctx, tensor_storage_map, this->prefix);
        }

        std::string get_desc() override {
            return "minimax_music3_flow";
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        // required by the base class; the music3 pipeline drives the runner
        // through predict_velocity instead.
        sd::Tensor<float> compute(int n_threads, const DiffusionParams& diffusion_params) override {
            SD_UNUSED(n_threads);
            SD_UNUSED(diffusion_params);
            LOG_ERROR("minimax_music3 flow runner must be driven by the music3 pipeline");
            return {};
        }

        bool ensure_host_params(int n_threads) {
            if (host_params_ready) {
                return true;
            }
            // the params live behind the weight manager and are only bound to
            // backend memory during a compute, so read them back via a tiny
            // graph rather than poking the param tensors directly
            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(64);
                auto runner_ctx = get_context();
                auto gctx       = runner_ctx.ggml_ctx;
                auto out        = ggml_concat(gctx, model.host_param("cond_layer_logits"),
                                              model.host_param("cond_layer_scale"), 0);
                auto time_flat  = ggml_reshape_1d(gctx, model.host_param("diffusion_transformer.timestep_features.weight"),
                                                  config.flow_fourier_dim / 2);
                out             = ggml_concat(gctx, out, ggml_ext_cont(gctx, time_flat), 0);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            GGML_ASSERT(result->numel() == config.condition_layers + 1 + config.flow_fourier_dim / 2);
            const float* d = result->data();
            std::vector<float> logits(d, d + config.condition_layers);
            float scale = d[config.condition_layers];
            time_proj.assign(d + config.condition_layers + 1, d + result->numel());

            float max_value = *std::max_element(logits.begin(), logits.end());
            double sum      = 0.0;
            for (float& v : logits) {
                v = std::exp(v - max_value);
                sum += v;
            }
            layer_weights.resize(logits.size());
            for (size_t i = 0; i < logits.size(); i++) {
                layer_weights[i] = (float)((double)logits[i] / sum) * scale;
            }
            LOG_DEBUG("mm3 condition layer weights ready (scale %.4f)", scale);
            host_params_ready = true;
            return true;
        }

        // frame_hiddens: [frames * condition_layers * hidden] (frame-major),
        // returns condition values, channel-major [condition_dim * out_frames]
        // (t fastest), already interpolated to the latent frame rate.
        bool encode_condition(int n_threads,
                              const float* frame_hiddens,
                              int64_t frames,
                              int64_t& out_frames,
                              std::vector<float>& condition) {
            if (!ensure_host_params(n_threads)) {
                return false;
            }
            const int64_t layers = config.condition_layers;
            const int64_t hd     = config.hidden_size;

            out_frames = (int64_t)((double)frames * config.condition_ratio);
            if (out_frames <= 0) {
                return false;
            }

            // weighted layer mix -> [frames, hidden] channel-major (t fastest)
            sd::Tensor<float> mixed({frames, hd, 1});
            std::fill(mixed.data(), mixed.data() + mixed.numel(), 0.0f);
            for (int64_t frame = 0; frame < frames; frame++) {
                for (int64_t layer = 0; layer < layers; layer++) {
                    const float w        = layer_weights[layer];
                    const float* src     = frame_hiddens + (frame * layers + layer) * hd;
                    float* dst           = mixed.data() + frame;
                    for (int64_t c = 0; c < hd; c++) {
                        dst[c * frames] += src[c] * w;
                    }
                }
            }

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf = new_graph_custom(1024);
                auto mixed_in   = make_input(mixed);
                auto runner_ctx = get_context();
                auto out        = model.condition_conv(&runner_ctx, mixed_in);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto projected = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!projected.has_value()) {
                return false;
            }
            // nearest interpolation frames -> out_frames along t, per channel
            const float* proj = projected->data();  // [frames, condition_dim]
            condition.assign((size_t)(config.condition_dim * out_frames), 0.0f);
            for (int64_t t = 0; t < out_frames; t++) {
                int64_t src = (int64_t)((double)t * (double)frames / (double)out_frames);
                if (src >= frames) {
                    src = frames - 1;
                }
                for (int64_t c = 0; c < config.condition_dim; c++) {
                    condition[(size_t)(c * out_frames + t)] = proj[c * frames + src];
                }
            }
            return true;
        }

        // latents: [in_channels * frames] channel-major, condition likewise
        // [condition_dim * frames]; returns velocity for cond (+ uncond when
        // batch == 2) in the same layout, cond first.
        bool predict_velocity(int n_threads,
                              const std::vector<float>& latents,
                              const std::vector<float>& condition,
                              int64_t frames,
                              float timestep,
                              int64_t batch,
                              std::vector<float>& velocity) {
            if (!ensure_host_params(n_threads)) {
                return false;
            }
            const int64_t C  = config.flow_in_channels;
            const int64_t CD = config.condition_dim;
            GGML_ASSERT((int64_t)latents.size() == C * frames);
            GGML_ASSERT((int64_t)condition.size() == CD * frames);
            GGML_ASSERT(batch == 1 || batch == 2);

            std::vector<float> latent_batch(latents);
            if (batch == 2) {
                latent_batch.insert(latent_batch.end(), latents.begin(), latents.end());
            }
            sd::Tensor<float> latents_tensor({frames, C, batch}, latent_batch);

            std::vector<float> condition_batch(condition);
            if (batch == 2) {
                condition_batch.resize((size_t)(2 * CD * frames), 0.0f);  // null condition = zeros
            }
            sd::Tensor<float> condition_tensor({frames, CD, batch}, condition_batch);

            // fourier features: [cos(2 pi t w); sin(2 pi t w)]
            const int64_t half = config.flow_fourier_dim / 2;
            std::vector<float> tf((size_t)(config.flow_fourier_dim * batch));
            for (int64_t i = 0; i < half; i++) {
                const float angle = 6.2831853071795864769f * timestep * time_proj[i];
                tf[(size_t)i]        = std::cos(angle);
                tf[(size_t)(half + i)] = std::sin(angle);
            }
            if (batch == 2) {
                std::copy(tf.begin(), tf.begin() + config.flow_fourier_dim, tf.begin() + config.flow_fourier_dim);
            }
            sd::Tensor<float> time_tensor({config.flow_fourier_dim, batch}, tf);

            std::vector<int32_t> pos(frames + 1);
            for (int64_t i = 0; i <= frames; i++) {
                pos[i] = (int32_t)i;
            }
            sd::Tensor<int32_t> pos_tensor({frames + 1}, pos);

            auto get_graph = [&]() -> ggml_cgraph* {
                ggml_cgraph* gf   = new_graph_custom(SD_MM3_FLOW_GRAPH_SIZE);
                auto latents_in   = make_input(latents_tensor);
                auto condition_in = make_input(condition_tensor);
                auto time_in      = make_input(time_tensor);
                auto pos_in       = make_input(pos_tensor);
                auto runner_ctx   = get_context();
                auto out          = model.flow_forward(&runner_ctx, latents_in, condition_in, time_in, pos_in);
                ggml_build_forward_expand(gf, out);
                return gf;
            };
            auto result = GGMLRunner::compute<float>(get_graph, n_threads, false, false, false);
            if (!result.has_value()) {
                return false;
            }
            GGML_ASSERT(result->numel() == batch * C * frames);
            velocity.assign(result->data(), result->data() + result->numel());
            return true;
        }

        static const int SD_MM3_FLOW_GRAPH_SIZE = 8192;
    };

}  // namespace MiniMaxMusic3

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_MUSIC3_HPP__
