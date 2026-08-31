#ifndef __SD_MODEL_DIFFUSION_ACE_STEP_LYRIC_HPP__
#define __SD_MODEL_DIFFUSION_ACE_STEP_LYRIC_HPP__

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

// Ref: https://github.com/ace-step/ACE-Step
//   acestep/models/lyrics_utils/lyric_tokenizer.py (VoiceBpeTokenizer)
//   acestep/pipeline_ace_step.py (tokenize_lyrics)
//
// The XTTS-style voice-BPE lyric tokenizer: plain (non byte-level) BPE over a
// 6681-entry vocab with a Whitespace pre-tokenizer, [UNK] fallback, and
// language/structure tags as atomic added tokens. tokenize_lyrics feeds one
// line at a time: [START] first, every line followed by [SPACE] (id 2), empty
// lines contribute just the [SPACE].
//
// Only the English cleaner path is ported (lowercase, number/abbreviation/
// symbol expansion). The reference detects the language per line and
// transliterates zh/ja/ko through pypinyin/cutlet/hangul-romanize; here every
// line takes the [en] path, so non-Latin scripts mostly fall back to [UNK]
// and will not sing correctly.
namespace AceStep {

#include "ace_step_lyric_vocab.inc"

    /* --------------------------- english cleaners --------------------------- */

    // num2words-en style cardinal: "fifty-five", "one hundred and one",
    // "one thousand, two hundred and thirty-four", "two thousand and twenty-three".
    inline std::string ace_number_to_words(long long n) {
        static const char* kUnits[] = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                                       "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                                       "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        static const char* kTens[]  = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                       "eighty", "ninety"};
        if (n < 0) {
            return "minus " + ace_number_to_words(-n);
        }
        if (n < 20) {
            return kUnits[n];
        }
        if (n < 100) {
            std::string out = kTens[n / 10];
            if (n % 10) {
                out += "-";
                out += kUnits[n % 10];
            }
            return out;
        }
        if (n < 1000) {
            std::string out = std::string(kUnits[n / 100]) + " hundred";
            if (n % 100) {
                out += " and " + ace_number_to_words(n % 100);
            }
            return out;
        }
        static const std::pair<long long, const char*> kScales[] = {
            {1000000000000LL, "trillion"},
            {1000000000LL, "billion"},
            {1000000LL, "million"},
            {1000LL, "thousand"},
        };
        std::string out;
        for (const auto& scale : kScales) {
            if (n >= scale.first) {
                long long count = n / scale.first;
                n               = n % scale.first;
                if (!out.empty()) {
                    out += ", ";
                }
                out += ace_number_to_words(count) + " " + scale.second;
            }
        }
        if (n > 0) {
            // num2words joins a trailing sub-hundred part with "and"
            out += n < 100 ? " and " : ", ";
            out += ace_number_to_words(n);
        }
        return out;
    }

    inline std::string ace_number_to_ordinal_words(long long n) {
        static const std::unordered_map<std::string, std::string> kIrregular = {
            {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
            {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
        std::string words = ace_number_to_words(n);
        // transform the last word (after the last space or hyphen)
        size_t pos       = words.find_last_of(" -");
        size_t start     = pos == std::string::npos ? 0 : pos + 1;
        std::string last = words.substr(start);
        auto it          = kIrregular.find(last);
        if (it != kIrregular.end()) {
            last = it->second;
        } else if (!last.empty() && last.back() == 'y') {
            last = last.substr(0, last.size() - 1) + "ieth";
        } else {
            last += "th";
        }
        return words.substr(0, start) + last;
    }

    inline std::string ace_replace_matches(const std::string& text,
                                           const std::regex& re,
                                           const std::function<std::string(const std::smatch&)>& fn) {
        std::string out;
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end   = std::sregex_iterator();
        size_t pos = 0;
        for (auto it = begin; it != end; ++it) {
            out += text.substr(pos, it->position() - pos);
            out += fn(*it);
            pos = it->position() + it->length();
        }
        out += text.substr(pos);
        return out;
    }

    inline std::string ace_expand_currency(const std::string& match, const char* unit, const char* units, const char* cent, const char* cents) {
        std::string digits;
        for (char c : match) {
            // the reference maps "," to "." first, then strips non-digits
            if ((c >= '0' && c <= '9') || c == '.' || c == ',') {
                digits += c == ',' ? '.' : c;
            }
        }
        double value = atof(digits.c_str());
        long long whole = (long long)value;
        long long frac  = llround((value - (double)whole) * 100.0);
        std::string out = ace_number_to_words(whole) + " " + (whole == 1 ? unit : units);
        if (frac > 0) {
            out += ", " + ace_number_to_words(frac) + " " + (frac == 1 ? cent : cents);
        }
        return out;
    }

    inline std::string ace_expand_numbers_en(std::string text) {
        // 1,000.5 -> 1000.5
        static const std::regex comma_number(R"(\b\d{1,3}(,\d{3})*(\.\d+)?\b)");
        text = ace_replace_matches(text, comma_number, [](const std::smatch& m) {
            std::string s = m.str();
            s.erase(std::remove(s.begin(), s.end(), ','), s.end());
            return s;
        });
        // currency ($/£/€ before or after the amount)
        static const std::regex gbp(R"(((\xC2\xA3[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\xC2\xA3)))");
        static const std::regex usd(R"(((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$)))");
        static const std::regex eur(R"((([0-9\.\,]*[0-9]+\xE2\x82\xAC)|((\xE2\x82\xAC[0-9\.\,]*[0-9]+))))");
        text = ace_replace_matches(text, gbp, [](const std::smatch& m) {
            return ace_expand_currency(m.str(), "pound", "pounds", "penny", "pence");
        });
        text = ace_replace_matches(text, usd, [](const std::smatch& m) {
            return ace_expand_currency(m.str(), "dollar", "dollars", "cent", "cents");
        });
        text = ace_replace_matches(text, eur, [](const std::smatch& m) {
            return ace_expand_currency(m.str(), "euro", "euro", "cent", "cents");
        });
        // 12.5 -> twelve point five
        static const std::regex decimal_number(R"(([0-9]+[.,][0-9]+))");
        text = ace_replace_matches(text, decimal_number, [](const std::smatch& m) {
            std::string s = m.str(1);
            size_t dot    = s.find_first_of(".,");
            std::string frac = s.substr(dot + 1);
            // float() round-trips drop trailing zeros; all-zero fractions keep one
            while (frac.size() > 1 && frac.back() == '0') {
                frac.pop_back();
            }
            std::string out = ace_number_to_words(atoll(s.substr(0, dot).c_str())) + " point";
            size_t lead = 0;
            while (lead + 1 < frac.size() && frac[lead] == '0') {
                out += " zero";
                lead++;
            }
            out += " " + ace_number_to_words(atoll(frac.substr(lead).c_str()));
            return out;
        });
        // 1st -> first
        static const std::regex ordinal(R"(([0-9]+)(st|nd|rd|th))");
        text = ace_replace_matches(text, ordinal, [](const std::smatch& m) {
            return ace_number_to_ordinal_words(atoll(m.str(1).c_str()));
        });
        // 42 -> forty-two
        static const std::regex number(R"([0-9]+)");
        text = ace_replace_matches(text, number, [](const std::smatch& m) {
            return ace_number_to_words(atoll(m.str().c_str()));
        });
        return text;
    }

    inline std::string ace_expand_abbreviations_en(std::string text) {
        static const std::pair<const char*, const char*> kAbbrev[] = {
            {"mrs", "misess"}, {"mr", "mister"}, {"dr", "doctor"}, {"st", "saint"},
            {"co", "company"}, {"jr", "junior"}, {"maj", "major"}, {"gen", "general"},
            {"drs", "doctors"}, {"rev", "reverend"}, {"lt", "lieutenant"}, {"hon", "honorable"},
            {"sgt", "sergeant"}, {"capt", "captain"}, {"esq", "esquire"}, {"ltd", "limited"},
            {"col", "colonel"}, {"ft", "fort"}};
        for (const auto& ab : kAbbrev) {
            std::regex re(std::string("\\b") + ab.first + "\\.", std::regex::icase);
            text = std::regex_replace(text, re, ab.second);
        }
        return text;
    }

    inline std::string ace_expand_symbols_en(std::string text) {
        static const std::pair<const char*, const char*> kSymbols[] = {
            {"&", " and "}, {"@", " at "}, {"%", " percent "}, {"#", " hash "},
            {"$", " dollar "}, {"\xC2\xA3", " pound "}, {"\xC2\xB0", " degree "}};
        for (const auto& sym : kSymbols) {
            size_t pos = 0;
            const std::string from = sym.first;
            while ((pos = text.find(from, pos)) != std::string::npos) {
                text.replace(pos, from.size(), sym.second);
                pos += strlen(sym.second);
            }
            // the reference collapses double spaces after every symbol pass
            size_t dbl;
            while ((dbl = text.find("  ")) != std::string::npos) {
                text.replace(dbl, 2, " ");
            }
        }
        // .strip()
        size_t first = text.find_first_not_of(" \t\r\n");
        size_t last  = text.find_last_not_of(" \t\r\n");
        return first == std::string::npos ? "" : text.substr(first, last - first + 1);
    }

    inline std::string ace_clean_lyrics_line_en(std::string text) {
        text.erase(std::remove(text.begin(), text.end(), '"'), text.end());
        for (char& c : text) {
            if (c >= 'A' && c <= 'Z') {
                c = (char)(c - 'A' + 'a');
            }
        }
        text = ace_expand_numbers_en(text);
        text = ace_expand_abbreviations_en(text);
        text = ace_expand_symbols_en(text);
        // collapse_whitespace: \s+ -> " "
        static const std::regex ws(R"(\s+)");
        return std::regex_replace(text, ws, " ");
    }

    /* ------------------------------- tokenizer ------------------------------ */

    struct AceLyricTokenizer {
        static constexpr int kStartId = 261;  // [START]
        static constexpr int kLineId  = 2;    // [SPACE], the per-line separator
        static constexpr int kUnkId   = 1;    // [UNK]

        std::vector<std::string> id_to_token;
        std::unordered_map<std::string, int> token_to_id;
        std::unordered_map<std::string, int> merge_rank;  // "left\x1eright" -> rank
        std::vector<std::pair<std::string, int>> added_tokens;  // longest first
        bool valid = false;

        AceLyricTokenizer() {
            auto for_lines = [](const unsigned char* data, size_t len, const std::function<void(std::string&&)>& fn) {
                const char* p = (const char*)data;
                size_t start  = 0;
                for (size_t i = 0; i <= len; i++) {
                    if (i == len || p[i] == '\n') {
                        fn(std::string(p + start, i - start));
                        start = i + 1;
                    }
                }
            };
            for_lines(kAceLyricVocabData, kAceLyricVocabData_len, [&](std::string&& tok) {
                token_to_id[tok] = (int)id_to_token.size();
                id_to_token.push_back(std::move(tok));
            });
            int rank = 0;
            for_lines(kAceLyricMergesData, kAceLyricMergesData_len, [&](std::string&& line) {
                size_t sp = line.find(' ');
                if (sp != std::string::npos) {
                    merge_rank[line.substr(0, sp) + '\x1e' + line.substr(sp + 1)] = rank++;
                }
            });
            for (const auto& tok : kAceLyricAddedTokens) {
                added_tokens.emplace_back(tok.content, tok.id);
            }
            std::sort(added_tokens.begin(), added_tokens.end(),
                      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
            valid = !id_to_token.empty() && !merge_rank.empty();
        }

        static std::vector<std::string> utf8_chars(const std::string& text) {
            std::vector<std::string> out;
            for (size_t i = 0; i < text.size();) {
                size_t len = 1;
                unsigned char c = (unsigned char)text[i];
                if ((c & 0xE0) == 0xC0) {
                    len = 2;
                } else if ((c & 0xF0) == 0xE0) {
                    len = 3;
                } else if ((c & 0xF8) == 0xF0) {
                    len = 4;
                }
                len = std::min(len, text.size() - i);
                out.push_back(text.substr(i, len));
                i += len;
            }
            return out;
        }

        // HF Whitespace pre-tokenizer classes: \w+ | [^\w\s]+
        // 0 = whitespace, 1 = word, 2 = punctuation
        static int char_class(const std::string& ch) {
            if (ch.size() == 1) {
                unsigned char c = (unsigned char)ch[0];
                if (isspace(c)) {
                    return 0;
                }
                return (isalnum(c) || c == '_') ? 1 : 2;
            }
            // common non-ASCII punctuation and spaces; everything else counts as \w
            static const char* kPunct[] = {"‘", "’", "“", "”", "„",
                                           "–", "—", "…", "«", "»",
                                           "¡", "¿", "·"};
            if (ch == " ") {
                return 0;
            }
            for (const char* p : kPunct) {
                if (ch == p) {
                    return 2;
                }
            }
            return 1;
        }

        void bpe_word(const std::vector<std::string>& chars, std::vector<int>* out) const {
            std::vector<std::string> symbols = chars;
            while (symbols.size() >= 2) {
                int best_rank = INT_MAX;
                size_t best   = 0;
                for (size_t i = 0; i + 1 < symbols.size(); i++) {
                    auto it = merge_rank.find(symbols[i] + '\x1e' + symbols[i + 1]);
                    if (it != merge_rank.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best      = i;
                    }
                }
                if (best_rank == INT_MAX) {
                    break;
                }
                symbols[best] += symbols[best + 1];
                symbols.erase(symbols.begin() + best + 1);
            }
            for (const auto& sym : symbols) {
                auto it = token_to_id.find(sym);
                out->push_back(it != token_to_id.end() ? it->second : kUnkId);
            }
        }

        void encode_plain(const std::string& text, std::vector<int>* out) const {
            auto chars = utf8_chars(text);
            std::vector<std::string> word;
            int word_class = -1;
            auto flush = [&]() {
                if (!word.empty()) {
                    bpe_word(word, out);
                    word.clear();
                }
            };
            for (auto& ch : chars) {
                int cls = char_class(ch);
                if (cls == 0) {
                    flush();
                    word_class = -1;
                    continue;
                }
                if (cls != word_class) {
                    flush();
                    word_class = cls;
                }
                word.push_back(std::move(ch));
            }
            flush();
        }

        // one preprocessed line: "[en]" prefix, spaces as [SPACE], added tokens atomic
        void encode_line(const std::string& cleaned, std::vector<int>* out) const {
            std::string text = "[en]" + cleaned;
            std::string plain;
            for (size_t i = 0; i < text.size();) {
                if (text[i] == ' ') {
                    if (!plain.empty()) {
                        encode_plain(plain, out);
                        plain.clear();
                    }
                    out->push_back(kLineId);  // [SPACE]
                    i++;
                    continue;
                }
                if (text[i] == '[') {
                    const std::pair<std::string, int>* hit = nullptr;
                    for (const auto& tok : added_tokens) {
                        if (text.compare(i, tok.first.size(), tok.first) == 0) {
                            hit = &tok;
                            break;
                        }
                    }
                    if (hit != nullptr) {
                        if (!plain.empty()) {
                            encode_plain(plain, out);
                            plain.clear();
                        }
                        out->push_back(hit->second);
                        i += hit->first.size();
                        continue;
                    }
                }
                plain += text[i];
                i++;
            }
            if (!plain.empty()) {
                encode_plain(plain, out);
            }
        }

        // the reference pipeline's tokenize_lyrics
        std::vector<int32_t> tokenize_lyrics(const std::string& lyrics) const {
            std::vector<int> ids = {kStartId};
            std::string line;
            auto handle_line = [&]() {
                size_t first = line.find_first_not_of(" \t\r");
                if (first == std::string::npos) {
                    ids.push_back(kLineId);
                    line.clear();
                    return;
                }
                size_t last = line.find_last_not_of(" \t\r");
                encode_line(ace_clean_lyrics_line_en(line.substr(first, last - first + 1)), &ids);
                ids.push_back(kLineId);
                line.clear();
            };
            for (char c : lyrics) {
                if (c == '\n') {
                    handle_line();
                } else {
                    line += c;
                }
            }
            handle_line();
            return std::vector<int32_t>(ids.begin(), ids.end());
        }
    };

}  // namespace AceStep

#endif  // __SD_MODEL_DIFFUSION_ACE_STEP_LYRIC_HPP__
