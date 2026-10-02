#include "serving-preset.h"

#include "common.h"
#include "gguf.h"
#include "ggml.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

struct preset_meta {
    std::string arch;
    int64_t     n_layer   = -1;
    int64_t     n_embd    = -1;
    int64_t     n_nextn   = 0;
    int64_t     file_type = -1;
    uint64_t    file_size = 0;
};

// settings a preset can carry; a negative / COUNT value means "leave alone"
struct preset_vals {
    int       ctx          = -1;
    int       flash_attn   = -1;
    int       n_parallel   = -1;
    ggml_type kv_k         = GGML_TYPE_COUNT;
    ggml_type kv_v         = GGML_TYPE_COUNT;
    bool      mtp          = false; // spec type draft-mtp
    int       draft_n_max  = -1;
    ggml_type draft_kv_k   = GGML_TYPE_COUNT;
    ggml_type draft_kv_v   = GGML_TYPE_COUNT;
    int       draft_vocab  = -1;
    bool      draft_vocab_adaptive = false;
};

// file_type values (llama_ftype)
constexpr int FTYPE_Q8_0   = 7;
constexpr int FTYPE_Q4_K_M = 15;
constexpr int FTYPE_Q6_K   = 18;

struct preset_identity {
    const char * name;
    const char * arch;
    int64_t      n_layer;
    int64_t      n_embd;
    int          file_type;
    preset_vals  vals;
};

preset_vals qwen38_27b_vals() {
    preset_vals v;
    v.ctx         = 262144;
    v.flash_attn  = 1;
    v.n_parallel  = 1;
    v.kv_k        = GGML_TYPE_Q8_0;
    v.kv_v        = GGML_TYPE_Q8_0;
    v.mtp         = true;
    v.draft_n_max = 4;
    v.draft_kv_k  = GGML_TYPE_Q8_0;
    v.draft_kv_v  = GGML_TYPE_Q8_0;
    v.draft_vocab = 98304;
    v.draft_vocab_adaptive = true;
    // note: -lm none / GGML_VK_HOST_GET_ROWS are deliberately not part of the preset (must be set together by the operator)
    return v;
}

preset_vals minicpm5_2b_vals() {
    preset_vals v;
    v.flash_attn = 1;
    v.n_parallel = 1;
    v.kv_k       = GGML_TYPE_F16;
    v.kv_v       = GGML_TYPE_F16;
    // TODO(W4): draft setting (spec type / n-max), to be filled in by measurement
    return v;
}

const preset_identity IDENTITY_PRESETS[] = {
    { "qwen3.8-27b-q6_k",   "qwen35", 65, 5120, FTYPE_Q6_K,   qwen38_27b_vals()  },
    { "qwen3.8-27b-q4_k_m", "qwen35", 65, 5120, FTYPE_Q4_K_M, qwen38_27b_vals()  },
    { "minicpm5-2b-q4_k_m", "llama",  42, 2048, FTYPE_Q4_K_M, minicpm5_2b_vals() },
    { "minicpm5-2b-q8_0",   "llama",  42, 2048, FTYPE_Q8_0,   minicpm5_2b_vals() },
};

preset_vals arch_fallback_vals() {
    preset_vals v;
    v.flash_attn  = 1;
    v.mtp         = true;
    v.draft_n_max = 3;
    v.n_parallel  = 1; // MTP with parallel > 1 roughly halves decode
    return v;
}

int64_t get_int(const gguf_context * ctx, const std::string & key, int64_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:               return def;
    }
}

bool read_meta(const std::string & path, preset_meta & m) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), gp);
    if (!ctx) {
        return false;
    }
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(ctx, arch_id) == GGUF_TYPE_STRING) {
        m.arch = gguf_get_val_str(ctx, arch_id);
    }
    if (!m.arch.empty()) {
        m.n_layer = get_int(ctx, m.arch + ".block_count",         -1);
        m.n_embd  = get_int(ctx, m.arch + ".embedding_length",    -1);
        m.n_nextn = get_int(ctx, m.arch + ".nextn_predict_layers", 0);
    }
    m.file_type = get_int(ctx, "general.file_type", -1);
    gguf_free(ctx);

    std::error_code ec;
    m.file_size = std::filesystem::file_size(path, ec);
    return !m.arch.empty();
}

bool is_explicit(const common_params & p, const char * opt) {
    return p.explicit_args.count(opt) > 0;
}

void apply_vals(common_params & p, const preset_vals & v, common_serving_preset_result & res) {
    auto field = [&](const char * opt, const std::string & val, auto && apply) {
        if (is_explicit(p, opt)) {
            res.skipped.emplace_back(opt, val);
        } else {
            apply();
            res.applied.emplace_back(opt, val);
        }
    };
    auto bool_str = [](bool b) { return std::string(b ? "on" : "off"); };

    if (v.ctx >= 0) {
        field("--ctx-size", std::to_string(v.ctx), [&] { p.n_ctx = v.ctx; });
    }
    if (v.flash_attn >= 0) {
        field("--flash-attn", bool_str(v.flash_attn), [&] {
            p.flash_attn_type = v.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        });
    }
    if (v.n_parallel >= 0) {
        field("--parallel", std::to_string(v.n_parallel), [&] { p.n_parallel = v.n_parallel; });
    }
    if (v.kv_k != GGML_TYPE_COUNT) {
        field("--cache-type-k", ggml_type_name(v.kv_k), [&] { p.cache_type_k = v.kv_k; });
    }
    if (v.kv_v != GGML_TYPE_COUNT) {
        field("--cache-type-v", ggml_type_name(v.kv_v), [&] { p.cache_type_v = v.kv_v; });
    }
    if (v.mtp) {
        field("--spec-type", "draft-mtp", [&] { p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP }; });
    }
    if (v.draft_n_max >= 0) {
        field("--spec-draft-n-max", std::to_string(v.draft_n_max), [&] { p.speculative.draft.n_max = v.draft_n_max; });
    }
    if (v.draft_kv_k != GGML_TYPE_COUNT) {
        field("--spec-draft-type-k", ggml_type_name(v.draft_kv_k), [&] { p.speculative.draft.cache_type_k = v.draft_kv_k; });
    }
    if (v.draft_kv_v != GGML_TYPE_COUNT) {
        field("--spec-draft-type-v", ggml_type_name(v.draft_kv_v), [&] { p.speculative.draft.cache_type_v = v.draft_kv_v; });
    }
    if (v.draft_vocab >= 0) {
        field("--spec-draft-vocab", std::to_string(v.draft_vocab), [&] { p.speculative.draft.vocab_n = v.draft_vocab; });
    }
    if (v.draft_vocab_adaptive) {
        field("--spec-draft-vocab-adaptive", "on", [&] { p.speculative.draft.vocab_adaptive = true; });
    }
}

} // namespace

common_serving_preset_result common_serving_preset_select(common_params & params) {
    common_serving_preset_result res;

    preset_meta m;
    if (params.model.path.empty() || !read_meta(params.model.path, m)) {
        return res;
    }

    for (const auto & id : IDENTITY_PRESETS) {
        if (m.arch == id.arch && m.n_layer == id.n_layer && m.n_embd == id.n_embd && m.file_type == id.file_type) {
            res.name = id.name;
            res.kind = "identity";
            apply_vals(params, id.vals, res);
            return res;
        }
    }

    if (m.n_nextn > 0) {
        res.name = m.arch;
        res.kind = "arch-fallback";
        apply_vals(params, arch_fallback_vals(), res);
    }
    return res;
}

std::string common_serving_preset_summary(const common_serving_preset_result & res) {
    std::string s = "preset: " + res.name + " (" + res.kind + ") applied:";
    for (const auto & kv : res.applied) {
        s += " " + kv.first + "=" + kv.second;
    }
    return s;
}

std::string common_serving_preset_describe(const common_serving_preset_result & res) {
    std::string s = "preset: " + res.name + " (" + res.kind + ")\n";
    for (const auto & kv : res.applied) {
        s += "  applied: " + kv.first + "=" + kv.second + "\n";
    }
    for (const auto & kv : res.skipped) {
        s += "  skipped (explicit): " + kv.first + " (preset: " + kv.second + ")\n";
    }
    return s;
}
