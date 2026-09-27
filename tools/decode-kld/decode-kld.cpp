// llama-decode-kld: teacher-forced decode-path KL divergence at depth.
//
// Prefills the first --depth tokens of --file in ubatches (no logits), then feeds the
// next --score tokens in batches of --batch tokens, requesting logits for every token.
// Either saves those logits (fp16, full vocab) or compares them to a saved base file.
//
//   llama-decode-kld -m M -ctk q8_0 -ctv q8_0 -c 262144 -fa on -ngl 99 -dev Vulkan1 \
//       --file corpus.txt --depth 70000 --score 512 --batch 4 [--save-logits F | --base-logits F]
//
// This exercises the small-N decode kernels (N<=8 FA, n=4 GEMV, split-KV combine) at
// real depth, which llama-perplexity's ubatch-only KLD does not.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char     MAGIC[4] = { 'D', 'K', 'L', 'D' };
constexpr uint32_t VERSION  = 1;

struct file_header {
    char     magic[4];
    uint32_t version;
    uint32_t n_vocab;
    uint32_t n_score;
    uint32_t depth;
    uint32_t batch;
    uint32_t n_ctx;
    uint32_t reserved;
};
// followed by: int32 next_token[n_score]; fp16 logits[n_score][n_vocab]

struct extra_args {
    int         depth = 70000;
    int         score = 512;
    int         batch = 4;
    std::string save_path;
    std::string base_path;
};

// Pull the tool-specific flags out of argv so common_params_parse sees only its own.
bool split_args(int argc, char ** argv, extra_args & ex, std::vector<char *> & rest) {
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        const char * v = nullptr;
        if (a == "--depth")            { if (!(v = next("--depth"))) return false; ex.depth = std::stoi(v); }
        else if (a == "--score")       { if (!(v = next("--score"))) return false; ex.score = std::stoi(v); }
        else if (a == "--batch")       { if (!(v = next("--batch"))) return false; ex.batch = std::stoi(v); }
        else if (a == "--save-logits") { if (!(v = next("--save-logits"))) return false; ex.save_path = v; }
        else if (a == "--base-logits") { if (!(v = next("--base-logits"))) return false; ex.base_path = v; }
        else rest.push_back(argv[i]);
    }
    return true;
}

// log-softmax in double
void log_softmax(const float * x, int n, std::vector<double> & out) {
    out.resize(n);
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, (double) x[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp((double) x[i] - mx);
    const double lse = mx + std::log(s);
    for (int i = 0; i < n; ++i) out[i] = (double) x[i] - lse;
}

int argmax(const float * x, int n) {
    return (int) (std::max_element(x, x + n) - x);
}

} // namespace

int main(int argc, char ** argv) {
    extra_args ex;
    std::vector<char *> rest;
    if (!split_args(argc, argv, ex, rest)) return 1;
    if (ex.save_path.empty() == ex.base_path.empty()) {
        fprintf(stderr, "exactly one of --save-logits F or --base-logits F is required\n");
        return 1;
    }
    if (ex.batch < 1 || ex.score < 1 || ex.depth < 0) {
        fprintf(stderr, "invalid --depth/--score/--batch\n");
        return 1;
    }

    common_params params;
    params.n_batch  = 512;
    params.n_ubatch = 512;
    params.warmup   = false;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    if (params.prompt.empty()) {
        fprintf(stderr, "--file is required\n");
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load model/context\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_ub    = (int) llama_n_ubatch(ctx);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    fprintf(stderr, "%s: tokenized %zu tokens, n_vocab %d, n_ctx %d, n_ubatch %d\n", __func__, tokens.size(), n_vocab, n_ctx, n_ub);
    const int need = ex.depth + ex.score + 1;
    if ((int) tokens.size() < need) {
        fprintf(stderr, "file has %zu tokens, need depth+score+1 = %d\n", tokens.size(), need);
        return 1;
    }
    if (ex.depth + ex.score > n_ctx) {
        fprintf(stderr, "depth+score %d exceeds n_ctx %d\n", ex.depth + ex.score, n_ctx);
        return 1;
    }

    llama_memory_clear(llama_get_memory(ctx), true);

    // 1) depth fill, no logits
    const int64_t t0 = ggml_time_us();
    {
        llama_batch batch = llama_batch_init(n_ub, 0, 1);
        for (int p = 0; p < ex.depth; p += n_ub) {
            common_batch_clear(batch);
            const int n = std::min(n_ub, ex.depth - p);
            for (int j = 0; j < n; ++j) common_batch_add(batch, tokens[p + j], p + j, { 0 }, false);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at depth fill pos %d\n", p);
                return 1;
            }
            if ((p / n_ub) % 20 == 0) fprintf(stderr, "\rfill %d / %d", p + n, ex.depth);
        }
        llama_batch_free(batch);
    }
    const int64_t t1 = ggml_time_us();
    fprintf(stderr, "\nfill done: %d tokens in %.1f s\n", ex.depth, (t1 - t0) / 1e6);

    // 2) teacher-forced scoring in --batch-token batches, logits for every token
    std::vector<ggml_fp16_t> cur((size_t) ex.score * n_vocab);
    std::vector<int32_t>     next_tok(ex.score);
    {
        llama_batch batch = llama_batch_init(ex.batch, 0, 1);
        for (int s = 0; s < ex.score; s += ex.batch) {
            common_batch_clear(batch);
            const int n = std::min(ex.batch, ex.score - s);
            for (int j = 0; j < n; ++j) {
                const int p = ex.depth + s + j;
                common_batch_add(batch, tokens[p], p, { 0 }, true);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed at score pos %d\n", s);
                return 1;
            }
            for (int j = 0; j < n; ++j) {
                const float * lg = llama_get_logits_ith(ctx, j);
                ggml_fp32_to_fp16_row(lg, cur.data() + (size_t) (s + j) * n_vocab, n_vocab);
                next_tok[s + j] = tokens[ex.depth + s + j + 1];
            }
        }
        llama_batch_free(batch);
    }
    const int64_t t2 = ggml_time_us();
    fprintf(stderr, "score done: %d tokens in batches of %d in %.2f s (%.1f t/s)\n", ex.score, ex.batch, (t2 - t1) / 1e6, ex.score / ((t2 - t1) / 1e6));

    if (!ex.save_path.empty()) {
        FILE * f = fopen(ex.save_path.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", ex.save_path.c_str()); return 1; }
        file_header h{};
        memcpy(h.magic, MAGIC, 4);
        h.version = VERSION; h.n_vocab = n_vocab; h.n_score = ex.score; h.depth = ex.depth;
        h.batch = ex.batch; h.n_ctx = n_ctx;
        fwrite(&h, sizeof(h), 1, f);
        fwrite(next_tok.data(), sizeof(int32_t), next_tok.size(), f);
        fwrite(cur.data(), sizeof(ggml_fp16_t), cur.size(), f);
        fclose(f);
        printf("saved %d x %d fp16 logits to %s (depth %d, batch %d)\n", ex.score, n_vocab, ex.save_path.c_str(), ex.depth, ex.batch);
        return 0;
    }

    // compare against base
    FILE * f = fopen(ex.base_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ex.base_path.c_str()); return 1; }
    file_header h{};
    std::vector<int32_t>     base_tok(ex.score);
    std::vector<ggml_fp16_t> base((size_t) ex.score * n_vocab);
    bool ok = fread(&h, sizeof(h), 1, f) == 1 && memcmp(h.magic, MAGIC, 4) == 0 && h.version == VERSION;
    if (!ok || (int) h.n_vocab != n_vocab || (int) h.n_score != ex.score || (int) h.depth != ex.depth) {
        fprintf(stderr, "base file header mismatch (vocab %u score %u depth %u)\n", h.n_vocab, h.n_score, h.depth);
        fclose(f);
        return 1;
    }
    if ((int) h.batch != ex.batch) {
        fprintf(stderr, "warning: base was recorded with --batch %u, this run uses %d\n", h.batch, ex.batch);
    }
    ok = fread(base_tok.data(), sizeof(int32_t), base_tok.size(), f) == base_tok.size() &&
         fread(base.data(), sizeof(ggml_fp16_t), base.size(), f) == base.size();
    fclose(f);
    if (!ok || base_tok != next_tok) {
        fprintf(stderr, "base file truncated or token stream differs (different corpus/tokenizer?)\n");
        return 1;
    }

    std::vector<float>  pb(n_vocab), pc(n_vocab);
    std::vector<double> lb, lc, kld(ex.score);
    int flips = 0;
    double sum_dlp = 0.0;
    for (int i = 0; i < ex.score; ++i) {
        ggml_fp16_to_fp32_row(base.data() + (size_t) i * n_vocab, pb.data(), n_vocab);
        ggml_fp16_to_fp32_row(cur.data()  + (size_t) i * n_vocab, pc.data(), n_vocab);
        log_softmax(pb.data(), n_vocab, lb);
        log_softmax(pc.data(), n_vocab, lc);
        double k = 0.0;
        for (int t = 0; t < n_vocab; ++t) {
            const double p = std::exp(lb[t]);
            if (p > 0.0) k += p * (lb[t] - lc[t]);
        }
        kld[i] = std::max(0.0, k);
        flips  += argmax(pb.data(), n_vocab) != argmax(pc.data(), n_vocab);
        sum_dlp += lc[next_tok[i]] - lb[next_tok[i]];
    }
    std::vector<double> sorted = kld;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0.0;
    for (double k : kld) mean += k;
    mean /= ex.score;
    const double p99 = sorted[std::min<size_t>(sorted.size() - 1, (size_t) std::ceil(0.99 * sorted.size()) - 1)];

    printf("decode-kld depth=%d score=%d batch=%d base=%s\n", ex.depth, ex.score, ex.batch, ex.base_path.c_str());
    printf("mean_kld      %.6f\n", mean);
    printf("p99_kld       %.6f\n", p99);
    printf("max_kld       %.6f\n", sorted.back());
    printf("top1_flip     %.4f%% (%d / %d)\n", 100.0 * flips / ex.score, flips, ex.score);
    printf("mean_dlogp    %+.6f  (current - base, actual next token)\n", sum_dlp / ex.score);
    return 0;
}
