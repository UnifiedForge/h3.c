/* accel-profiling: time the shipped DiT attention (h3_gpu_sdpa_bf16, MPSGraph
 * SDPA, query-blocked) and a QKV-sized bf16 linear at the DiT head shape
 * (56 heads x 128, hidden 5376) over a range of sequence lengths.
 *
 * usage: h3_sdpa_bench [N ...]       default N list below
 * env:   H3_SDPA_QUERY_BLOCK (read by the library per call)
 *        BENCH_BLOCKS="256,512,1024" sweeps block sizes per N
 *        BENCH_LINEAR=0 skips the linear reference
 *        BENCH_REPS (default 2) */
#include "h3_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dispatch/dispatch.h>

enum { HEADS = 56, HEAD_DIM = 128, HIDDEN = 5376, INNER = HEADS * HEAD_DIM };

static uint16_t bf16_of(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static float float_of(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t *last_host; /* kept when keep_host is set */
static int keep_host;

static h3_gpu_tensor *random_bf16(h3_gpu *gpu, size_t elements, uint32_t seed,
                                  float amplitude) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    uint16_t *host = malloc(elements * sizeof(*host));
    if (!tensor || !host) { free(host); return tensor; }
    uint32_t state = seed * 2654435761u + 1u;
    for (size_t i = 0; i < elements; i++) {
        state = state * 1664525u + 1013904223u;
        float u = (float)(state >> 8) / 16777216.0f * 2.0f - 1.0f;
        host[i] = bf16_of(u * amplitude);
    }
    h3_gpu_tensor_write_bf16(tensor, host, elements);
    if (keep_host) last_host = host; else free(host);
    return tensor;
}

#include <time.h>
static double h3_bench_wall(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static double gpu_now(h3_gpu *gpu) {
    h3_gpu_stats stats;
    h3_gpu_get_stats(gpu, &stats);
    return stats.gpu_seconds;
}

static double time_sdpa(h3_gpu *gpu, h3_gpu_tensor *out, h3_gpu_tensor *q,
                        h3_gpu_tensor *k, h3_gpu_tensor *v, uint32_t n,
                        int reps) {
    double best = 1e30;
    for (int r = 0; r <= reps; r++) {
        double before = h3_bench_wall();
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16(gpu, out, q, k, v, n, HEADS, HEAD_DIM,
                              0.08838834764831845f) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "sdpa N=%u failed: %s\n", n, h3_gpu_error(gpu));
            return -1.0;
        }
        double t = h3_bench_wall() - before;
        if (r > 0 && t < best) best = t; /* r == 0 is graph compile warm-up */
    }
    return best;
}

static double time_linear(h3_gpu *gpu, h3_gpu_tensor *out, h3_gpu_tensor *x,
                          h3_gpu_tensor *w, uint32_t rows, uint32_t in,
                          uint32_t outd, int reps) {
    double best = 1e30;
    for (int r = 0; r <= reps; r++) {
        double before = h3_bench_wall();
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_linear_bf16(gpu, out, x, w, NULL, rows, in, outd) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "linear rows=%u failed: %s\n", rows,
                    h3_gpu_error(gpu));
            return -1.0;
        }
        double t = h3_bench_wall() - before;
        if (r > 0 && t < best) best = t;
    }
    return best;
}

/* float64 reference for a few query rows (all heads), compared with the
 * row-major blocked output and the head-major whole-sequence output. */
static void accuracy(h3_gpu *gpu, uint32_t n, const uint16_t *hq,
                     const uint16_t *hk, const uint16_t *hv,
                     h3_gpu_tensor *blocked, h3_gpu_tensor *whole) {
    enum { SAMPLES = 8 };
    size_t elements = (size_t)n * INNER;
    uint16_t *ob = malloc(elements * 2), *oh = malloc(elements * 2);
    h3_gpu_tensor_read_bf16(blocked, ob, elements);
    h3_gpu_tensor_read_bf16(whole, oh, elements);
    double *ref = calloc((size_t)SAMPLES * INNER, sizeof(double));
    uint32_t *rows = malloc(SAMPLES * sizeof(*rows));
    for (int s = 0; s < SAMPLES; s++)
        rows[s] = (uint32_t)(((uint64_t)n - 1) * (uint64_t)s / (SAMPLES - 1));
    dispatch_apply(SAMPLES * HEADS, dispatch_get_global_queue(0, 0),
                   ^(size_t job) {
        int s = (int)(job / HEADS), h = (int)(job % HEADS);
        uint32_t i = rows[s];
        double *score = malloc((size_t)n * sizeof(double));
        double mx = -1e300;
        for (uint32_t j = 0; j < n; j++) {
            double acc = 0.0;
            const uint16_t *qr = hq + ((size_t)i * HEADS + h) * HEAD_DIM;
            const uint16_t *kr = hk + ((size_t)j * HEADS + h) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                acc += (double)float_of(qr[d]) * float_of(kr[d]);
            score[j] = acc / sqrt((double)HEAD_DIM);
            if (score[j] > mx) mx = score[j];
        }
        double sum = 0.0;
        for (uint32_t j = 0; j < n; j++) sum += (score[j] = exp(score[j] - mx));
        double *out = ref + ((size_t)s * HEADS + h) * HEAD_DIM;
        for (uint32_t j = 0; j < n; j++) {
            const uint16_t *vr = hv + ((size_t)j * HEADS + h) * HEAD_DIM;
            double p = score[j] / sum;
            for (int d = 0; d < HEAD_DIM; d++) out[d] += p * float_of(vr[d]);
        }
        free(score);
    });
    double nb = 0, nh = 0, nr = 0, nbh = 0;
    for (int s = 0; s < SAMPLES; s++)
        for (int h = 0; h < HEADS; h++)
            for (int d = 0; d < HEAD_DIM; d++) {
                double r = ref[((size_t)s * HEADS + h) * HEAD_DIM + d];
                double b = float_of(ob[((size_t)rows[s] * HEADS + h) * HEAD_DIM + d]);
                double w = float_of(oh[((size_t)h * n + rows[s]) * HEAD_DIM + d]);
                nb += (b - r) * (b - r); nh += (w - r) * (w - r);
                nbh += (b - w) * (b - w); nr += r * r;
            }
    printf("accuracy,%u,rel_l2_blocked=%.3e,rel_l2_whole_headmajor=%.3e,"
           "blocked_vs_whole=%.3e\n", n, sqrt(nb / nr), sqrt(nh / nr),
           sqrt(nbh / nr));
    fflush(stdout);
    free(ob); free(oh); free(ref); free(rows);
}

static double time_sdpa_whole(h3_gpu *gpu, h3_gpu_tensor *out,
                              h3_gpu_tensor *q, h3_gpu_tensor *k,
                              h3_gpu_tensor *v, uint32_t n, int reps) {
    double best = 1e30;
    for (int r = 0; r <= reps; r++) {
        double before = h3_bench_wall();
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16_head_major_output(gpu, out, q, k, v, n, HEADS,
                HEAD_DIM, 0.08838834764831845f) ||
            !h3_gpu_submit(gpu)) {
            fprintf(stderr, "whole sdpa N=%u failed: %s\n", n,
                    h3_gpu_error(gpu));
            return -1.0;
        }
        double t = h3_bench_wall() - before;
        if (r > 0 && t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv) {
    static const uint32_t defaults[] = {6144, 12288, 18432, 24576, 36864,
                                        49152, 61440, 86016};
    uint32_t list[64];
    size_t count = 0;
    for (int i = 1; i < argc && count < 64; i++)
        list[count++] = (uint32_t)strtoul(argv[i], NULL, 10);
    if (!count)
        for (; count < sizeof(defaults) / sizeof(*defaults); count++)
            list[count] = defaults[count];
    int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 2;
    int linear = !getenv("BENCH_LINEAR") || strcmp(getenv("BENCH_LINEAR"), "0");
    const char *blocks = getenv("BENCH_BLOCKS");

    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    printf("device_m5=%d nax_mlp=%d int8_mlp=%d\n", h3_gpu_is_m5(gpu),
           h3_gpu_has_nax_mlp(gpu), h3_gpu_has_int8_mlp(gpu));
    printf("kind,N,block,seconds,tflops\n");
    for (size_t c = 0; c < count; c++) {
        uint32_t n = list[c];
        size_t elements = (size_t)n * INNER;
        int acc = getenv("BENCH_ACCURACY") && strcmp(getenv("BENCH_ACCURACY"), "0");
        keep_host = acc;
        h3_gpu_tensor *q = random_bf16(gpu, elements, 1, 2.0f);
        uint16_t *hq = last_host;
        h3_gpu_tensor *k = random_bf16(gpu, elements, 2, 2.0f);
        uint16_t *hk = last_host;
        h3_gpu_tensor *v = random_bf16(gpu, elements, 3, 1.0f);
        uint16_t *hv = last_host;
        keep_host = 0;
        h3_gpu_tensor *o = h3_gpu_tensor_new_bf16(gpu, elements);
        if (!q || !k || !v || !o) { fprintf(stderr, "alloc N=%u\n", n); return 1; }
        double flops = 4.0 * (double)n * n * INNER;
        char list_copy[256];
        snprintf(list_copy, sizeof(list_copy), "%s", blocks ? blocks : "");
        char *cursor = blocks ? list_copy : NULL;
        do {
            char *token = cursor ? strsep(&cursor, ",") : NULL;
            if (token) setenv("H3_SDPA_QUERY_BLOCK", token, 1);
            const char *b = getenv("H3_SDPA_QUERY_BLOCK");
            double t = time_sdpa(gpu, o, q, k, v, n, reps);
            printf("sdpa,%u,%s,%.4f,%.2f\n", n, b ? b : "256", t,
                   t > 0 ? flops / t * 1e-12 : 0.0);
            fflush(stdout);
        } while (cursor);
        unsetenv("H3_SDPA_QUERY_BLOCK");
        if (getenv("BENCH_WHOLE") && strcmp(getenv("BENCH_WHOLE"), "0")) {
            h3_gpu_tensor *ow = h3_gpu_tensor_new_bf16(gpu, elements);
            double t = time_sdpa_whole(gpu, ow, q, k, v, n, reps);
            printf("sdpa_whole_headmajor,%u,-,%.4f,%.2f\n", n, t,
                   t > 0 ? flops / t * 1e-12 : 0.0);
            fflush(stdout);
            if (acc && t > 0) {
                time_sdpa(gpu, o, q, k, v, n, 0);
                accuracy(gpu, n, hq, hk, hv, o, ow);
            }
            h3_gpu_tensor_free(ow);
        }
        free(hq); free(hk); free(hv);
        h3_gpu_tensor_free(q); h3_gpu_tensor_free(k);
        h3_gpu_tensor_free(v); h3_gpu_tensor_free(o);
        if (linear) {
            h3_gpu_tensor *x = random_bf16(gpu, (size_t)n * HIDDEN, 4, 1.0f);
            h3_gpu_tensor *w = random_bf16(gpu, (size_t)HIDDEN * INNER * 3, 5,
                                           0.02f);
            h3_gpu_tensor *y = h3_gpu_tensor_new_bf16(gpu, (size_t)n * INNER * 3);
            double t = time_linear(gpu, y, x, w, n, HIDDEN, INNER * 3, reps);
            printf("linear_qkv,%u,-,%.4f,%.2f\n", n, t,
                   t > 0 ? 2.0 * n * HIDDEN * INNER * 3 / t * 1e-12 : 0.0);
            fflush(stdout);
            h3_gpu_tensor_free(x); h3_gpu_tensor_free(w); h3_gpu_tensor_free(y);
        }
    }
    h3_gpu_free(gpu);
    return 0;
}
