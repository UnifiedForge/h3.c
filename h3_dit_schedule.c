#include "h3_dit_schedule.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition,
                        float **features_out, float **times_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 1.0f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    float *features = malloc((size_t)count * TIME_INPUT * sizeof(*features));
    if (!times || !features) {
        free(times);
        free(features);
        fail(error, error_size, "out of memory allocating timestep features");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = 0.999f;
    if (audio_condition) times[audio_condition_row] = 1.0f;
    for (uint32_t row = 0; row < count; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    *features_out = features;
    *times_out = times;
    return 1;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

/* The delta branch for the two AdaLN targets. Same three dispatches as the
 * four per-step projections, on the same shared intermediate,
 * except that here the "rows" are timestep rows and the whole thing runs once
 * at load: both tensors below are gone before the denoiser starts. One delta
 * buffer covers both targets, because FINAL_OUTPUT < BLOCK_OUTPUT. */
typedef struct {
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *delta;
} adaln_lora_scratch;

static void free_adaln_scratch(adaln_lora_scratch *scratch) {
    free_tensor(&scratch->hidden);
    free_tensor(&scratch->delta);
}

static uint64_t adaln_lora_rank(const h3_lora_set *loras) {
    uint64_t rank = 0;
    if (!loras) return 0;
    for (size_t index = 0; index < loras->count; index++) {
        const h3_lora_adapter *adapter = loras->entries[index].adapter;
        for (size_t which = 0; which < adapter->pair_count; which++) {
            const h3_lora_pair *pair = &adapter->pairs[which];
            /* Read per pair, never per file: upstream is rank 16 here and 64
             * on the backbone. */
            if (pair->adaln && pair->rank > rank) rank = pair->rank;
        }
    }
    return rank;
}

/* Encodes every branch of `site` onto `y`, which already holds the base
 * projection. The base weight is only read by the caller's own op, so H1 and
 * H2 hold: nothing here names a checkpoint tensor. Runs inside the caller's
 * h3_gpu_begin/h3_gpu_submit pair, which is also why the site is the caller's
 * to build and to free: releasing a branch marks its buffers purgeable, so it
 * has to outlive the submit that reads them. */
static int adaln_lora_dispatch(h3_gpu *gpu, const h3_lora_site *site,
                               const adaln_lora_scratch *scratch,
                               h3_gpu_tensor *y, const h3_gpu_tensor *time,
                               uint32_t rows, uint32_t output_dim,
                               char *error, size_t error_size) {
    return h3_lora_dispatch_branch(gpu, site, y, time, scratch->hidden,
                                   scratch->delta, rows, H3_DIT_TIME_DIM,
                                   output_dim, "AdaLN LoRA", error,
                                   error_size);
}

/* quant: AdaLN table cache.
 *
 * The table is a pure function of the AdaLN weights and of the timestep rows,
 * so it can be stored once and read back instead of streaming the 24 GiB of
 * adaln_proj weights at every load. One file per row set:
 *   <component>/adaln_cache/adaln-<key>.safetensors
 * where <component> is the parent of the transformer directory and <key>
 * hashes the exact float bits of the timestep rows (in row order) plus a
 * fingerprint of the weights. A hit is the very table the same rows produced
 * on the GPU, so the run is byte-identical. LoRA AdaLN pairs are still added
 * on top at load, exactly as before, from the time embedding alone.
 *
 * Files are read whenever present. They are written only if the adaln_cache
 * directory already exists (created by the quantize script), so new sigma sets
 * never fill the disk silently. H3_ADALN_CACHE=0 disables the cache,
 * H3_ADALN_CACHE_DIR overrides the directory. */

#define ADALN_CACHE_VERSION "1"

static uint64_t fnv64(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

/* Fingerprint of everything the table depends on besides the rows: shapes
 * and the first/last 64 KiB of the time embedder and AdaLN tensors. Cheap,
 * and independent of paths and mtimes, so a copy on another Mac still hits. */
static int adaln_fingerprint(const h3_weight_store *weights, uint64_t *out) {
    static const char *names[] = {
        "time_embedder.proj_in.weight", "time_embedder.proj_out.weight",
        "blocks.0.adaln_proj.linear.weight", "blocks.0.adaln_proj.linear.bias",
        "blocks.25.adaln_proj.linear.weight",
        "blocks.49.adaln_proj.linear.weight",
        "blocks.49.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias"
    };
    enum { SAMPLE = 64 * 1024 };
    unsigned char *buffer = malloc(SAMPLE);
    if (!buffer) return 0;
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    hash = fnv64(hash, ADALN_CACHE_VERSION, 1);
    int ok = 1;
    for (size_t i = 0; ok && i < sizeof(names) / sizeof(*names); i++) {
        const h3_st_header *header = NULL;
        const h3_st_tensor *tensor = h3_weight_find(weights, names[i], &header);
        if (!tensor || !header) { ok = 0; break; }
        hash = fnv64(hash, names[i], strlen(names[i]));
        hash = fnv64(hash, &tensor->dtype, sizeof(tensor->dtype));
        hash = fnv64(hash, tensor->shape, sizeof(tensor->shape[0]) *
                                          (size_t)tensor->ndim);
        uint64_t size = tensor->data_end - tensor->data_begin;
        int fd = open(header->path, O_RDONLY);
        if (fd < 0) { ok = 0; break; }
        uint64_t take = size < SAMPLE ? size : SAMPLE;
        uint64_t offsets[2] = {tensor->file_offset,
                               tensor->file_offset + size - take};
        for (int which = 0; ok && which < 2; which++) {
            ssize_t got = pread(fd, buffer, (size_t)take, (off_t)offsets[which]);
            if (got != (ssize_t)take) ok = 0;
            else hash = fnv64(hash, buffer, (size_t)take);
        }
        close(fd);
    }
    free(buffer);
    *out = hash;
    return ok;
}

static int adaln_cache_disabled(void) {
    const char *value = getenv("H3_ADALN_CACHE");
    return value && !strcmp(value, "0");
}

/* <component>/adaln_cache, derived from the shard holding blocks.0. */
static int adaln_cache_dir(const h3_weight_store *weights, char *out,
                           size_t out_size) {
    const char *override = getenv("H3_ADALN_CACHE_DIR");
    if (override && *override) {
        return snprintf(out, out_size, "%s", override) < (int)out_size;
    }
    const h3_st_header *header = NULL;
    if (!h3_weight_find(weights, "blocks.0.adaln_proj.linear.weight", &header) ||
        !header || !header->path) return 0;
    char path[4096];
    if (snprintf(path, sizeof(path), "%s", header->path) >= (int)sizeof(path))
        return 0;
    for (int strip = 0; strip < 2; strip++) {
        char *slash = strrchr(path, '/');
        if (!slash) return 0;
        *slash = '\0';
    }
    return snprintf(out, out_size, "%s/adaln_cache", path) < (int)out_size;
}

typedef struct {
    char path[4200];
    char source[64];
    int usable;
} adaln_cache_key;

static void adaln_cache_key_make(const h3_weight_store *weights,
                                 const float *times, uint32_t rows,
                                 adaln_cache_key *key) {
    memset(key, 0, sizeof(*key));
    if (adaln_cache_disabled()) return;
    char directory[4096];
    uint64_t fingerprint = 0;
    if (!adaln_cache_dir(weights, directory, sizeof(directory)) ||
        !adaln_fingerprint(weights, &fingerprint)) return;
    uint64_t hash = fnv64(fingerprint, &rows, sizeof(rows));
    hash = fnv64(hash, times, (size_t)rows * sizeof(*times));
    snprintf(key->source, sizeof(key->source), "%016llx",
             (unsigned long long)fingerprint);
    if (snprintf(key->path, sizeof(key->path),
                 "%s/adaln-%016llx.safetensors", directory,
                 (unsigned long long)hash) >= (int)sizeof(key->path)) return;
    key->usable = 1;
}

/* Fills blocks[] and *final from the cache file. All-or-nothing: on any
 * mismatch everything loaded is freed and the caller recomputes. */
static int adaln_cache_load(const adaln_cache_key *key, h3_gpu *gpu,
                            const float *times, uint32_t rows,
                            h3_gpu_tensor **blocks, h3_gpu_tensor **final) {
    if (!key->usable) return 0;
    struct stat info;
    if (stat(key->path, &info) != 0) return 0;
    h3_st_header header;
    char error[256];
    if (!h3_st_read_header(key->path, &header, error, sizeof(error))) {
        fprintf(stderr, "h3: AdaLN cache %s unreadable: %s\n", key->path, error);
        return 0;
    }
    int ok = 0;
    float *stored = NULL;
    const char *kind = h3_st_metadata(&header, "h3_quant");
    const char *version = h3_st_metadata(&header, "h3_adaln_version");
    const char *source = h3_st_metadata(&header, "source");
    const h3_st_tensor *t = h3_st_find(&header, "t");
    if (!kind || strcmp(kind, "adaln-table") || !version ||
        strcmp(version, ADALN_CACHE_VERSION) || !source ||
        strcmp(source, key->source) || !t || t->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(t) != rows) goto done;
    stored = malloc((size_t)rows * sizeof(*stored));
    if (!stored || !h3_st_read_data(&header, t, stored,
                                    (size_t)rows * sizeof(*stored),
                                    error, sizeof(error)) ||
        memcmp(stored, times, (size_t)rows * sizeof(*stored))) goto done;
    for (unsigned block = 0; block <= H3_DIT_BLOCKS; block++) {
        char name[32];
        uint64_t width = block < H3_DIT_BLOCKS ? BLOCK_OUTPUT : FINAL_OUTPUT;
        if (block < H3_DIT_BLOCKS) snprintf(name, sizeof(name), "blocks.%u", block);
        else snprintf(name, sizeof(name), "final");
        const h3_st_tensor *tensor = h3_st_find(&header, name);
        if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
            h3_st_tensor_elements(tensor) != (uint64_t)rows * width) goto done;
        h3_gpu_tensor *loaded = h3_gpu_tensor_load_bf16(
            gpu, key->path, tensor->file_offset, (size_t)rows * width);
        if (!loaded) goto done;
        if (block < H3_DIT_BLOCKS) blocks[block] = loaded;
        else *final = loaded;
    }
    ok = 1;
done:
    if (!ok) {
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
            free_tensor(&blocks[block]);
        free_tensor(final);
    }
    free(stored);
    h3_st_free_header(&header);
    return ok;
}

/* Streams the table to <path>.tmp as it is computed, renamed on success.
 * Data order: blocks.0..49, final, t. */
typedef struct {
    FILE *file;
    char temporary[4300];
    const adaln_cache_key *key;
    uint16_t *host;
    int failed;
} adaln_cache_writer;

static void adaln_cache_writer_open(adaln_cache_writer *writer,
                                    const adaln_cache_key *key,
                                    uint32_t rows) {
    memset(writer, 0, sizeof(*writer));
    writer->key = key;
    if (!key->usable) return;
    char directory[4200];
    snprintf(directory, sizeof(directory), "%s", key->path);
    char *slash = strrchr(directory, '/');
    struct stat info;
    if (!slash) return;
    *slash = '\0';
    if (stat(directory, &info) != 0 || !S_ISDIR(info.st_mode)) return;
    size_t header_cap = 64 * 1024;
    char *json = malloc(header_cap);
    writer->host = malloc((size_t)rows * BLOCK_OUTPUT * sizeof(uint16_t));
    if (!json || !writer->host) {
        free(json);
        free(writer->host);
        writer->host = NULL;
        return;
    }
    size_t used = (size_t)snprintf(json, header_cap,
        "{\"__metadata__\":{\"h3_quant\":\"adaln-table\","
        "\"h3_adaln_version\":\"" ADALN_CACHE_VERSION "\","
        "\"source\":\"%s\"}", key->source);
    uint64_t offset = 0;
    for (unsigned block = 0; block <= H3_DIT_BLOCKS + 1; block++) {
        char name[32];
        uint64_t width = block < H3_DIT_BLOCKS ? BLOCK_OUTPUT : FINAL_OUTPUT;
        const char *dtype = "BF16";
        uint64_t bytes = (uint64_t)rows * width * 2;
        if (block < H3_DIT_BLOCKS) {
            snprintf(name, sizeof(name), "blocks.%u", block);
            used += (size_t)snprintf(json + used, header_cap - used,
                ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%u,%llu],"
                "\"data_offsets\":[%llu,%llu]}", name, dtype, rows,
                (unsigned long long)width, (unsigned long long)offset,
                (unsigned long long)(offset + bytes));
        } else if (block == H3_DIT_BLOCKS) {
            used += (size_t)snprintf(json + used, header_cap - used,
                ",\"final\":{\"dtype\":\"BF16\",\"shape\":[%u,%u],"
                "\"data_offsets\":[%llu,%llu]}", rows, (unsigned)FINAL_OUTPUT,
                (unsigned long long)offset,
                (unsigned long long)(offset + bytes));
        } else {
            bytes = (uint64_t)rows * 4;
            used += (size_t)snprintf(json + used, header_cap - used,
                ",\"t\":{\"dtype\":\"F32\",\"shape\":[%u],"
                "\"data_offsets\":[%llu,%llu]}", rows,
                (unsigned long long)offset,
                (unsigned long long)(offset + bytes));
        }
        offset += bytes;
    }
    used += (size_t)snprintf(json + used, header_cap - used, "}");
    while (used % 8) json[used++] = ' ';
    snprintf(writer->temporary, sizeof(writer->temporary), "%s.tmp.%d",
             key->path, (int)getpid());
    writer->file = fopen(writer->temporary, "wb");
    uint64_t header_size = used;
    if (!writer->file ||
        fwrite(&header_size, sizeof(header_size), 1, writer->file) != 1 ||
        fwrite(json, 1, used, writer->file) != used) {
        writer->failed = 1;
    }
    free(json);
}

static void adaln_cache_writer_tensor(adaln_cache_writer *writer,
                                      const h3_gpu_tensor *tensor,
                                      size_t elements) {
    if (!writer->file || writer->failed) return;
    if (!h3_gpu_tensor_read_bf16(tensor, writer->host, elements) ||
        fwrite(writer->host, sizeof(uint16_t), elements, writer->file) !=
            elements) writer->failed = 1;
}

static void adaln_cache_writer_close(adaln_cache_writer *writer,
                                     const float *times, uint32_t rows,
                                     int success) {
    if (writer->file) {
        if (success && !writer->failed &&
            fwrite(times, sizeof(*times), rows, writer->file) != rows)
            writer->failed = 1;
        if (fclose(writer->file) != 0) writer->failed = 1;
        if (success && !writer->failed &&
            rename(writer->temporary, writer->key->path) == 0) {
            fprintf(stderr, "h3: AdaLN table stored %s\n", writer->key->path);
        } else {
            if (success)
                fprintf(stderr, "h3: AdaLN table not stored (%s)\n",
                        strerror(errno));
            unlink(writer->temporary);
        }
    }
    free(writer->host);
    writer->host = NULL;
    writer->file = NULL;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, const h3_lora_set *loras,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    adaln_lora_scratch scratch = {NULL, NULL};
    float *features = NULL;
    float *times = NULL;
    adaln_cache_key cache_key;
    adaln_cache_writer cache_writer;
    memset(&cache_key, 0, sizeof(cache_key));
    memset(&cache_writer, 0, sizeof(cache_writer));
    if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                      &features, &times, error, error_size)) goto failed;
    h3_gpu_tensor *time = time_embeddings(weights, gpu, schedule->time_rows,
                                           features, error, error_size);
    free(features);
    features = NULL;
    if (!time) goto failed;

    adaln_cache_key_make(weights, times, schedule->time_rows, &cache_key);
    int cache_hit = adaln_cache_load(&cache_key, gpu, times,
                                     schedule->time_rows, schedule->blocks,
                                     &schedule->final);
    if (cache_key.usable)
        fprintf(stderr, "h3: AdaLN table cache %s (%u rows)\n",
                cache_hit ? "hit, adaln_proj weights not read" : "miss",
                schedule->time_rows);
    if (!cache_hit)
        adaln_cache_writer_open(&cache_writer, &cache_key,
                                schedule->time_rows);

    uint64_t lora_rank = adaln_lora_rank(loras);
    if (lora_rank) {
        scratch.hidden = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * lora_rank);
        scratch.delta = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!scratch.hidden || !scratch.delta) {
            fail(error, error_size,
                 "cannot allocate the AdaLN LoRA branch scratch: %s",
                 h3_gpu_error(gpu));
            h3_gpu_tensor_free(time);
            goto failed;
        }
    }

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        if (cache_hit) {
            /* The stored table is the base projection; only the LoRA delta,
             * if any, is added on top, exactly as after the linear below. */
            snprintf(operation, sizeof(operation), "AdaLN block %u", block);
            char pair_name[128];
            snprintf(pair_name, sizeof(pair_name),
                     "blocks.%u.adaln_proj.linear", block);
            h3_lora_site site;
            int ok = h3_lora_site_build(loras, gpu, pair_name, &site,
                                        error, error_size) &&
                (!lora_rank || (
                gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
                adaln_lora_dispatch(gpu, &site, &scratch,
                    schedule->blocks[block], time, schedule->time_rows,
                    BLOCK_OUTPUT, error, error_size) &&
                gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                       operation)));
            h3_lora_site_free(&site);
            if (!ok) {
                h3_gpu_tensor_free(time);
                goto failed;
            }
            if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                                   progress_opaque);
            continue;
        }
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, BLOCK_OUTPUT, error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate AdaLN block %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        char pair_name[128];
        snprintf(pair_name, sizeof(pair_name), "blocks.%u.adaln_proj.linear",
                 block);
        h3_lora_site site;
        int ok = h3_lora_site_build(loras, gpu, pair_name, &site,
                                    error, error_size) &&
            gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, H3_DIT_TIME_DIM, BLOCK_OUTPUT),
                error, error_size, operation) &&
            (!cache_writer.file || (
             gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation) &&
             (adaln_cache_writer_tensor(&cache_writer,
                  schedule->blocks[block],
                  (size_t)schedule->time_rows * BLOCK_OUTPUT), 1) &&
             gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                    operation))) &&
            adaln_lora_dispatch(gpu, &site, &scratch,
                schedule->blocks[block], time, schedule->time_rows,
                BLOCK_OUTPUT, error, error_size) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        h3_lora_site_free(&site);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }

    h3_gpu_tensor *final_w = NULL, *final_b = NULL;
    if (cache_hit) {
        h3_lora_site cached_site = {NULL, 0};
        int cached_ok = h3_lora_site_build(
                loras, gpu, "final_layer.adaln_proj.linear", &cached_site,
                error, error_size) &&
            (!lora_rank || (
            gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                   "begin final AdaLN") &&
            adaln_lora_dispatch(gpu, &cached_site, &scratch, schedule->final,
                                time, schedule->time_rows, FINAL_OUTPUT,
                                error, error_size) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                   "submit final AdaLN")));
        h3_lora_site_free(&cached_site);
        if (!cached_ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        goto finished;
    }
    final_w = weight_bf16_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, H3_DIT_TIME_DIM, error, error_size);
    final_b = weight_bf16_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    /* One pair, not one per block: final_layer is the 51st AdaLN target
     *. */
    /* Zeroed up front: the short-circuit below can skip the build, and the
     * free runs either way. */
    h3_lora_site final_site = {NULL, 0};
    int final_ok = final_w && final_b && schedule->final &&
        h3_lora_site_build(loras, gpu, "final_layer.adaln_proj.linear",
                           &final_site, error, error_size) &&
        gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
               "begin final AdaLN") &&
        gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            H3_DIT_TIME_DIM, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") &&
        (!cache_writer.file || (
         gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN") &&
         (adaln_cache_writer_tensor(&cache_writer, schedule->final,
              (size_t)schedule->time_rows * FINAL_OUTPUT), 1) &&
         gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN LoRA"))) &&
        adaln_lora_dispatch(gpu, &final_site, &scratch, schedule->final, time,
                            schedule->time_rows, FINAL_OUTPUT,
                            error, error_size) &&
        gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
               "submit final AdaLN");
    h3_lora_site_free(&final_site);
    if (!final_ok) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    adaln_cache_writer_close(&cache_writer, times, schedule->time_rows, 1);
finished:
    free_adaln_scratch(&scratch);
    h3_gpu_tensor_free(time);
    free(times);
    return schedule;

failed:
    adaln_cache_writer_close(&cache_writer, times, schedule ?
                             schedule->time_rows : 0, 0);
    free(times);
    free(features);
    free_adaln_scratch(&scratch);
    h3_dit_schedule_free(schedule);
    return NULL;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}
