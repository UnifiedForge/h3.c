/* te_cache: on-disk cache of the conditioning encoders (see h3_te_cache.h). */
#include "h3_te_cache.h"

#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define TC_MAGIC "H3TC\x01\0\0\0"
#define TC_TEXT_LAYERS 50

static char tc_root[PATH_MAX];
static int tc_on = -1;
static unsigned tc_ref_asked, tc_ref_hit, tc_text_asked, tc_text_hit;

static int tc_enabled(void) {
    if (tc_on < 0) {
        const char *value = getenv("H3_TE_CACHE");
        tc_on = !(value && !strcmp(value, "0"));
    }
    return tc_on && tc_root[0];
}

void h3_tc_begin(const char *model_dir) {
    tc_ref_asked = tc_ref_hit = tc_text_asked = tc_text_hit = 0;
    tc_root[0] = '\0';
    const char *dir = getenv("H3_TE_CACHE_DIR");
    if (dir && *dir) {
        snprintf(tc_root, sizeof(tc_root), "%s", dir);
        return;
    }
    if (!model_dir || !*model_dir) return;
    char model[PATH_MAX];
    snprintf(model, sizeof(model), "%s", model_dir);
    size_t length = strlen(model);
    while (length > 1 && model[length - 1] == '/') model[--length] = '\0';
    char *slash = strrchr(model, '/');
    if (slash) *slash = '\0';
    else snprintf(model, sizeof(model), ".");
    snprintf(tc_root, sizeof(tc_root), "%s/te_cache", model);
}

void h3_tc_report(void) {
    if (!tc_enabled() || (!tc_ref_asked && !tc_text_asked)) return;
    fprintf(stderr, "h3: te_cache: references %u/%u from cache, text %s\n",
            tc_ref_hit, tc_ref_asked,
            !tc_text_asked ? "not asked" :
            tc_text_hit ? "from cache" : "encoded");
}

/* ---- keys ---- */

typedef struct { CC_SHA256_CTX sha; } tc_key;

static void key_bytes(tc_key *key, const void *data, size_t size) {
    const unsigned char *bytes = data;
    while (size) {
        CC_LONG chunk = size > (1u << 30) ? (1u << 30) : (CC_LONG)size;
        CC_SHA256_Update(&key->sha, bytes, chunk);
        bytes += chunk;
        size -= chunk;
    }
}
static void key_u64(tc_key *key, uint64_t value) { key_bytes(key, &value, sizeof(value)); }
static void key_str(tc_key *key, const char *text) {
    size_t length = text ? strlen(text) : 0;
    key_u64(key, length);
    if (length) key_bytes(key, text, length);
}
static void key_stat(tc_key *key, const char *path) {
    struct stat status;
    key_str(key, path ? path : "");
    if (path && stat(path, &status) == 0) {
        key_u64(key, (uint64_t)status.st_size);
        key_u64(key, (uint64_t)status.st_mtimespec.tv_sec);
        key_u64(key, (uint64_t)status.st_mtimespec.tv_nsec);
    } else {
        key_u64(key, UINT64_MAX);
    }
}

static int name_compare(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Weight folder identity: every top-level file, sorted, with size and mtime. */
static void key_dir(tc_key *key, const char *dir) {
    key_str(key, dir);
    DIR *handle = dir ? opendir(dir) : NULL;
    if (!handle) { key_u64(key, UINT64_MAX); return; }
    char **names = NULL;
    size_t count = 0, capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(handle))) {
        if (entry->d_name[0] == '.') continue;
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 32;
            char **grown = realloc(names, capacity * sizeof(*names));
            if (!grown) break;
            names = grown;
        }
        names[count] = strdup(entry->d_name);
        if (names[count]) count++;
    }
    closedir(handle);
    if (count) qsort(names, count, sizeof(*names), name_compare);
    for (size_t index = 0; index < count; index++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", dir, names[index]);
        key_stat(key, path);
        free(names[index]);
    }
    free(names);
}

/* The engine build: this binary and the shader source it compiles. */
static void key_engine(tc_key *key) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) exe[0] = '\0';
    if (exe[0] && realpath(exe, real)) snprintf(exe, sizeof(exe), "%s", real);
    key_stat(key, exe);
    char shader[PATH_MAX];
    snprintf(shader, sizeof(shader), "%s", exe);
    char *slash = strrchr(shader, '/');
    if (slash) snprintf(slash + 1, sizeof(shader) - (size_t)(slash + 1 - shader),
                        "h3_shaders.metal");
    key_stat(key, shader);
    key_stat(key, "h3_shaders.metal");
}

static void key_begin(tc_key *key, const char *kind, const char *weights) {
    CC_SHA256_Init(&key->sha);
    key_str(key, "h3 te_cache v1");
    key_str(key, kind);
    key_engine(key);
    key_dir(key, weights);
}

static void key_vision(tc_key *key, const h3_vision_output *vision) {
    key_u64(key, (uint64_t)vision->grid_h);
    key_u64(key, (uint64_t)vision->grid_w);
    key_u64(key, vision->tokens);
    size_t bytes = vision->tokens * H3_VISION_OUTPUT_WIDTH * sizeof(uint16_t);
    key_bytes(key, vision->merged, bytes);
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
        key_bytes(key, vision->deepstack[index], bytes);
}

static void key_path(tc_key *key, const char *kind, char *path, size_t size,
                     unsigned char digest[CC_SHA256_DIGEST_LENGTH]) {
    CC_SHA256_Final(digest, &key->sha);
    char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (int index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        snprintf(hex + 2 * index, 3, "%02x", digest[index]);
    snprintf(path, size, "%s/%s/%s.h3tc", tc_root, kind, hex);
}

/* ---- files ---- */

typedef struct { unsigned char *data; size_t size, capacity; int failed; } tc_out;

static void out_put(tc_out *out, const void *data, size_t size) {
    if (out->failed) return;
    if (out->size + size > out->capacity) {
        size_t capacity = out->capacity ? out->capacity : 1 << 20;
        while (capacity < out->size + size) capacity *= 2;
        unsigned char *grown = realloc(out->data, capacity);
        if (!grown) { out->failed = 1; return; }
        out->data = grown;
        out->capacity = capacity;
    }
    memcpy(out->data + out->size, data, size);
    out->size += size;
}
static void out_u64(tc_out *out, uint64_t value) { out_put(out, &value, sizeof(value)); }

typedef struct { unsigned char *data; size_t size, at; int failed; } tc_in;

static int in_get(tc_in *in, void *data, size_t size) {
    if (in->failed || in->size - in->at < size) { in->failed = 1; return 0; }
    memcpy(data, in->data + in->at, size);
    in->at += size;
    return 1;
}
static uint64_t in_u64(tc_in *in) { uint64_t value = 0; in_get(in, &value, sizeof(value)); return value; }
static void *in_array(tc_in *in, size_t count, size_t element) {
    if (in->failed || (element && count > SIZE_MAX / element)) { in->failed = 1; return NULL; }
    void *data = malloc(count * element ? count * element : 1);
    if (!data || !in_get(in, data, count * element)) { free(data); in->failed = 1; return NULL; }
    return data;
}

static int file_load(const char *path, const unsigned char digest[CC_SHA256_DIGEST_LENGTH],
                     tc_in *in) {
    memset(in, 0, sizeof(*in));
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    struct stat status;
    if (fstat(fileno(file), &status) != 0 || status.st_size < 40) { fclose(file); return 0; }
    in->size = (size_t)status.st_size;
    in->data = malloc(in->size);
    int ok = in->data && fread(in->data, 1, in->size, file) == in->size;
    fclose(file);
    if (ok) ok = !memcmp(in->data, TC_MAGIC, 8) &&
                 !memcmp(in->data + 8, digest, CC_SHA256_DIGEST_LENGTH);
    if (!ok) { free(in->data); memset(in, 0, sizeof(*in)); return 0; }
    in->at = 8 + CC_SHA256_DIGEST_LENGTH;
    (void)utimes(path, NULL);   /* LRU: a hit is a use */
    return 1;
}

typedef struct { char *path; off_t size; struct timespec mtime; } tc_file;

static int file_age_compare(const void *a, const void *b) {
    const tc_file *x = a, *y = b;
    if (x->mtime.tv_sec != y->mtime.tv_sec) return x->mtime.tv_sec < y->mtime.tv_sec ? -1 : 1;
    if (x->mtime.tv_nsec != y->mtime.tv_nsec) return x->mtime.tv_nsec < y->mtime.tv_nsec ? -1 : 1;
    return 0;
}

static void evict(void) {
    const char *value = getenv("H3_TE_CACHE_MAX_GB");
    double gigabytes = value && *value ? atof(value) : 10.0;
    if (!(gigabytes > 0)) gigabytes = 10.0;
    off_t limit = (off_t)(gigabytes * 1024.0 * 1024.0 * 1024.0);
    static const char *kinds[] = {"vae", "vaem", "vision", "audio", "text"};
    tc_file *files = NULL;
    size_t count = 0, capacity = 0;
    off_t total = 0;
    for (size_t kind = 0; kind < sizeof(kinds) / sizeof(*kinds); kind++) {
        char dir[PATH_MAX];
        snprintf(dir, sizeof(dir), "%s/%s", tc_root, kinds[kind]);
        DIR *handle = opendir(dir);
        if (!handle) continue;
        struct dirent *entry;
        while ((entry = readdir(handle))) {
            size_t length = strlen(entry->d_name);
            if (length < 6 || strcmp(entry->d_name + length - 5, ".h3tc")) continue;
            char path[PATH_MAX];
            snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);
            struct stat status;
            if (stat(path, &status) != 0) continue;
            if (count == capacity) {
                capacity = capacity ? capacity * 2 : 256;
                tc_file *grown = realloc(files, capacity * sizeof(*files));
                if (!grown) break;
                files = grown;
            }
            files[count].path = strdup(path);
            files[count].size = status.st_size;
            files[count].mtime = status.st_mtimespec;
            if (files[count].path) { total += status.st_size; count++; }
        }
        closedir(handle);
    }
    if (total > limit && count) {
        qsort(files, count, sizeof(*files), file_age_compare);
        off_t target = limit / 10 * 9;
        unsigned removed = 0;
        for (size_t index = 0; index < count && total > target; index++)
            if (unlink(files[index].path) == 0) { total -= files[index].size; removed++; }
        fprintf(stderr, "h3: te_cache: over %.1f GiB, %u oldest entries removed\n",
                gigabytes, removed);
    }
    for (size_t index = 0; index < count; index++) free(files[index].path);
    free(files);
}

static void file_store(const char *path, const unsigned char digest[CC_SHA256_DIGEST_LENGTH],
                       tc_out *out) {
    if (out->failed) { free(out->data); return; }
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = '\0';
    (void)mkdir(tc_root, 0755);
    (void)mkdir(dir, 0755);
    char temporary[PATH_MAX];
    snprintf(temporary, sizeof(temporary), "%s.tmp.%d", path, (int)getpid());
    FILE *file = fopen(temporary, "wb");
    int ok = file != NULL;
    if (ok) ok = fwrite(TC_MAGIC, 1, 8, file) == 8 &&
                 fwrite(digest, 1, CC_SHA256_DIGEST_LENGTH, file) == CC_SHA256_DIGEST_LENGTH &&
                 fwrite(out->data, 1, out->size, file) == out->size;
    if (file && fclose(file) != 0) ok = 0;
    if (ok) ok = rename(temporary, path) == 0;
    if (!ok) {
        (void)unlink(temporary);
        fprintf(stderr, "h3: te_cache: warning: cannot write %s\n", path);
    }
    free(out->data);
    if (ok) evict();
}

/* ---- serialisation of the encoder outputs ---- */

static void put_video(tc_out *out, const h3_video_latent *latent, size_t values) {
    out_u64(out, (uint64_t)latent->time);
    out_u64(out, (uint64_t)latent->height);
    out_u64(out, (uint64_t)latent->width);
    out_u64(out, values);
    out_put(out, latent->values, values * sizeof(float));
}
static int get_video(tc_in *in, h3_video_latent *latent) {
    memset(latent, 0, sizeof(*latent));
    latent->time = (int)in_u64(in);
    latent->height = (int)in_u64(in);
    latent->width = (int)in_u64(in);
    size_t values = (size_t)in_u64(in);
    latent->values = in_array(in, values, sizeof(float));
    return !in->failed;
}

static void put_vision(tc_out *out, const h3_vision_output *vision) {
    out_u64(out, (uint64_t)vision->grid_h);
    out_u64(out, (uint64_t)vision->grid_w);
    out_u64(out, vision->tokens);
    size_t bytes = vision->tokens * H3_VISION_OUTPUT_WIDTH * sizeof(uint16_t);
    out_put(out, vision->merged, bytes);
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
        out_put(out, vision->deepstack[index], bytes);
}
static int get_vision(tc_in *in, h3_vision_output *vision) {
    memset(vision, 0, sizeof(*vision));
    vision->grid_h = (int)in_u64(in);
    vision->grid_w = (int)in_u64(in);
    vision->tokens = (size_t)in_u64(in);
    size_t elements = vision->tokens * H3_VISION_OUTPUT_WIDTH;
    vision->merged = in_array(in, elements, sizeof(uint16_t));
    for (unsigned index = 0; index < H3_VISION_DEEPSTACKS; index++)
        vision->deepstack[index] = in_array(in, elements, sizeof(uint16_t));
    if (in->failed) { h3_vision_output_free(vision); return 0; }
    return 1;
}

static void put_text(tc_out *out, const h3_text_embedding *text) {
    out_u64(out, text->tokens);
    out_u64(out, text->width);
    out_put(out, text->values, text->tokens * text->width * sizeof(uint16_t));
    out_u64(out, text->tags ? 1 : 0);
    if (text->tags) out_put(out, text->tags, text->tokens);
}
static int get_text(tc_in *in, h3_text_embedding *text) {
    memset(text, 0, sizeof(*text));
    text->tokens = (size_t)in_u64(in);
    text->width = (size_t)in_u64(in);
    if (!in->failed && text->width && text->tokens > SIZE_MAX / text->width) in->failed = 1;
    text->values = in_array(in, text->tokens * text->width, sizeof(uint16_t));
    if (in_u64(in)) text->tags = in_array(in, text->tokens, 1);
    if (in->failed) { h3_text_embedding_free(text); return 0; }
    return 1;
}

/* ---- per reference ---- */

int h3_tc_video_vae_encode(const char *weight_directory,
                           const char *shader_source_path,
                           const float *pixels, int frames, int height, int width,
                           h3_video_encoder_progress progress, void *progress_opaque,
                           h3_video_latent *output,
                           char *error, size_t error_size) {
    if (!tc_enabled() || !pixels || frames < 1 || height < 1 || width < 1)
        return h3_video_vae_encode(weight_directory, shader_source_path, pixels,
                                   frames, height, width, progress,
                                   progress_opaque, output, error, error_size);
    tc_ref_asked++;
    tc_key key;
    key_begin(&key, "video VAE", weight_directory);
    key_u64(&key, (uint64_t)frames);
    key_u64(&key, (uint64_t)height);
    key_u64(&key, (uint64_t)width);
    key_bytes(&key, pixels, (size_t)3 * frames * height * width * sizeof(float));
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    key_path(&key, "vae", path, sizeof(path), digest);
    tc_in in;
    if (file_load(path, digest, &in)) {
        int ok = get_video(&in, output);
        free(in.data);
        if (ok) {
            tc_ref_hit++;
            if (progress) progress(1, 1, progress_opaque);
            return 1;
        }
        h3_video_latent_free(output);
    }
    if (!h3_video_vae_encode(weight_directory, shader_source_path, pixels,
                             frames, height, width, progress, progress_opaque,
                             output, error, error_size)) return 0;
    tc_out out = {0};
    put_video(&out, output, (size_t)24 * output->time * output->height * output->width);
    file_store(path, digest, &out);
    return 1;
}

int h3_tc_video_vae_encode_image_sampled(
                           const char *weight_directory,
                           const char *shader_source_path,
                           const float *pixels, int height, int width,
                           const float *posterior_noise,
                           h3_video_encoder_progress progress, void *progress_opaque,
                           h3_video_latent *output,
                           char *error, size_t error_size) {
    if (!tc_enabled() || !pixels || height < 1 || width < 1)
        return h3_video_vae_encode_image_sampled(
            weight_directory, shader_source_path, pixels, height, width,
            posterior_noise, progress, progress_opaque, output, error, error_size);
    tc_ref_asked++;
    tc_key key;
    key_begin(&key, "image VAE moments", weight_directory);
    key_u64(&key, (uint64_t)height);
    key_u64(&key, (uint64_t)width);
    key_bytes(&key, pixels, (size_t)3 * height * width * sizeof(float));
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    key_path(&key, "vaem", path, sizeof(path), digest);
    float stats[48];
    int have = 0;
    tc_in in;
    if (file_load(path, digest, &in)) {
        have = get_video(&in, output) && in_get(&in, stats, sizeof(stats));
        free(in.data);
        if (have) tc_ref_hit++;
        else h3_video_latent_free(output);
        if (have && progress) progress(1, 1, progress_opaque);
    }
    if (!have) {
        if (!h3_video_vae_encode_image_moments(
                weight_directory, shader_source_path, pixels, height, width,
                progress, progress_opaque, output, stats, error, error_size))
            return 0;
        tc_out out = {0};
        put_video(&out, output, (size_t)48 * output->time * output->height * output->width);
        out_put(&out, stats, sizeof(stats));
        file_store(path, digest, &out);
    }
    h3_video_vae_sample_image_moments(output, posterior_noise, stats);
    return 1;
}

int h3_tc_vision_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_vision_progress progress, void *progress_opaque,
                        h3_vision_output *output,
                        char *error, size_t error_size) {
    if (!tc_enabled() || !pixels || frames < 1 || height < 1 || width < 1)
        return h3_vision_encode_bf16(weight_directory, shader_source_path, pixels,
                                     frames, height, width, progress,
                                     progress_opaque, output, error, error_size);
    tc_ref_asked++;
    tc_key key;
    key_begin(&key, "Qwen vision", weight_directory);
    key_u64(&key, (uint64_t)frames);
    key_u64(&key, (uint64_t)height);
    key_u64(&key, (uint64_t)width);
    key_bytes(&key, pixels, (size_t)3 * frames * height * width * sizeof(float));
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    key_path(&key, "vision", path, sizeof(path), digest);
    tc_in in;
    if (file_load(path, digest, &in)) {
        int ok = get_vision(&in, output);
        free(in.data);
        if (ok) {
            tc_ref_hit++;
            if (progress) progress(1, 1, progress_opaque);
            return 1;
        }
    }
    if (!h3_vision_encode_bf16(weight_directory, shader_source_path, pixels,
                               frames, height, width, progress, progress_opaque,
                               output, error, error_size)) return 0;
    tc_out out = {0};
    put_vision(&out, output);
    file_store(path, digest, &out);
    return 1;
}

int h3_tc_audio_vae_encode(const char *weight_directory,
                           const char *shader_source_path,
                           const float *pcm, int samples,
                           h3_audio_vae_progress progress, void *progress_opaque,
                           h3_audio_latent *output,
                           char *error, size_t error_size) {
    if (!tc_enabled() || !pcm || samples < 1)
        return h3_audio_vae_encode(weight_directory, shader_source_path, pcm,
                                   samples, progress, progress_opaque, output,
                                   error, error_size);
    tc_ref_asked++;
    tc_key key;
    key_begin(&key, "audio VAE", weight_directory);
    key_u64(&key, (uint64_t)samples);
    key_bytes(&key, pcm, (size_t)2 * samples * sizeof(float));
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    key_path(&key, "audio", path, sizeof(path), digest);
    tc_in in;
    if (file_load(path, digest, &in)) {
        memset(output, 0, sizeof(*output));
        output->channels = (int)in_u64(&in);
        output->stereo = (int)in_u64(&in);
        output->length = (int)in_u64(&in);
        size_t values = (size_t)output->channels * output->stereo * output->length;
        output->values = in_array(&in, values, sizeof(float));
        int ok = !in.failed;
        free(in.data);
        if (ok) {
            tc_ref_hit++;
            if (progress) progress(1, 1, progress_opaque);
            return 1;
        }
        h3_audio_latent_free(output);
    }
    if (!h3_audio_vae_encode(weight_directory, shader_source_path, pcm, samples,
                             progress, progress_opaque, output, error, error_size))
        return 0;
    tc_out out = {0};
    out_u64(&out, (uint64_t)output->channels);
    out_u64(&out, (uint64_t)output->stereo);
    out_u64(&out, (uint64_t)output->length);
    out_put(&out, output->values, (size_t)output->channels * output->stereo *
                                  output->length * sizeof(float));
    file_store(path, digest, &out);
    return 1;
}

/* ---- the text encoder ---- */

static int text_lookup(tc_key *key, char *path, size_t size,
                       unsigned char digest[CC_SHA256_DIGEST_LENGTH],
                       h3_text_embedding *output, uint32_t **ids_out,
                       size_t *id_count_out,
                       h3_text_progress progress, void *progress_opaque) {
    tc_text_asked++;
    key_path(key, "text", path, size, digest);
    tc_in in;
    if (!file_load(path, digest, &in)) return 0;
    int ok = get_text(&in, output);
    uint32_t *ids = NULL;
    size_t id_count = 0;
    if (ok && in_u64(&in)) {
        id_count = (size_t)in_u64(&in);
        ids = in_array(&in, id_count, sizeof(uint32_t));
        ok = !in.failed;
    }
    free(in.data);
    if (ok && ids_out && !ids) ok = 0;
    if (!ok) { h3_text_embedding_free(output); free(ids); return 0; }
    if (ids_out) { *ids_out = ids; *id_count_out = id_count; }
    else free(ids);
    tc_text_hit++;
    if (progress) progress(TC_TEXT_LAYERS, TC_TEXT_LAYERS, progress_opaque);
    return 1;
}

static void text_store(const char *path, const unsigned char *digest,
                       const h3_text_embedding *output,
                       const uint32_t *ids, size_t id_count) {
    tc_out out = {0};
    put_text(&out, output);
    out_u64(&out, ids ? 1 : 0);
    if (ids) {
        out_u64(&out, id_count);
        out_put(&out, ids, id_count * sizeof(uint32_t));
    }
    file_store(path, digest, &out);
}

int h3_tc_text_encode(const char *weight_directory,
                      const char *shader_source_path,
                      const uint32_t *token_ids, size_t token_count,
                      h3_text_progress progress, void *progress_opaque,
                      h3_text_embedding *output,
                      char *error, size_t error_size) {
    if (!tc_enabled() || !token_ids)
        return h3_text_encode_bf16(weight_directory, shader_source_path,
                                   token_ids, token_count, progress,
                                   progress_opaque, output, error, error_size);
    tc_key key;
    key_begin(&key, "text ids", weight_directory);
    key_u64(&key, token_count);
    key_bytes(&key, token_ids, token_count * sizeof(*token_ids));
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (text_lookup(&key, path, sizeof(path), digest, output, NULL, NULL,
                    progress, progress_opaque)) return 1;
    if (!h3_text_encode_bf16(weight_directory, shader_source_path, token_ids,
                             token_count, progress, progress_opaque, output,
                             error, error_size)) return 0;
    text_store(path, digest, output, NULL, 0);
    return 1;
}

int h3_tc_fl2va(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                const char *weight_directory, const char *shader_source_path,
                const char *prompt,
                const h3_vision_output *images, size_t image_count,
                h3_text_progress progress, void *progress_opaque,
                h3_text_embedding *output,
                char *error, size_t error_size) {
    if (!tc_enabled() || !prompt)
        return h3_multimodal_encode_fl2va_bf16(
            tokenizer, weight_directory, shader_source_path, prompt, images,
            image_count, progress, progress_opaque, output, error, error_size);
    tc_key key;
    key_begin(&key, "text fl2va", weight_directory);
    key_stat(&key, tokenizer_path);
    key_str(&key, prompt);
    key_u64(&key, image_count);
    for (size_t index = 0; index < image_count; index++) key_vision(&key, &images[index]);
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (text_lookup(&key, path, sizeof(path), digest, output, NULL, NULL,
                    progress, progress_opaque)) return 1;
    if (!h3_multimodal_encode_fl2va_bf16(
            tokenizer, weight_directory, shader_source_path, prompt, images,
            image_count, progress, progress_opaque, output, error, error_size))
        return 0;
    text_store(path, digest, output, NULL, 0);
    return 1;
}

int h3_tc_ref2va(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                 const char *weight_directory, const char *shader_source_path,
                 const char *prompt,
                 const h3_reference_presentation *references,
                 size_t reference_count,
                 h3_text_progress progress, void *progress_opaque,
                 h3_text_embedding *output,
                 char *error, size_t error_size) {
    if (!tc_enabled() || !prompt || (reference_count && !references))
        return h3_multimodal_encode_ref2va_bf16(
            tokenizer, weight_directory, shader_source_path, prompt, references,
            reference_count, progress, progress_opaque, output, error, error_size);
    tc_key key;
    key_begin(&key, "text ref2va", weight_directory);
    key_stat(&key, tokenizer_path);
    key_str(&key, prompt);
    key_u64(&key, reference_count);
    for (size_t index = 0; index < reference_count; index++) {
        const h3_reference_presentation *reference = &references[index];
        key_u64(&key, (uint64_t)reference->kind);
        key_u64(&key, (uint64_t)reference->has_audio);
        key_u64(&key, reference->vision_count);
        key_u64(&key, reference->timestamps ? 1 : 0);
        if (reference->timestamps)
            key_bytes(&key, reference->timestamps,
                      reference->vision_count * sizeof(*reference->timestamps));
        key_u64(&key, reference->vision ? 1 : 0);
        if (reference->vision)
            for (size_t block = 0; block < reference->vision_count; block++)
                key_vision(&key, &reference->vision[block]);
    }
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (text_lookup(&key, path, sizeof(path), digest, output, NULL, NULL,
                    progress, progress_opaque)) return 1;
    if (!h3_multimodal_encode_ref2va_bf16(
            tokenizer, weight_directory, shader_source_path, prompt, references,
            reference_count, progress, progress_opaque, output, error, error_size))
        return 0;
    text_store(path, digest, output, NULL, 0);
    return 1;
}

int h3_tc_image_edit(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                     const char *weight_directory, const char *shader_source_path,
                     const char *prompt, size_t max_prompt_tokens,
                     const h3_vision_output *images, size_t image_count,
                     h3_text_progress progress, void *progress_opaque,
                     h3_text_embedding *output,
                     uint32_t **ids_out, size_t *id_count_out,
                     char *error, size_t error_size) {
    if (!tc_enabled() || !prompt || !ids_out || !id_count_out)
        return h3_multimodal_encode_image_edit_bf16(
            tokenizer, weight_directory, shader_source_path, prompt,
            max_prompt_tokens, images, image_count, progress, progress_opaque,
            output, ids_out, id_count_out, error, error_size);
    tc_key key;
    key_begin(&key, "text image edit", weight_directory);
    key_stat(&key, tokenizer_path);
    key_str(&key, prompt);
    key_u64(&key, max_prompt_tokens);
    key_u64(&key, image_count);
    for (size_t index = 0; index < image_count; index++) key_vision(&key, &images[index]);
    char path[PATH_MAX];
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (text_lookup(&key, path, sizeof(path), digest, output, ids_out,
                    id_count_out, progress, progress_opaque)) return 1;
    if (!h3_multimodal_encode_image_edit_bf16(
            tokenizer, weight_directory, shader_source_path, prompt,
            max_prompt_tokens, images, image_count, progress, progress_opaque,
            output, ids_out, id_count_out, error, error_size)) return 0;
    text_store(path, digest, output, *ids_out, *id_count_out);
    return 1;
}
