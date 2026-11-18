/* te_cache: an on-disk cache of the conditioning encoders' outputs, so a
 * repeated prompt or a reference seen before skips the encoder and its
 * weights entirely (Q5.1).
 *
 * Two levels, both lossless:
 *   - per reference: video VAE latent (or the H3 Image VAE moments, before the
 *     seeded posterior draw), Qwen vision output, audio VAE latent;
 *   - the whole text encoder output for one presentation.
 *
 * Every key is a SHA-256 of the exact bytes the encoder would read (pixels or
 * PCM after FFmpeg, token ids, prompt, upstream vision outputs), the encoder's
 * weight files (name, size, mtime), the h3 binary and its shader source. A hit
 * is therefore the same bytes the encoder would have produced; any change of
 * input, weights or engine build is a miss by construction.
 *
 * Each wrapper has the signature of the function it caches and falls back to
 * it on any cache problem: the cache never fails a run.
 *
 * Env: H3_TE_CACHE=0 off; H3_TE_CACHE_DIR (default <engine>/te_cache, next to
 * the model folder); H3_TE_CACHE_MAX_GB (default 10, LRU by file mtime). */
#ifndef H3_TE_CACHE_H
#define H3_TE_CACHE_H

#include "h3_audio_vae.h"
#include "h3_multimodal.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_vision_encoder.h"

#include <stddef.h>
#include <stdint.h>

/* Called once per generation with the model folder; resets the hit counters. */
void h3_tc_begin(const char *model_dir);
/* One summary line on stderr: "h3: te_cache: references 3/4 from cache,
 * text from cache" (nothing when the cache is off or nothing was asked). */
void h3_tc_report(void);

int h3_tc_video_vae_encode(const char *weight_directory,
                           const char *shader_source_path,
                           const float *pixels, int frames, int height, int width,
                           h3_video_encoder_progress progress, void *progress_opaque,
                           h3_video_latent *output,
                           char *error, size_t error_size);
int h3_tc_video_vae_encode_image_sampled(
                           const char *weight_directory,
                           const char *shader_source_path,
                           const float *pixels, int height, int width,
                           const float *posterior_noise,
                           h3_video_encoder_progress progress, void *progress_opaque,
                           h3_video_latent *output,
                           char *error, size_t error_size);
int h3_tc_vision_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_vision_progress progress, void *progress_opaque,
                        h3_vision_output *output,
                        char *error, size_t error_size);
int h3_tc_audio_vae_encode(const char *weight_directory,
                           const char *shader_source_path,
                           const float *pcm, int samples,
                           h3_audio_vae_progress progress, void *progress_opaque,
                           h3_audio_latent *output,
                           char *error, size_t error_size);

int h3_tc_text_encode(const char *weight_directory,
                      const char *shader_source_path,
                      const uint32_t *token_ids, size_t token_count,
                      h3_text_progress progress, void *progress_opaque,
                      h3_text_embedding *output,
                      char *error, size_t error_size);
int h3_tc_fl2va(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                const char *weight_directory, const char *shader_source_path,
                const char *prompt,
                const h3_vision_output *images, size_t image_count,
                h3_text_progress progress, void *progress_opaque,
                h3_text_embedding *output,
                char *error, size_t error_size);
int h3_tc_ref2va(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                 const char *weight_directory, const char *shader_source_path,
                 const char *prompt,
                 const h3_reference_presentation *references,
                 size_t reference_count,
                 h3_text_progress progress, void *progress_opaque,
                 h3_text_embedding *output,
                 char *error, size_t error_size);
int h3_tc_image_edit(const char *tokenizer_path, const h3_tokenizer *tokenizer,
                     const char *weight_directory, const char *shader_source_path,
                     const char *prompt, size_t max_prompt_tokens,
                     const h3_vision_output *images, size_t image_count,
                     h3_text_progress progress, void *progress_opaque,
                     h3_text_embedding *output,
                     uint32_t **ids_out, size_t *id_count_out,
                     char *error, size_t error_size);

#endif
