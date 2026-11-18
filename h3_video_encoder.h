#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
/* h3image (MiniMax-H3-Image edit references): encode one image
 * [3,1,H,W] and draw the latent from its posterior as the released recipe
 * does: tile moments blended first, z = mean + std * posterior_noise
 * (posterior_noise is [24,1,H/16,W/16]; NULL takes the mean), rounded to
 * float16, then normalized. The output keeps the 48-channel allocation; only
 * its first 24 channels are the latent. */
int h3_video_vae_encode_image_sampled(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int height, int width,
                        const float *posterior_noise,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
/* te_cache: the GPU half of h3_video_vae_encode_image_sampled. Output keeps
 * the blended [48,1,H,W] moments (mean, then log-variance), stats receives the
 * 24 latents_mean then the 24 latents_std. sample_image_moments draws in place
 * exactly as encode_image_sampled does. */
int h3_video_vae_encode_image_moments(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output, float *stats,
                        char *error, size_t error_size);
void h3_video_vae_sample_image_moments(h3_video_latent *latent,
                                       const float *posterior_noise,
                                       const float *stats);
void h3_video_latent_free(h3_video_latent *latent);

#endif
