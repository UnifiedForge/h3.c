/* accel: time one 17-frame guide clip through h3_video_vae_encode with the
 * scratch pool off and on, and check the latents are bit-identical.
 * usage: h3_encoder_bench VAE_DIR VIDEO WIDTH HEIGHT */
#include "h3_ffmpeg.h"
#include "h3_video_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s VAE_DIR VIDEO W H\n", argv[0]); return 2; }
    int width = atoi(argv[3]), height = atoi(argv[4]);
    char error[512];
    float *pixels = NULL;
    int frames = 0;
    if (!h3_ffmpeg_read_video_f32(argv[2], width, height, 17, &pixels, &frames,
                                  error, sizeof(error)) || frames < 17) {
        fprintf(stderr, "read: %s (%d frames)\n", error, frames);
        return 1;
    }
    h3_video_latent latent[2];
    double seconds[2];
    for (int run = 0; run < 2; run++) {
        setenv("H3_ENCODER_POOL", run ? "1" : "0", 1);
        double started = now();
        if (!h3_video_vae_encode(argv[1], "h3_shaders.metal", pixels, 17,
                                 height, width, NULL, NULL, &latent[run],
                                 error, sizeof(error))) {
            fprintf(stderr, "encode: %s\n", error);
            return 1;
        }
        seconds[run] = now() - started;
        printf("pool=%s seconds=%.2f latent=%dx%dx%d\n", run ? "on" : "off",
               seconds[run], latent[run].time, latent[run].height,
               latent[run].width);
    }
    size_t count = (size_t)24 * latent[0].time * latent[0].height * latent[0].width;
    int same = latent[0].time == latent[1].time &&
        !memcmp(latent[0].values, latent[1].values, count * sizeof(float));
    printf("bit_identical=%s speedup=%.2fx\n", same ? "yes" : "NO",
           seconds[0] / seconds[1]);
    return same ? 0 : 3;
}
