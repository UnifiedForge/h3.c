/* accel: time one 17-frame guide clip through h3_video_vae_encode with the
 * scratch pool off and on, and check the latents are bit-identical.
 * usage: h3_encoder_bench VAE_DIR VIDEO WIDTH HEIGHT */
#include "h3_ffmpeg.h"
#include "h3_video_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
    if (!h3_ffmpeg_read_video_f32(argv[2], width, height, 22, &pixels, &frames,
                                  error, sizeof(error)) || frames < 17) {
        fprintf(stderr, "read: %s (%d frames)\n", error, frames);
        return 1;
    }
    /* channel-major [3][frames][h][w]: keep the first 17 frames */
    size_t area = (size_t)width * height;
    for (int channel = 1; channel < 3; channel++)
        memmove(pixels + (size_t)channel * 17 * area,
                pixels + (size_t)channel * frames * area, 17 * area * sizeof(float));
    h3_video_latent latent[2];
    double seconds[2];
    for (int run = 0; run < 2; run++) {
        setenv("H3_ENCODER_POOL", run ? "1" : "0", 1);
        if (run && getenv("BENCH_TILE")) setenv("H3_ENCODER_TILE", getenv("BENCH_TILE"), 1);
        if (run && getenv("BENCH_AS2D")) setenv("H3_CONV3D_AS_2D", "1", 1);
        if (run && getenv("BENCH_HALF")) setenv("H3_CONV3D_HALF", getenv("BENCH_HALF"), 1);
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
    double num = 0, den = 0, max = 0;
    for (size_t i = 0; i < count && latent[0].time == latent[1].time; i++) {
        double d = (double)latent[1].values[i] - latent[0].values[i];
        num += d * d; den += (double)latent[0].values[i] * latent[0].values[i];
        if (d < 0) d = -d;
        if (d > max) max = d;
    }
    printf("bit_identical=%s rel_l2=%.3e max_abs=%.3e speedup=%.2fx\n",
           same ? "yes" : "NO", den > 0 ? sqrt(num / den) : 0.0, max,
           seconds[0] / seconds[1]);
    return 0;
}
