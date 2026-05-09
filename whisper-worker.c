/**
 * whisper-worker.c
 * 微型 whisper.cpp 常驻进程 — stdin/stdout 协议
 *
 * 编译: gcc -O2 -o whisper-worker whisper-worker.c -lwhisper -I/opt/homebrew/include
 *
 * 协议:
 *   TRANSCRIBE <lang>\n<4字节长度><WAV数据>
 *   RESULT <text>\n
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "whisper.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context *ctx = whisper_init_from_file_with_params(argv[1], cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("READY\n");
    fflush(stdout);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress   = false;
    params.print_special    = false;
    params.print_realtime   = false;
    params.print_timestamps = false;
    params.no_timestamps    = true;

    char cmd[64];
    char lang[16];
    uint32_t audio_len;

    while (1) {
        if (scanf("%63s", cmd) != 1) break;
        if (strcmp(cmd, "TRANSCRIBE") == 0) {
            if (scanf("%15s %u", lang, &audio_len) != 2) break;
            // consume newline
            getchar();

            float *audio = (float *)malloc(audio_len * sizeof(float));
            if (!audio) break;
            if (fread(audio, sizeof(float), audio_len, stdin) != audio_len) {
                free(audio);
                break;
            }

            params.language = lang;
            int ret = whisper_full(ctx, params, audio, audio_len);
            free(audio);

            if (ret == 0) {
                int n = whisper_full_n_segments(ctx);
                printf("RESULT ");
                for (int i = 0; i < n; i++) {
                    const char *text = whisper_full_get_segment_text(ctx, i);
                    if (text) printf("%s", text);
                }
                printf("\n");
            } else {
                printf("RESULT \n");
            }
            fflush(stdout);
        } else if (strcmp(cmd, "EXIT") == 0) {
            break;
        }
    }

    whisper_free(ctx);
    return 0;
}
