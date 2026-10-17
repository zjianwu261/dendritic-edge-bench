/* PC build of the same infer.c (gcc) for bit-exactness checks:
 * pc_main <images.u8 (N*784)> <out_logits.i32 (N*10 int32 + class byte file)> */
#include <stdio.h>
#include <stdlib.h>
#include "infer.h"
int main(int argc, char **argv)
{
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    uint8_t img[N_IN];
    if (!fi || !fo) return 1;
    while (fread(img, 1, N_IN, fi) == N_IN) {
        uint8_t c = infer(img);
        fwrite(logits, sizeof(int32_t), N_OUT, fo);
        fwrite(&c, 1, 1, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
