/* Unit test of rf_gen (PC gcc and ucsim/SDCC): generate the first N_RF dendrites of seed
 * constant RF_C and dump base (LE16) + 16 offsets per dendrite = 18 bytes each.
 * PC: rf_test <out.bin>.  ucsim: dump lands in xdata rfout[], then parks in sim_done(). */
#include "prng_rf.h"
#ifndef N_RF
#define N_RF 64
#endif
#ifdef __SDCC
#include "cyc.h"
XDATA __at(0x0400) uint8_t rfout[N_RF * 18];
XDATA __at(0x0300) uint8_t res[16];   /* [0..3] cycles for N_RF rf_gen calls, [4..7] overhead, [8]=0xA5 */
void sim_done(void) { while (1); }
void main(void)
{
    uint16_t n, s = RF_C, b;
    uint8_t t;
    XDATA uint8_t *p = rfout;
    uint32_t cy, cy0;
    for (n = 0; n < N_RF; n++) {
        s += RF_K;
        b = rf_gen(s);
        *p++ = b; *p++ = b >> 8;
        for (t = 0; t < RF_SYN; t++) *p++ = rf_off[t];
    }
    /* timing: N_RF generations only (same loop without the dump) */
    cyc_init();
    cyc_start(); cy0 = cyc_stop();
    s = RF_C;
    cyc_start();
    for (n = 0; n < N_RF; n++) { s += RF_K; rf_gen(s); }
    cy = cyc_stop();
    res[0] = cy; res[1] = cy >> 8; res[2] = cy >> 16; res[3] = cy >> 24;
    res[4] = cy0; res[5] = cy0 >> 8; res[6] = cy0 >> 16; res[7] = cy0 >> 24;
    res[8] = 0xA5;
    sim_done();
}
#else
#include <stdio.h>
int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "wb");
    uint16_t n, s = RF_C, b;
    if (!f) return 1;
    for (n = 0; n < N_RF; n++) {
        s += RF_K;
        b = rf_gen(s);
        fputc(b & 0xFF, f); fputc(b >> 8, f);
        fwrite(rf_off, 1, RF_SYN, f);
    }
    fclose(f);
    return 0;
}
#endif
