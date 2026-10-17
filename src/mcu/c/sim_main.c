/* ucsim (s51) driver: image preloaded by the simulator script into img[] (fixed xdata
 * address), result written to res[] then execution parks in sim_done(). */
#include "infer.h"
#include "cyc.h"
#define MODEL_DEFINES_ONLY   /* model.h: constants only, no weight tables in this unit */
#include "model.h"
#ifdef MODEL_DANN_PRNG
#include "prng_rf.h"
#endif
XDATA __at(0x1C00) uint8_t img[N_IN];
XDATA __at(0x1B00) uint8_t res[64];   /* [0]=class, [1..40]=logits LE, [41..44]=cycles LE,
                                         [45..48]=empty-timing cycles, [49]=0xA5 done,
                                         [50..53]=rf_gen-only cycles (PRNG models) */
void sim_done(void) { while (1); }
void sim_mark(void) { }   /* breakpoint: end of the inference phase (instruction count) */
static void put32(uint8_t i, uint32_t v)
{
    res[i] = v; res[i + 1] = v >> 8; res[i + 2] = v >> 16; res[i + 3] = v >> 24;
}
void main(void)
{
    uint8_t c, i;
    uint32_t cy, cy0;
    cyc_init();
    cyc_start(); cy0 = cyc_stop();          /* timing overhead */
    cyc_start();
    c = infer(img);
    cy = cyc_stop();
    res[0] = c;
    for (i = 0; i < N_OUT; i++) put32(1 + 4 * i, (uint32_t)logits[i]);
    put32(41, cy);
    put32(45, cy0);
    sim_mark();
#ifdef MODEL_DANN_PRNG
    {   /* step 5: cycles of the index regeneration alone (all H1 rf_gen calls, same seeds) */
        DATA uint16_t n, s = RF_C;
        cyc_start();
        for (n = 0; n < H1; n++) { s += RF_K; rf_gen(s); }
        cy = cyc_stop();
        put32(50, cy);   /* [50..53] = rf_gen-only cycles incl. timing overhead (subtract [45..48]) */
    }
#endif
    res[49] = 0xA5;
    sim_done();
}
