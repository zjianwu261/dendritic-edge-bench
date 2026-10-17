/* T1m step 5: pseudo-random local receptive fields (see src/mcu/prng_rf.py for the spec).
 * Written byte-wise for the 8051: the 16-bit xorshift (7, 9, 8) is split into its hi/lo bytes
 *   x ^= x << 7 :  hi ^= (hi << 7) | (lo >> 1);  lo ^= lo << 7   (both from the old bytes)
 *   x ^= x >> 9 :  lo ^= hi >> 1
 *   x ^= x << 8 :  hi ^= lo
 * and all products are 8x8 -> 16 (one MUL AB). */
#include "prng_rf.h"

DATA uint8_t rf_off[RF_SYN] = {29, 30, 31, 57, 58, 59, 85, 86, 87, 0, 0, 0, 0, 0, 0, 0};
/* radius-2 ring of the 5x5 window (window offsets) and "used" stamps, both in idata (@Ri) */
static IDATA uint8_t rf_ring[16] = {0, 1, 2, 3, 4, 28, 32, 56, 60, 84, 88, 112, 113, 114, 115, 116};
static IDATA uint8_t rf_used[16];   /* slot used in this call <=> rf_used[slot] == stamp */
static DATA uint8_t xh, xl, k, n_;
/* 255 -> the first call wraps to 0 and clears rf_used[] explicitly (idata is not guaranteed to be
 * zero after reset on the board; initialised data is, via the C startup code) */
static DATA uint8_t stamp = 255;

#define XS16() do { \
        DATA uint8_t t_ = (uint8_t)(xh << 7) | (xl >> 1); \
        xl ^= (uint8_t)(xl << 7); \
        xh ^= t_; \
        xl ^= xh >> 1; \
        xh ^= xl; \
    } while (0)

/* try ring slot nib (0..15): append it unless already drawn for this dendrite */
#define TAKE(nib) do { \
        n_ = (nib); \
        if (rf_used[n_] != stamp) { rf_used[n_] = stamp; rf_off[k++] = rf_ring[n_]; } \
    } while (0)

uint16_t rf_gen(uint16_t s)
{
    DATA uint8_t row, col;
    DATA uint16_t p;
    if (!s) s = 1;
    xh = s >> 8;
    xl = (uint8_t)s;
    XS16();
    p = (uint8_t)xh * (uint8_t)xh;          /* Feistel round: lo += hi(hi*hi) */
    xl += (uint8_t)(p >> 8);
    XS16();
    XS16();
    p = (uint8_t)xh * (uint8_t)28;
    row = p >> 8;
    p = (uint8_t)xl * (uint8_t)28;
    col = p >> 8;
    if (row < 2) row = 2; else if (row > 25) row = 25;
    if (col < 2) col = 2; else if (col > 25) col = 25;
    k = 9;
    if (++stamp == 0) {            /* stamps are unique within 255 calls: clear on wrap */
        for (n_ = 0; n_ < 16; n_++) rf_used[n_] = 0;
        stamp = 1;
    }
    for (;;) {
        XS16();
        TAKE(xl & 15);
        if (k == RF_SYN) break;
        TAKE(xl >> 4);
        if (k == RF_SYN) break;
        TAKE(xh & 15);
        if (k == RF_SYN) break;
        TAKE(xh >> 4);
        if (k == RF_SYN) break;
    }
    p = (uint8_t)(row - 2) * (uint8_t)28;
    return p + (uint8_t)(col - 2);
}
