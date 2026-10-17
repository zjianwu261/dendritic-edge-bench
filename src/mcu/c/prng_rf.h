/* T1m step 5: on-chip regeneration of pseudo-random local receptive fields.
 * Bit-exact twin of src/mcu/prng_rf.py (see there for the definition). */
#ifndef T1M_PRNG_RF_H
#define T1M_PRNG_RF_H
#include "platform.h"
#define RF_K 0x9E37u
#define RF_SYN 16
/* rf_off[0..8] = fixed 3x3 core, rf_off[9..15] rewritten by rf_gen() */
extern DATA uint8_t rf_off[RF_SYN];
/* s = RF_K*(n+1)+RF_C (mod 2^16). Fills rf_off[9..15], returns base pixel of the 5x5 window. */
uint16_t rf_gen(uint16_t s);
#endif
