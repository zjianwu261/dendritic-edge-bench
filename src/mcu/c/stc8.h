/* Minimal STC8A8K64D4 SFR subset (on top of the standard 8052 SFRs from <8052.h>). */
#ifndef T1M_STC8_H
#define T1M_STC8_H
#include <8052.h>
__sfr __at(0x8E) AUXR;   /* b7 T0x12, b4 T2R, b2 T2x12, b1 EXTRAM, b0 S1ST2 */
__sfr __at(0xA2) P_SW1;  /* UART1 pin select (0 = P3.0/P3.1) */
__sfr __at(0xB1) P3M1;
__sfr __at(0xB2) P3M0;
__sfr __at(0xD6) T2H;
__sfr __at(0xD7) T2L;
#endif
