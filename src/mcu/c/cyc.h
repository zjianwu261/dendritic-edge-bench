/* Cycle counter on Timer0 (mode 1, 16-bit, overflow counted in ISR).
 * STC8 (T0x12 = 1): counts SYSclk cycles. Classic 8051 / ucsim: counts machine cycles (12 clk). */
#ifndef T1M_CYC_H
#define T1M_CYC_H
#include <stdint.h>
extern volatile uint16_t t0_ovf;
void t0_isr(void) __interrupt(1);
void cyc_init(void);
void cyc_start(void);
uint32_t cyc_stop(void);
#endif
