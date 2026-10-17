#include "stc8.h"
#include "cyc.h"
volatile uint16_t t0_ovf;
void t0_isr(void) __interrupt(1) { t0_ovf++; }
void cyc_init(void)
{
#ifdef STC8
    AUXR |= 0x80;            /* T0x12: timer0 clocked by SYSclk (1T) */
#endif
    TMOD = (TMOD & 0xF0) | 0x01; /* timer0 mode 1: 16-bit, no auto-reload */
    ET0 = 1;
    EA = 1;
}
void cyc_start(void)
{
    TR0 = 0; TF0 = 0; TH0 = 0; TL0 = 0; t0_ovf = 0;
    TR0 = 1;
}
uint32_t cyc_stop(void)
{
    uint32_t c;
    TR0 = 0;
    if (TF0) { TF0 = 0; t0_ovf++; }   /* overflow pending but ISR not yet taken */
    c = ((uint32_t)t0_ovf << 16) | ((uint16_t)TH0 << 8) | TL0;
    return c;
}
