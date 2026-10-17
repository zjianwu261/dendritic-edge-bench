/* STC8A8K64D4 board driver (flashed by src/mcu/step4_flash_and_measure.sh).
 * UART1 on P3.0/P3.1 (CH340), 8N1, BAUD from F_OSC (compile-time, must match stcgal -t).
 * Protocol (host: src/mcu/host_uart.py):
 *   host 'I'                -> "T1M <model> F_OSC=<hz>\n"
 *   host 0xA5 + 784 bytes   -> 0x5A, class, 10 x int32 LE logits, uint32 LE cycles,
 *                              uint32 LE empty-timing cycles   (50 bytes)
 * Cycles = SYSclk cycles of infer() (Timer0 in 1T mode). */
#include "stc8.h"
#include "infer.h"
#include "cyc.h"
#ifndef F_OSC
#define F_OSC 35000000UL
#endif
#ifndef BAUD
#define BAUD 115200UL
#endif
#define T2_RELOAD (65536UL - (F_OSC / 4UL + BAUD / 2UL) / BAUD)
#define STR2(x) #x
#define STR(x) STR2(x)

static XDATA uint8_t img[N_IN];

static void uart_init(void)
{
    P3M0 &= ~0x03; P3M1 &= ~0x03;  /* P3.0/P3.1 quasi-bidirectional */
    P_SW1 &= ~0xC0;                /* UART1 on P3.0/P3.1 */
    SCON = 0x50;                   /* mode 1, receive enable */
    T2L = T2_RELOAD & 0xFF;
    T2H = T2_RELOAD >> 8;
    AUXR |= 0x15;                  /* T2R | T2x12 | S1ST2: timer2 1T as UART1 baud source */
}
static uint8_t u_getc(void) { while (!RI); RI = 0; return SBUF; }
static void u_putc(uint8_t c) { SBUF = c; while (!TI); TI = 0; }
static void puts_(const CODE char *s) { while (*s) u_putc(*s++); }
static void put32(uint32_t v) { u_putc(v); u_putc(v >> 8); u_putc(v >> 16); u_putc(v >> 24); }

void main(void)
{
    uint16_t i;
    uint8_t c;
    uint32_t cy, cy0;
    uart_init();
    cyc_init();
    for (;;) {
        c = u_getc();
        if (c == 'I') {
            puts_("T1M "); puts_(model_name); puts_(" F_OSC=" STR(F_OSC) "\n");
        } else if (c == 0xA5) {
            for (i = 0; i < N_IN; i++) img[i] = u_getc();
            cyc_start(); cy0 = cyc_stop();
            cyc_start();
            c = infer(img);
            cy = cyc_stop();
            u_putc(0x5A);
            u_putc(c);
            for (c = 0; c < N_OUT; c++) put32((uint32_t)logits[c]);
            put32(cy);
            put32(cy0);
        }
    }
}
