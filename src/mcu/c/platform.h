/* T1m: memory-space qualifiers. SDCC/8051: weights in __code (Flash), buffers in __xdata. */
#ifndef T1M_PLATFORM_H
#define T1M_PLATFORM_H
#include <stdint.h>
#ifdef __SDCC
#define CODE __code
#define XDATA __xdata
#define DATA __data   /* hot locals in internal RAM (--model-large defaults to xdata) */
#define IDATA __idata
#else
#define CODE
#define XDATA
#define DATA
#define IDATA
#endif
#endif
