#ifndef T1M_INFER_H
#define T1M_INFER_H
#include "platform.h"
#define N_IN 784
#define N_OUT 10
extern XDATA int32_t logits[N_OUT];
extern const CODE char model_name[];
/* x: 784 raw uint8 pixels in xdata. Fills logits[], returns first-argmax class. */
uint8_t infer(const XDATA uint8_t *x);
#endif
