/* cnnvar (E-10 CNN-variant extension, PROTOCOL_cnnvar.md): buffers and 3x3 stride-1 pad-1 tap macros.
 * Included by infer.c after the MODEL_CNN TAPS_* macros (T(o, k) = one MAC with the DATA locals p, px, pw, acc).
 * Integer semantics: src/mcu/cnnvar.py ref_forward_var (bit-exact). All activations CHW uint8 0..127. */
#if defined(MODEL_CNNPOOL)
static XDATA uint8_t sb[196];          /* conv1 output of one channel, 14x14 */
static XDATA uint8_t a1[C1 * 49];      /* conv1 + maxpool, 7x7 */
static XDATA uint8_t sb2[36];          /* conv2 output of one channel, rows/cols 0..5 (floor pool drops row/col 6) */
static XDATA uint8_t a2[C2 * 9];       /* conv2 + maxpool, 3x3 */
#define NF (C2 * 9)
#elif defined(MODEL_CNNS4)
static XDATA uint8_t a1[C1 * 49];      /* conv4x4 s4 output, 7x7 */
static XDATA uint8_t a2[C2 * 49];      /* conv3x3 s1 output, 7x7 */
#define NF (C2 * 49)
#elif defined(MODEL_CNNDS)
static XDATA uint8_t sb[196];          /* conv1 output of one channel, 14x14 (fused with dw1 per channel) */
static XDATA uint8_t ba[CH * 49];      /* ping */
static XDATA uint8_t bb[CH * 49];      /* pong */
static XDATA uint8_t a2[CH];           /* GAP output */
#define NF CH
#endif

/* 3x3 stride 1 pad 1 on a 7x7 CHW input: px points at (max(y-1,0), max(x-1,0)); padding taps are skipped.
 * RW_*: one kernel row (F full, L left column padded, R right column padded); K_*: rows (F all, T top padded, B bottom). */
#define RW_F(o, k) T(o, k); T((o) + 1, (k) + 1); T((o) + 2, (k) + 2)
#define RW_L(o, k) T(o, (k) + 1); T((o) + 1, (k) + 2)
#define RW_R(o, k) T(o, k); T((o) + 1, (k) + 1)
#define K_F(R) R(0, 0); R(7, 3); R(14, 6)
#define K_T(R) R(0, 3); R(7, 6)
#define K_B(R) R(0, 0); R(7, 3)
#define LOOP_S1(NCH, BODY) for (i = 0; i < (NCH); i++, pw += 9, px += 49) { BODY; }
#define C3S1_ROW(NCH, K) \
    if (xx == 0) LOOP_S1(NCH, K(RW_L)) else if (xx == 6) LOOP_S1(NCH, K(RW_R)) else LOOP_S1(NCH, K(RW_F))
/* acc += 3x3 s1 p1 window at output (y, xx) over NCH channels of src (stride 49), weights pw (9 per channel) */
#define C3S1(NCH) do { \
        px = src + (y ? (uint16_t)(y - 1) * 7 : 0) + (xx ? xx - 1 : 0); \
        if (y == 0) { C3S1_ROW(NCH, K_T) } else if (y == 6) { C3S1_ROW(NCH, K_B) } else { C3S1_ROW(NCH, K_F) } \
    } while (0)

/* conv3x3 s2 p1, 28x28x1 -> 14x14, output channel o -> dst (196 B), same tap code as MODEL_CNN conv1 */
#define CONV1_S2(o, dst, W, B, SH) do { \
        pa = dst; \
        for (y = 0; y < 14; y++) { \
            for (xx = 0; xx < 14; xx++) { \
                acc = B[o]; \
                pw = W + (uint16_t)(o) * 9; \
                if (y) { \
                    px = x + (uint16_t)(2 * y - 1) * 28; \
                    if (xx) { px += 2 * xx - 1; TAPS_IN(28); } else { TAPS_LEFT(28); } \
                } else { \
                    px = x; \
                    if (xx) { px += 2 * xx - 1; TAPS_TOP(28); } else { TAPS_CORNER(28); } \
                } \
                *pa++ = relu8((acc + RND(SH)) >> SH); \
            } \
        } \
    } while (0)
