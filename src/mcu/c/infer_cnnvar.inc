/* cnnvar body of infer() (included inside infer.c, #elif defined(MODEL_CNNVAR)); see infer_cnnvar_defs.inc. */
    DATA uint8_t o, i, y, xx, v;
    DATA int32_t acc;
    const CODE int8_t * DATA pw;
    XDATA uint8_t * DATA pa;
    const XDATA uint8_t * DATA px;
    const XDATA uint8_t * DATA src;
    DATA int16_t p;
#if defined(MODEL_CNNDS)
    DATA uint16_t sum;
    DATA uint32_t g;
#endif
    (void)j;
#if defined(MODEL_CNNPOOL)
    /* conv1 (s2) per channel -> sb, then 2x2 max-pool 14x14 -> 7x7 into a1 */
    for (o = 0; o < C1; o++) {
        CONV1_S2(o, sb, w1, b1, SH1);
        pa = a1 + (uint16_t)o * 49;
        for (y = 0; y < 7; y++) {
            px = sb + (uint16_t)y * 28;
            for (xx = 0; xx < 7; xx++, px += 2) {
                v = px[0];
                if (px[1] > v) v = px[1];
                if (px[14] > v) v = px[14];
                if (px[15] > v) v = px[15];
                *pa++ = v;
            }
        }
    }
    /* conv2 3x3 s1 p1 (only the 6x6 outputs the floor pool uses) -> sb2, then 2x2 max-pool -> 3x3 into a2 */
    for (o = 0; o < C2; o++) {
        pa = sb2;
        for (y = 0; y < 6; y++) {
            for (xx = 0; xx < 6; xx++) {
                acc = b2[o];
                pw = w2 + (uint16_t)o * (C1 * 9);
                src = a1;
                C3S1(C1);
                *pa++ = relu8((acc + RND(SH2)) >> SH2);
            }
        }
        pa = a2 + (uint16_t)o * 9;
        for (y = 0; y < 3; y++) {
            px = sb2 + (uint16_t)y * 12;
            for (xx = 0; xx < 3; xx++, px += 2) {
                v = px[0];
                if (px[1] > v) v = px[1];
                if (px[6] > v) v = px[6];
                if (px[7] > v) v = px[7];
                *pa++ = v;
            }
        }
    }
#elif defined(MODEL_CNNS4)
    /* conv1 4x4 stride 4, no padding: 28x28 -> 7x7xC1 */
    pa = a1;
    for (o = 0; o < C1; o++) {
        for (y = 0; y < 7; y++) {
            for (xx = 0; xx < 7; xx++) {
                acc = b1[o];
                pw = w1 + (uint16_t)o * 16;
                px = x + (uint16_t)y * 112 + 4 * xx;
                T(0, 0); T(1, 1); T(2, 2); T(3, 3);
                T(28, 4); T(29, 5); T(30, 6); T(31, 7);
                T(56, 8); T(57, 9); T(58, 10); T(59, 11);
                T(84, 12); T(85, 13); T(86, 14); T(87, 15);
                *pa++ = relu8((acc + RND(SH1)) >> SH1);
            }
        }
    }
    /* conv2 3x3 s1 p1: 7x7xC1 -> 7x7xC2 */
    pa = a2;
    for (o = 0; o < C2; o++) {
        for (y = 0; y < 7; y++) {
            for (xx = 0; xx < 7; xx++) {
                acc = b2[o];
                pw = w2 + (uint16_t)o * (C1 * 9);
                src = a1;
                C3S1(C1);
                *pa++ = relu8((acc + RND(SH2)) >> SH2);
            }
        }
    }
#elif defined(MODEL_CNNDS)
    /* conv1 (s2) per channel -> sb, fused with depthwise 3x3 s2 p1 of the same channel: 14x14 -> 7x7 into ba */
    for (o = 0; o < CH; o++) {
        CONV1_S2(o, sb, w1, b1, SH1);
        pa = ba + (uint16_t)o * 49;
        for (y = 0; y < 7; y++) {
            for (xx = 0; xx < 7; xx++) {
                acc = b2[o];
                pw = w2 + (uint16_t)o * 9;
                px = sb + (y ? (uint16_t)(2 * y - 1) * 14 : 0) + (xx ? 2 * xx - 1 : 0);
                if (y && xx) TAPS_IN(14);
                else if (xx) TAPS_TOP(14);
                else if (y) TAPS_LEFT(14);
                else TAPS_CORNER(14);
                *pa++ = relu8((acc + RND(SH2)) >> SH2);
            }
        }
    }
    /* pointwise 1x1 CH -> CH: ba -> bb (input channel stride 49) */
    pa = bb;
    for (o = 0; o < CH; o++) {
        for (y = 0; y < 49; y++) {
            acc = b3[o];
            pw = w3 + (uint16_t)o * CH;
            px = ba + y;
            for (i = 0; i < CH; i++, px += 49) {
                p = *px * *pw++;
                acc += p;
            }
            *pa++ = relu8((acc + RND(SH3)) >> SH3);
        }
    }
    /* depthwise 3x3 s1 p1: bb -> ba */
    pa = ba;
    for (o = 0; o < CH; o++) {
        src = bb + (uint16_t)o * 49;
        for (y = 0; y < 7; y++) {
            for (xx = 0; xx < 7; xx++) {
                acc = b4[o];
                pw = w4 + (uint16_t)o * 9;
                C3S1(1);
                *pa++ = relu8((acc + RND(SH4)) >> SH4);
            }
        }
    }
    /* pointwise 1x1 CH -> CH fused with global average pool (sum x GAP_M >> GAP_S, clip 127): ba -> a2 */
    for (o = 0; o < CH; o++) {
        sum = 0;
        for (y = 0; y < 49; y++) {
            acc = b5[o];
            pw = w5 + (uint16_t)o * CH;
            px = ba + y;
            for (i = 0; i < CH; i++, px += 49) {
                p = *px * *pw++;
                acc += p;
            }
            sum += relu8((acc + RND(SH5)) >> SH5);
        }
        g = ((uint32_t)sum * GAP_M + (1UL << (GAP_S - 1))) >> GAP_S;
        a2[o] = g > 127 ? 127 : (uint8_t)g;
    }
#endif
    best = 0;
    for (c = 0; c < N_OUT; c++) {
        logits[c] = dot_u8(wf + (uint16_t)c * NF, a2, NF, bf[c]);
        if (logits[c] > logits[best])
            best = c;
    }
    return best;
