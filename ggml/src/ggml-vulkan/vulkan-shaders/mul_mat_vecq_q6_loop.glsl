// q6_K fast n>=5 MMVQ row loop, instantiated once per grouped weight binding (Q6_FN / Q6_DA / Q6_DAP)
void Q6_FN(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const uint first_row, const uint num_rows, const uint a_off, const uint b_off, const uint tid) {
    const uint nbpr = p.ncols / QUANT_K;
    const uint it_size = BLOCK_SIZE / 16;
    const uint itid = tid % 16;
    const uint ix = tid / 16;
    const uint v_im = itid / 8;
    const uint v_in = itid % 8;
    const uint l0 = 4 * v_in;
    const uint is = v_in / 4;
    const uint ql_o = (64 * v_im + l0) / 2;
    const uint qh_o = (32 * v_im + l0) / 2;
    const uint sc_o = 8 * v_im + is;

    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = FLOAT_TYPE(0.0f);
        }
    }

    for (uint sb0 = 0; sb0 < nbpr; sb0 += it_size * Q6_UNR) {
        [[unroll]] for (uint u = 0; u < 4; ++u) {
            if (u < Q6_UNR) {
                const uint sb = sb0 + u * it_size + ix;
                const float valid = sb < nbpr ? 1.0 : 0.0;
                const uint sbc = min(sb, nbpr - 1);

                int32_t bq[NUM_COLS][4];
                float bd[NUM_COLS][4];
                [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                    const uint outer = (j * p.batch_stride_b / QUANT_K_Q8_1 + b_off + sbc * 8 + v_im * 4) / 4;
                    [[unroll]] for (uint qq = 0; qq < 4; ++qq) {
                        bq[j][qq] = data_b[outer].qs[v_in + 8 * qq];
                        bd[j][qq] = float(data_b[outer].ds[qq].x);
                    }
                }

                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    const uint ib = a_off + (first_row + n) * nbpr + sbc;
                    const uint32_t ql0 = uint32_t(Q6_DAP[ib].ql[ql_o]) | (uint32_t(Q6_DAP[ib].ql[ql_o + 1]) << 16);
                    const uint32_t ql1 = uint32_t(Q6_DAP[ib].ql[ql_o + 16]) | (uint32_t(Q6_DAP[ib].ql[ql_o + 17]) << 16);
                    const uint32_t qh  = uint32_t(Q6_DAP[ib].qh[qh_o]) | (uint32_t(Q6_DAP[ib].qh[qh_o + 1]) << 16);
                    const float d = float(Q6_DA[ib].d) * valid;
                    float sc[4];
                    [[unroll]] for (uint qq = 0; qq < 4; ++qq) {
                        sc[qq] = float(Q6_DA[ib].scales[sc_o + 2 * qq]) * d;
                    }
                    const uint32_t v0 = (ql0 & 0x0F0F0F0F)        | ((qh & 0x03030303) << 4);
                    const uint32_t v1 = (ql1 & 0x0F0F0F0F)        | ((qh & 0x0C0C0C0C) << 2);
                    const uint32_t v2 = ((ql0 >> 4) & 0x0F0F0F0F) | (qh & 0x30303030);
                    const uint32_t v3 = ((ql1 >> 4) & 0x0F0F0F0F) | ((qh & 0xC0C0C0C0) >> 2);
                    const int32_t q0 = int32_t(((v0 ^ 0x80808080) - 0x20202020) ^ 0x80808080);
                    const int32_t q1 = int32_t(((v1 ^ 0x80808080) - 0x20202020) ^ 0x80808080);
                    const int32_t q2 = int32_t(((v2 ^ 0x80808080) - 0x20202020) ^ 0x80808080);
                    const int32_t q3 = int32_t(((v3 ^ 0x80808080) - 0x20202020) ^ 0x80808080);
                    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                        float acc = bd[j][0] * sc[0] * float(dotPacked4x8EXT(q0, bq[j][0]));
                        acc = fma(bd[j][1] * sc[1], float(dotPacked4x8EXT(q1, bq[j][1])), acc);
                        acc = fma(bd[j][2] * sc[2], float(dotPacked4x8EXT(q2, bq[j][2])), acc);
                        acc = fma(bd[j][3] * sc[3], float(dotPacked4x8EXT(q3, bq[j][3])), acc);
                        temp[j][n] += FLOAT_TYPE(acc);
                    }
                }
            }
        }
    }

}
