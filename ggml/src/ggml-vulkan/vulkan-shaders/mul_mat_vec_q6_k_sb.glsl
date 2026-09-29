void SB_NAME(const uint a_offset, const uint b_offset, const uint itid, const uint ix, const uint ql_offset, const uint qh_offset, const uint s_offset, const uint y_offset, const uint i, const uint num_blocks_per_row, const uint first_row, const uint num_rows, const bool all_threads) {
    const uint y_idx = i * QUANT_K + y_offset;

    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        const uint ib0 = a_offset + (first_row+n)*num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) { // when we don't have enough blocks to use all threads
            if (i < num_blocks_per_row)
                sccache[csel][ix][itid] = FLOAT_TYPE(SB_A[ib0 + i].scales[itid]);
            barrier();

            if (i >= num_blocks_per_row)
                continue;
        }

        const uint32_t ql0_u32 =  uint32_t(SB_A16[ib0 + i].ql[ql_offset / 2]) | (uint32_t(SB_A16[ib0 + i].ql[ql_offset / 2 + 1]) << 16);
        const uint32_t ql32_u32 = uint32_t(SB_A16[ib0 + i].ql[ql_offset / 2 + 16]) | (uint32_t(SB_A16[ib0 + i].ql[ql_offset / 2 + 17]) << 16);

        const uint32_t ql0_u32_lo4 = ql0_u32 & 0x0F0F0F0F;
        const uint32_t ql0_u32_hi4 = (ql0_u32 >> 4) & 0x0F0F0F0F;
        const uint32_t ql32_u32_lo4 = ql32_u32 & 0x0F0F0F0F;
        const uint32_t ql32_u32_hi4 = (ql32_u32 >> 4) & 0x0F0F0F0F;

        const uint32_t qh_u32 = uint32_t(SB_A16[ib0 + i].qh[qh_offset / 2]) | (uint32_t(SB_A16[ib0 + i].qh[qh_offset / 2 + 1]) << 16);
        const uint32_t qh0_u32 = (qh_u32 & 0x03030303) << 4;
        const uint32_t qh2_u32 = (qh_u32 & 0x0C0C0C0C) << 2;
        const uint32_t qh4_u32 = (qh_u32 & 0x30303030);
        const uint32_t qh6_u32 = (qh_u32 & 0xC0C0C0C0) >> 2;

        const uint32_t q0_u32 = ql0_u32_lo4  | qh0_u32;
        const uint32_t q1_u32 = ql32_u32_lo4 | qh2_u32;
        const uint32_t q2_u32 = ql0_u32_hi4  | qh4_u32;
        const uint32_t q3_u32 = ql32_u32_hi4 | qh6_u32;

        const vec4 q0 = vec4(unpack8(q0_u32)) - 32;
        const vec4 q1 = vec4(unpack8(q1_u32)) - 32;
        const vec4 q2 = vec4(unpack8(q2_u32)) - 32;
        const vec4 q3 = vec4(unpack8(q3_u32)) - 32;

        if (all_threads) {
            sccache[csel][ix][itid] = FLOAT_TYPE(SB_A[ib0 + i].scales[itid]);
            barrier();
        }

        const FLOAT_TYPE d = FLOAT_TYPE(SB_A[ib0 + i].d);

        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            vec4 by0  = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4     ]);
            vec4 by32 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 +  8]);
            vec4 by64 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 16]);
            vec4 by96 = vec4(data_b_v4[(j*p.batch_stride_b + b_offset + y_idx) / 4 + 24]);

            FLOAT_TYPE sum[4] = {0, 0, 0, 0};
            [[unroll]] for (uint l = 0; l < 4; ++l) {
                sum[0] = fma(FLOAT_TYPE(by0[l]), q0[l], sum[0]);
                sum[1] = fma(FLOAT_TYPE(by32[l]), q1[l], sum[1]);
                sum[2] = fma(FLOAT_TYPE(by64[l]), q2[l], sum[2]);
                sum[3] = fma(FLOAT_TYPE(by96[l]), q3[l], sum[3]);
            }
            temp[j][n] = fma(fma(sum[0], sccache[csel][ix][s_offset], fma(sum[1], sccache[csel][ix][s_offset + 2], fma(sum[2], sccache[csel][ix][s_offset + 4], sum[3] * sccache[csel][ix][s_offset + 6]))), d, temp[j][n]);
        }
    }
}

