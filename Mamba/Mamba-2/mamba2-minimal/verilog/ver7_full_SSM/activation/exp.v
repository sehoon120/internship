// ================================================================
// exp16_base2_pwl8: FP16 e^x with base-2 range reduction + 8-seg PWL
//   e^x = 2^(x*log2(e)) = 2^(k+f) = 2^k * 2^f
//   where k = floor(t), f = t - k ∈ [0,1)
//   2^f ≈ y0(seg) + slope(seg) * (f - f0(seg)), seg = floor(f*8)
// Notes:
//  - Replace fp16_add / fp16_mul with your Vivado FP16 IP wrappers.
//  - Fill y0/slope ROM constants (FP16 hex). (아래 Python 스니펫으로 자동 생성 권장)
//  - II=1 유지: 각 단계 valid를 정렬(pipe_bus)해 주세요.
// ================================================================
`timescale 1ns/1ps

module exp16_base2_pwl8 #(
    parameter integer DW       = 16,
    parameter integer LAT_MUL  = 1,   // your FP16 mul latency
    parameter integer LAT_ADD  = 1,  // your FP16 add latency
    parameter integer K_MIN    = -16, // supported integer power range
    parameter integer K_MAX    =  16
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] x_i,        // FP16
    output wire [DW-1:0] y_o,        // FP16 ≈ exp(x)
    output wire          valid_o
);
    // ---------------- constants ----------------
    localparam [DW-1:0] H_LOG2E  = 16'h3DC5; // ≈ 1.4427  (FP16 hex)
    localparam [DW-1:0] H_ZERO   = 16'h0000;
    localparam [DW-1:0] H_ONE    = 16'h3C00;
    localparam [DW-1:0] H_EIGHT  = 16'h4100; // 8.0

    // 1) t = x * log2(e)
    wire [DW-1:0] t_w; wire v_t;
    fp16_mul u_mul_log2e (
        .clk(clk), .rstn(rstn),
        .valid_i(valid_i),
        .a_i(x_i), .b_i(H_LOG2E),
        .p_o(t_w), .valid_o(v_t)
    );

    // 2) k = floor(t), f = t - k (0<=f<1). Implement with threshold picker (finite range)
    wire signed [7:0] k_s;   // integer
    wire [DW-1:0]     k_as_fp16; // FP16 representation of integer k
    wire [DW-1:0]     f_w;   // t - k
    wire              v_tf;
    base2_k_picker #(.DW(DW), .K_MIN(K_MIN), .K_MAX(K_MAX)) u_kpick (
        .clk(clk), .rstn(rstn),
        .valid_i(v_t),
        .t_i(t_w),
        .k_o(k_s), .k_fp16_o(k_as_fp16),
        .f_o(f_w), .valid_o(v_tf)
    );

    // 3) segment index and left boundary f0 = seg/8
    wire [2:0] seg_idx; wire v_seg;
    seg8_indexer #(.DW(DW)) u_seg (
        .clk(clk), .rstn(rstn),
        .valid_i(v_tf), .f_i(f_w),
        .seg_o(seg_idx), .valid_o(v_seg)
    );

    reg [DW-1:0]       f_w1, f_w2;
//    reg                v_seg1, v_seg2;
    
    always @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            f_w1 <= {DW{1'b0}};
            f_w2 <= {DW{1'b0}};
        end else begin
            f_w1 <= f_w;
            f_w2 <= f_w1;
        end
    end

    // 4) ROM: y0(seg)=2^(seg/8), slope(seg) = 8*(y1 - y0)
    wire [DW-1:0] y0_seg, slope_seg; wire v_rom;
    pwl8_rom #(.DW(DW)) u_rom (
        .clk(clk), .rstn(rstn),
        .valid_i(v_seg), .seg_i(seg_idx),
        .y0_o(y0_seg), .slope_o(slope_seg), .valid_o(v_rom)
    );
    
    reg [DW-1:0]       slope_seg1, slope_seg2, slope_seg3;
    reg [DW-1:0]       y0_seg1, y0_seg2, y0_seg3, y0_seg4;
    always @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            slope_seg1 <= {DW{1'b0}};
            slope_seg2 <= {DW{1'b0}};
            slope_seg3 <= {DW{1'b0}};
            y0_seg1 <= {DW{1'b0}};
            y0_seg2 <= {DW{1'b0}};
            y0_seg3 <= {DW{1'b0}};
            y0_seg4 <= {DW{1'b0}};
        end else begin
            slope_seg1 <= slope_seg;
            slope_seg2 <= slope_seg1;
            slope_seg3 <= slope_seg2;
            y0_seg1 <= y0_seg;
            y0_seg2 <= y0_seg1;
            y0_seg3 <= y0_seg2;
            y0_seg4 <= y0_seg3;
        end
    end

    // 5) df = f - f0(seg), with f0 = seg/8
    wire [DW-1:0] f0_seg; wire v_f0;
    seg8_f0_rom #(.DW(DW)) u_f0 (
        .clk(clk), .rstn(rstn),
        .valid_i(v_seg), .seg_i(seg_idx),
        .f0_o(f0_seg), .valid_o(v_f0)
    );

    // 5) df = f - f0  (align 단계 추가: adder에 넣기 전에 한 번 더 래치)
    reg [DW-1:0] f_w2_s, f0_s;
    reg          v_s;
    
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            f_w2_s <= {DW{1'b0}};
            f0_s   <= {DW{1'b0}};
            v_s    <= 1'b0;
        end else begin
            v_s    <= v_f0;          // f0_seg가 유효해질 때만 동시 캡처
            if (v_f0) begin
                f_w2_s <= f_w2;      // 이미 2단 지연된 f
                f0_s   <= f0_seg;    // ROM 출력
            end
        end
    end

    // 6) two_pow_f ≈ y0 + slope*(f - f0)
    wire [DW-1:0] df_w; wire v_df;
    fp16_add u_sub_f0 (
        .clk(clk), .rstn(rstn),
        .valid_i(v_s),
        .a_i(f_w2_s), .b_i(f0_s),
        .sum_o(df_w), .valid_o(v_df)
    );
    
    // 출력도 레지스터에 잡기(글리치가 밖으로 안 나가게)
    reg [DW-1:0] df_w_r;
    reg          v_df_r;
    
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            df_w_r <= {DW{1'b0}};
            v_df_r <= 1'b0;
        end else begin
            v_df_r <= v_df;
            if (v_df) begin
                df_w_r <= df_w;
            end
            // v_df_raw==0이면 유지 → 파형 흔들림 없음
        end
    end

    wire [DW-1:0] slope_df; wire v_sdf;
    fp16_mul u_mul_slope (
        .clk(clk), .rstn(rstn),
        .valid_i(v_df_r),
        .a_i(slope_seg3), .b_i(df_w_r),
        .p_o(slope_df), .valid_o(v_sdf)
    );

    wire [DW-1:0] two_pow_f; wire v_2f;
    fp16_add u_add_y0 (
        .clk(clk), .rstn(rstn),
        .valid_i(v_sdf),
        .a_i(y0_seg4), .b_i(slope_df),
        .sum_o(two_pow_f), .valid_o(v_2f)
    );

    // 7) y = 2^k * 2^f   (2^k from ROM)
    wire [DW-1:0] two_pow_k; wire v_2k;
    pow2k_rom #(.DW(DW), .K_MIN(K_MIN), .K_MAX(K_MAX)) u_pow2k (
        .clk(clk), .rstn(rstn),
        .valid_i(v_tf), .k_i(k_s),
        .two_pow_k_o(two_pow_k), .valid_o(v_2k)
    );

//    // align two_pow_f and two_pow_k
//    wire [DW-1:0] two_pow_f_a, two_pow_k_a; wire v_mul;
//    align2 #(.DW(DW)) u_align_last (
//        .clk(clk), .rstn(rstn),
//        .a_i(two_pow_f), .va_i(v_2f),
//        .b_i(two_pow_k), .vb_i(v_2k),
//        .a_o(two_pow_f_a), .b_o(two_pow_k_a),
//        .v_o(v_mul)
//    );

    reg [DW-1:0]       two_pow_k1, two_pow_k2, two_pow_k3, two_pow_k4, two_pow_k5, two_pow_k6;
    always @(posedge clk or negedge rstn) begin
        if(!rstn) begin
            two_pow_k1 <= {DW{1'b0}};
            two_pow_k2 <= {DW{1'b0}};
            two_pow_k3 <= {DW{1'b0}};
            two_pow_k4 <= {DW{1'b0}};
            two_pow_k5 <= {DW{1'b0}};
            two_pow_k6 <= {DW{1'b0}};
        end else begin
            two_pow_k1 <= two_pow_k;
            two_pow_k2 <= two_pow_k1;
            two_pow_k3 <= two_pow_k2;
            two_pow_k4 <= two_pow_k3;
            two_pow_k5 <= two_pow_k4;
            two_pow_k6 <= two_pow_k5;
        end
    end

    fp16_mul u_mul_final (
        .clk(clk), .rstn(rstn),
        .valid_i(v_2f),
        .a_i(two_pow_f), .b_i(two_pow_k6),
        .p_o(y_o), .valid_o(valid_o)
    );
endmodule
