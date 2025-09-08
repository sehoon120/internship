module exp16_fast #(
    parameter integer DW = 16,
    parameter integer K_MIN = -16,
    parameter integer K_MAX =  16,
    // IP latencies (네 FP16 * / + IP 지연으로 세팅)
    parameter integer LAT_MUL = 6,
    parameter integer LAT_ADD = 11
)(
    input  wire             clk,
    input  wire             rstn,
    input  wire             valid_i,
    input  wire [DW-1:0]    x_i,       // FP16
    output wire [DW-1:0]    exp_o,     // FP16
    output wire             valid_o
);
    // ====== 상수 ======
    // FP16 상수 (예시: IEEE-754 half)
    // ln2 ≈ 0.6931472 → 0x398C
    // 미리 ROM에 넣거나 파라미터로 전달
    localparam [DW-1:0] LN2  = 16'h398C;
    // A1=LN2, A2=(LN2^2)/2, A3=(LN2^3)/6 → 네가 FP16 곱으로 미리 구해도 되고, ROM으로 적재
    localparam [DW-1:0] A1   = 16'h398C; // ~0.6931
    localparam [DW-1:0] A2   = 16'h2E9B; // ~0.240226 (예시)
    localparam [DW-1:0] A3   = 16'h2637; // ~0.055504 (예시)

    // ====== k 구간 경계 ROM (bk = k*ln2),  k ∈ [K_MIN..K_MAX+1] ======
    // x ∈ [bk, b(k+1)) → 그 k를 선택
    // 간단히 33~34 엔트리 작은 ROM. valid_i 기준으로 1개씩 읽는 구조 or 전부 비교.
    // 여기선 개념만: 비교기+우선순위 인코더 블록으로 가정.
    wire signed [7:0] k_sel;    // 선택된 k
    wire [DW-1:0]     bk_val;   // k*ln2 (FP16)
    wire              v_k;

    rr_k_picker #(.DW(DW), .K_MIN(K_MIN), .K_MAX(K_MAX)) u_kpick (
        .clk(clk), .rstn(rstn),
        .valid_i(valid_i),
        .x_i(x_i),
        .k_o(k_sel), .bk_o(bk_val),
        .valid_o(v_k)
    );

    // ====== f = x - k*ln2 ======
    wire [DW-1:0] f_w; wire v_f;
    fp16_add u_sub_x_kln2 ( // x - bk
        .clk(clk), .rstn(rstn),
        .valid_i(v_k),
        .a_i(x_i), .b_i({~bk_val[DW-1], bk_val[DW-2:0]}), // negate bk
        .sum_o(f_w), .valid_o(v_f)
    );

    // ====== P(f) ≈ 1 + A1 f + A2 f^2 + A3 f^3 ======
    wire [DW-1:0] f2_w, f3_w, A1f, A2f2, A3f3, one_w, poly_w;
    wire v_f2, v_f3, v_A1f, v_A2f2, v_A3f3, v_poly;

    fp16_mul u_f2 ( .clk(clk), .rstn(rstn), .valid_i(v_f),  .a_i(f_w), .b_i(f_w), .p_o(f2_w), .valid_o(v_f2) );
    fp16_mul u_f3 ( .clk(clk), .rstn(rstn), .valid_i(v_f2), .a_i(f2_w), .b_i(f_w), .p_o(f3_w), .valid_o(v_f3) );

    fp16_mul u_A1f  ( .clk(clk), .rstn(rstn), .valid_i(v_f),  .a_i(A1), .b_i(f_w),  .p_o(A1f),  .valid_o(v_A1f) );
    fp16_mul u_A2f2 ( .clk(clk), .rstn(rstn), .valid_i(v_f2), .a_i(A2), .b_i(f2_w), .p_o(A2f2), .valid_o(v_A2f2) );
    fp16_mul u_A3f3 ( .clk(clk), .rstn(rstn), .valid_i(v_f3), .a_i(A3), .b_i(f3_w), .p_o(A3f3), .valid_o(v_A3f3) );

    // one + (A1f + A2f2) + A3f3
    fp16_const_one u_one (.one_o(one_w)); // 1.0 as FP16
    wire [DW-1:0] s12_w; wire v_s12;
    fp16_add u_add12 ( .clk(clk), .rstn(rstn), .valid_i(v_A2f2 & v_A1f), .a_i(A1f), .b_i(A2f2), .sum_o(s12_w), .valid_o(v_s12) );

    wire [DW-1:0] s123_w; wire v_s123;
    fp16_add u_add123 ( .clk(clk), .rstn(rstn), .valid_i(v_s12 & v_A3f3), .a_i(s12_w), .b_i(A3f3), .sum_o(s123_w), .valid_o(v_s123) );

    fp16_add u_add1 ( .clk(clk), .rstn(rstn), .valid_i(v_s123), .a_i(one_w), .b_i(s123_w), .sum_o(poly_w), .valid_o(v_poly) );

    // ====== 2^k LUT 스케일 ======
    wire [DW-1:0] pow2k_w; wire v_pow2k;
    pow2k_rom #(.DW(DW), .K_MIN(K_MIN), .K_MAX(K_MAX)) u_pow2k (
        .clk(clk), .rstn(rstn),
        .valid_i(v_k),
        .k_i(k_sel),
        .scale_o(pow2k_w), .valid_o(v_pow2k)
    );

    // 정렬: v_poly와 v_pow2k를 맞춰준다 (파이프 지연 조정)
    wire [DW-1:0] poly_aligned, pow2k_aligned; wire v_align;
    align_pair #(.DW(DW)) u_align (
        .clk(clk), .rstn(rstn),
        .a_i(poly_w),  .va_i(v_poly),
        .b_i(pow2k_w), .vb_i(v_pow2k),
        .a_o(poly_aligned), .b_o(pow2k_aligned),
        .v_o(v_align)
    );

    // 최종: exp = poly * 2^k
    fp16_mul u_mul_final (
        .clk(clk), .rstn(rstn),
        .valid_i(v_align),
        .a_i(poly_aligned), .b_i(pow2k_aligned),
        .p_o(exp_o), .valid_o(valid_o)
    );
endmodule
