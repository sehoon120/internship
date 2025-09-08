module softplus16 #(
    parameter integer DW = 16,
    // 레이턴시: exp 모듈/덧셈/곱셈 정렬
    parameter integer LAT_MUL = 6,
    parameter integer LAT_ADD = 11
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] x_i,         // FP16
    output wire [DW-1:0] y_o,         // softplus(x)
    output wire          valid_o
);
    // ====== |x|, sign ======
    wire [DW-1:0] ax_w; wire sgn_neg; wire v_ax;
    fp16_abs u_abs (.x_i(x_i), .abs_o(ax_w));               // 조그만 조합회로
    fp16_sign u_sgn(.x_i(x_i), .is_neg_o(sgn_neg));         // 조합회로
    pipe_v #(.D(1)) u_v0 (.clk(clk), .rstn(rstn), .vi(valid_i), .vo(v_ax)); // valid 파이프

    // ====== t = exp(-|x|) ======
    wire [DW-1:0] n_ax_w; wire v_nax;
    fp16_neg u_neg (.x_i(ax_w), .neg_o(n_ax_w));            // -|x|
    pipe_v #(.D(0)) u_v1 (.clk(clk), .rstn(rstn), .vi(v_ax), .vo(v_nax));

    wire [DW-1:0] t_w; wire v_t;
    exp16_fast u_exp (.clk(clk), .rstn(rstn), .valid_i(v_nax), .x_i(n_ax_w), .exp_o(t_w), .valid_o(v_t));

    // ====== 임계기반 단축: |x| > TH_HI → softplus ≈ max(0,x) ======
    // FP16 TH_HI ≈ 11.0 (exp(-11)≈1.67e-5 → log1p≈그대로 t)
    localparam [DW-1:0] TH_HI = 16'h4B00; // ~11.0 (예시, 필요시 조정)
    wire is_large;  // ax > TH_HI
    fp16_gt u_gt_large (.a_i(ax_w), .b_i(TH_HI), .gt_o(is_large));

    // base = max(0,x)
    wire [DW-1:0] zero16; fp16_const_zero u_zero(.zero_o(zero16));
    wire [DW-1:0] base_w;  // if (x<0)?0:x
    fp16_selpos u_selbase (.x_i(x_i), .pos_o(base_w)); // sign bit로 0 vs x 선택

    // ====== log1p(t) 근사: t - 1/2 t^2 + 1/3 t^3 (t in [0,1])
    wire [DW-1:0] t2_w, t3_w, half, third;
    wire v_t2, v_t3;

    fp16_mul u_t2 (.clk(clk), .rstn(rstn), .valid_i(v_t),   .a_i(t_w), .b_i(t_w), .p_o(t2_w), .valid_o(v_t2));
    fp16_mul u_t3 (.clk(clk), .rstn(rstn), .valid_i(v_t2),  .a_i(t2_w), .b_i(t_w), .p_o(t3_w), .valid_o(v_t3));

    fp16_const_half  u_half (.half_o(half));   // 0.5
    fp16_const_third u_thrd (.third_o(third)); // 0.333333

    wire [DW-1:0] mhalf_t2; wire v_mhalf_t2;
    fp16_mul u_mhalf (.clk(clk), .rstn(rstn), .valid_i(v_t2), .a_i(half), .b_i(t2_w), .p_o(mhalf_t2), .valid_o(v_mhalf_t2));

    wire [DW-1:0] thrd_t3; wire v_thrd_t3;
    fp16_mul u_thrd (.clk(clk), .rstn(rstn), .valid_i(v_t3), .a_i(third), .b_i(t3_w), .p_o(thrd_t3), .valid_o(v_thrd_t3));

    // corr = t - 0.5 t^2 + (1/3)t^3
    wire [DW-1:0] s1_w; wire v_s1;
    fp16_add u_s1 (.clk(clk), .rstn(rstn), .valid_i(v_t & v_mhalf_t2), .a_i(t_w), .b_i({~mhalf_t2[DW-1], mhalf_t2[DW-2:0]}), .sum_o(s1_w), .valid_o(v_s1));
    wire [DW-1:0] corr_w; wire v_corr;
    fp16_add u_s2 (.clk(clk), .rstn(rstn), .valid_i(v_s1 & v_thrd_t3), .a_i(s1_w), .b_i(thrd_t3), .sum_o(corr_w), .valid_o(v_corr));

    // ====== 최종 y = is_large ? base : (base + corr)
    wire [DW-1:0] base_al, corr_al; wire v_al;
    align_pair #(.DW(DW)) u_align2 (
        .clk(clk), .rstn(rstn),
        .a_i(base_w), .va_i(v_corr),   // base를 corr 타이밍에 맞춤
        .b_i(corr_w), .vb_i(v_corr),
        .a_o(base_al), .b_o(corr_al), .v_o(v_al)
    );

    wire [DW-1:0] y_full; wire v_full;
    fp16_add u_add_final (.clk(clk), .rstn(rstn), .valid_i(v_al), .a_i(base_al), .b_i(corr_al), .sum_o(y_full), .valid_o(v_full));

    // large-path 선택
    // is_large는 초기에 평가되므로, 그 valid를 v_full 타이밍으로 가져오자
    wire is_large_d; pipe_flag #(.D( /*corr latency*/  )) u_large_d (.clk(clk), .rstn(rstn), .vi(valid_i), .flag_i(is_large), .flag_o(is_large_d));

    assign y_o     = is_large_d ? base_al : y_full;
    assign valid_o = v_full;  // large일 때도 v_full 타이밍에 base_al가 정렬됨
endmodule
