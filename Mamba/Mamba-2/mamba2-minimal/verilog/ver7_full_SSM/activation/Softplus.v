//------------------------------------------------------------------------------
// softplus_or_exp16.v
// Mode-selectable FP16 softplus/exp using a single exp16_base2_pwl8 instance.
// y_soft ≈ where(x==0, ln2, x / (1 - exp(-x/ln2)))
// y_exp  = exp(x)
//
// Notes:
// - Pipeline latencies are parameterized; set to match your IP configs.
// - exp가 더 먼저 나옴: mode가 0일때
//------------------------------------------------------------------------------

module softplus_or_exp16 #(
    parameter integer DW       = 16,
    // Latencies for your FP16 IPs (set these to your actual IP configs)
    parameter integer LAT_MUL  = 6,    // FP16 mul latency
    parameter integer LAT_ADD  = 11,   // FP16 add/sub latency
    parameter integer LAT_DIV  = 14,   // FP16 div latency
    // Latency from exp.valid_i to exp.valid_o for your exp16_base2_pwl8
    parameter integer LAT_EXP  = 20,   // <-- set to actual exp module latency
    // FP16 constants (IEEE-754 half precision bit patterns)
    parameter [DW-1:0] FP16_ONE     = 16'h3C00, // 1.0
    parameter [DW-1:0] FP16_LN2     = 16'h398C, // ln(2) ≈ 0.693147
    parameter [DW-1:0] FP16_INV_LN2 = 16'h3DC5  // 1/ln(2) ≈ 1.442695
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire          mode_softplus_i, // 1: softplus, 0: exp
    input  wire [DW-1:0] x_i,
    output wire [DW-1:0] y_o_S,
    output wire          valid_o_S,
    output wire [DW-1:0] y_o_e,
    output wire          valid_o_e,
    output wire          mode_softplus_o
);

    // --------------------------
    // Stage A: Preprocess for EXP input
    // Compute t_soft = (-x) * (1/ln2) when softplus, else pass x.
    // Align both to feed exp on the same cycle.
    // --------------------------

    // Negate x in FP16 by flipping sign bit (OK for ±0 too)
    wire [DW-1:0] x_neg = {~x_i[DW-1], x_i[DW-2:0]};

    // x_delayed_for_exp_path: delay x by LAT_MUL to align with the mul output
    wire [DW-1:0] x_d_mul_aligned;
    wire          v_d_mul_aligned;
    wire          mode_d_mul_aligned;

    shift_reg #(.DW(DW+2), .DEPTH(LAT_MUL)) u_align_x_mode_preexp (
        .clk(clk), .rstn(rstn),
        .din({x_i, mode_softplus_i, valid_i}),
        .dout({x_d_mul_aligned, mode_d_mul_aligned, v_d_mul_aligned})
    );

    // t_soft = x_neg * inv_ln2
    wire [DW-1:0] t_soft;
    wire          t_soft_v;

    fp16_mul #(.DW(DW), .LAT(LAT_MUL)) u_mul_invln2 (
        .clk(clk), .rstn(rstn),
        .valid_i(valid_i),
        .a_i(x_neg),
        .b_i(FP16_INV_LN2),
        .p_o(t_soft),
        .valid_o(t_soft_v)
    );
    

    // Select exp input: softplus ? t_soft : x (aligned)
    wire [DW-1:0] exp_x_i  = mode_d_mul_aligned ? t_soft       : x_d_mul_aligned;
    wire          exp_v_i  = v_d_mul_aligned; // both arms aligned in time

    // --------------------------
    // Stage B: Single EXP engine (shared)
    // --------------------------

    wire [DW-1:0] exp_y;
    wire          exp_v_o;

    exp16_base2_pwl8 #(
        .DW(DW),
        .LAT_MUL(LAT_MUL),
        .LAT_ADD(LAT_ADD)
        // K_MIN/K_MAX as needed
    ) u_exp (
        .clk(clk),
        .rstn(rstn),
        .valid_i(exp_v_i),
        .x_i(exp_x_i),   // interprets as "natural exp input"; module does base-2 internally
        .y_o(exp_y),     // ≈ exp(exp_x_i)
        .valid_o(exp_v_o)
    );

    // Also carry forward x & mode to match exp_y timing (LAT_EXP cycles after exp_v_i)
    wire [DW-1:0] x_d_expo;
    wire          mode_d_expo;
    wire          v_d_expo;

    shift_reg #(.DW(DW+2), .DEPTH(LAT_EXP)) u_align_x_mode_postexp (
        .clk(clk), .rstn(rstn),
        .din({x_d_mul_aligned, mode_d_mul_aligned, exp_v_i}),
        .dout({x_d_expo,       mode_d_expo,       v_d_expo})
    );

    assign y_o_e = exp_y;
    assign valid_o_e = exp_v_o & ~mode_d_expo;

    // --------------------------
    // Stage C: Post-EXP
    // For softplus: denom = (1 - exp_y), y_soft = x / denom
    // For exp: just pass exp_y through a delay to match overall latency
    // --------------------------

    // denom = 1 - exp_y
    wire [DW-1:0] one_minus_exp;
    wire          one_minus_exp_v;

    fp16_add #(.DW(DW), .LAT(LAT_ADD)) u_sub_one_minus_exp (
        .clk(clk), .rstn(rstn),
        .valid_i(exp_v_o),
        .a_i(FP16_ONE),
        .b_i({~exp_y[DW-1], exp_y[DW-2:0]}),
        .sum_o(one_minus_exp),
        .valid_o(one_minus_exp_v)
    );
    
    // y_soft = x / (1 - exp_y)
    // x must be aligned to one_minus_exp_v timing
    wire [DW-1:0] x_d_for_div;
    wire          v_d_for_div;
    wire          mode_d_for_div;

    shift_reg #(.DW(DW+2), .DEPTH(LAT_ADD)) u_align_x_mode_for_div (
        .clk(clk), .rstn(rstn),
        .din({x_d_expo, mode_d_expo, exp_v_o}),
        .dout({x_d_for_div, mode_d_for_div, v_d_for_div})
    );

    wire [DW-1:0] y_soft_div;
    wire          y_soft_div_v;

    div_fp16 #(.DW(DW), .LAT(LAT_DIV)) u_div_softplus (
        .clk(clk), .rstn(rstn),
        .valid_i(one_minus_exp_v),
        .a_i(x_d_for_div),
        .b_i(one_minus_exp),
        .y_o(y_soft_div),
        .valid_o(y_soft_div_v)
    );

    // --------------------------
    // Stage D: x==0 ? ln2 : (x/(1-exp(...)))  (softplus 전용 보호)
    // 입력에서의 x==0 플래그를 최종 출력 타이밍까지 전달
    // 총 지연 = StageA(LAT_MUL) + LAT_EXP + LAT_ADD + LAT_DIV
    //       = LAT_MUL + LAT_EXP + POST_SOFT_LAT
    // --------------------------
    localparam integer POST_SOFT_LAT = (LAT_ADD + LAT_DIV);
    localparam integer TOT_LAT = LAT_MUL + LAT_EXP + POST_SOFT_LAT;

    wire x_is_zero_in  = (x_i[DW-2:0] == { (DW-1){1'b0} }); // exp==0 && frac==0 → ±0
    wire x_zero_at_out;

    shift_reg #(.DW(1), .DEPTH(TOT_LAT)) u_zero_flag_delay (
        .clk(clk), .rstn(rstn),
        .din(x_is_zero_in),
        .dout(x_zero_at_out)
    );

    wire mode_softplus_o_delay;

    shift_reg #(.DW(1), .DEPTH(TOT_LAT)) u_mode_delay (
        .clk(clk), .rstn(rstn),
        .din(mode_softplus_i),
        .dout(mode_softplus_o_delay)
    );

    assign mode_softplus_o = mode_softplus_o_delay;

    assign y_o_S     = x_zero_at_out ? FP16_LN2 : y_soft_div;
    assign valid_o_S = y_soft_div_v & mode_d_for_div;

endmodule

// ----------------------
// Simple shift register for alignment
// ----------------------
module shift_reg #(
    parameter integer DW    = 8,
    parameter integer DEPTH = 1
)(
    input  wire           clk,
    input  wire           rstn,
    input  wire [DW-1:0]  din,
    output wire [DW-1:0]  dout
);
    generate
        if (DEPTH == 0) begin : g_bypass
            assign dout = din;
        end else begin : g_shift
            reg [DW-1:0] pipe [0:DEPTH-1];
            integer i;
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    for (i=0; i<DEPTH; i=i+1) pipe[i] <= {DW{1'b0}};
                end else begin
                    pipe[0] <= din;
                    for (i=1; i<DEPTH; i=i+1) pipe[i] <= pipe[i-1];
                end
            end
            assign dout = pipe[DEPTH-1];
        end
    endgenerate
endmodule
