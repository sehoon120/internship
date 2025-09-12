// 2-cycle total latency, glitch-free f_o
module base2_k_picker #(
    parameter integer DW    = 16,
    parameter integer K_MIN = -16,
    parameter integer K_MAX =  16,
    parameter integer LAT_ADD =  1
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] t_i,          // FP16 (1-5-10)
    output reg  signed [7:0] k_o,      // aligned with valid_o
    output reg  [DW-1:0]     k_fp16_o, // aligned with valid_o
    output wire [DW-1:0]     f_o,      // aligned with valid_o
    output reg              valid_o
);

    // ---------------- FP16 fields ----------------
    wire        s = t_i[15];
    wire [4:0]  e = t_i[14:10];
    wire [9:0]  m = t_i[9:0];

    wire is_zero    = (e==5'd0) && (m==10'd0);
    wire is_subnorm = (e==5'd0) && (m!=10'd0);
    wire is_inf_nan = (e==5'd31);
    wire signed [6:0] shift = $signed({1'b0,e}) - 7'sd15;

    // ------------- floor(t): PURE COMB -------------
    reg  signed [15:0] k_floor_c;
    reg  [15:0]        ipart_abs;
    reg                frac_nz;

    always @* begin
        ipart_abs = 16'd0;
        frac_nz   = 1'b0;
        k_floor_c = 16'sd0;

        if (is_inf_nan) begin
            k_floor_c = s ? -32768 : 32767; // later saturate
        end else if (is_zero) begin
            k_floor_c = 16'sd0;
        end else if (is_subnorm) begin
            k_floor_c = s ? -16'sd1 : 16'sd0;
        end else begin
            if (shift >= 10) begin
                ipart_abs = (16'd1024 | {6'd0,m}) << (shift - 10);
                frac_nz   = 1'b0;
            end else if (shift >= 0) begin
                ipart_abs = (16'd1024 | {6'd0,m}) >> (10 - shift);
                case (10 - shift)
                    1:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h0001) != 0;
                    2:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h0003) != 0;
                    3:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h0007) != 0;
                    4:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h000F) != 0;
                    5:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h001F) != 0;
                    6:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h003F) != 0;
                    7:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h007F) != 0;
                    8:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h00FF) != 0;
                    9:  frac_nz = ((16'd1024 | {6'd0,m}) & 16'h01FF) != 0;
                    10: frac_nz = ((16'd1024 | {6'd0,m}) & 16'h03FF) != 0;
                    default: frac_nz = 1'b0;
                endcase
            end else begin
                ipart_abs = 16'd0;
                frac_nz   = 1'b1;
            end

            if (!s) begin
                k_floor_c = (shift < 0) ? 16'sd0 : $signed(ipart_abs);
            end else begin
                if (shift < 0)           k_floor_c = -16'sd1;
                else if (frac_nz)        k_floor_c = -$signed(ipart_abs + 16'd1);
                else                     k_floor_c = -$signed(ipart_abs);
            end
        end

        // saturate to [K_MIN, K_MAX]
        if (k_floor_c < K_MIN)      k_floor_c = K_MIN;
        else if (k_floor_c > K_MAX) k_floor_c = K_MAX;
    end

    // -------- int -> FP16 (exact for small ints) --------
    function [15:0] int_to_fp16;
        input signed [15:0] kval;
        reg sign_b; reg [15:0] abs_k; integer p; reg [4:0] exp_b; reg [10:0] comb;
        begin
            if (kval == 0) int_to_fp16 = 16'h0000;
            else begin
                sign_b = (kval < 0);
                abs_k  = sign_b ? -kval : kval;
                p = 15; while (p>0 && ~abs_k[p]) p = p-1;
                exp_b  = p + 15;
                comb   = abs_k << (10 - p);
                int_to_fp16 = {sign_b, exp_b, comb[9:0]};
            end
        end
    endfunction

    // ---------------- Stage-1 (t+1) ----------------
    reg           v1;
    reg  [DW-1:0] t_d1;
//    reg  signed [7:0] k_d1;
//    reg  [DW-1:0] kfp_d1;
    reg  [DW-1:0] nkfp_d1;   // -k (FP16)

    wire [DW-1:0] kfp_c     = int_to_fp16(k_floor_c);
    wire [DW-1:0] nkfp_c    = {~kfp_c[DW-1], kfp_c[DW-2:0]};

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v1      <= 1'b0;
            t_d1    <= {DW{1'b0}};
//            k_d1    <= 8'sd0;
//            kfp_d1  <= {DW{1'b0}};
            nkfp_d1 <= {DW{1'b0}};
        end else begin
            v1      <= valid_i;
            t_d1    <= t_i;
//            k_d1    <= k_floor_c[7:0];
//            kfp_d1  <= kfp_c;
            nkfp_d1 <= nkfp_c;
        end
    end

//    wire [DW-1:0]       two_pow_k_align;
//    wire  signed [7:0] k_align;
    wire  signed [7:0] k_d_align;
//    wire  [DW-1:0] kfp_align;
    wire  [DW-1:0] kfp_d_align;
    
    shift_reg #(.DW(8), .DEPTH(LAT_ADD)) u_align_k (
        .clk(clk), .rstn(rstn),
        .din(k_floor_c[7:0]),
        .dout(k_d_align)
    );
    
    shift_reg #(.DW(DW), .DEPTH(LAT_ADD)) u_align_kfp (
        .clk(clk), .rstn(rstn),
        .din(kfp_c),
        .dout(kfp_d_align)
    );

    // ---------------- Adder (1-cycle) ----------------
    wire [DW-1:0] sum_w;
    wire          add_v;

    fp16_add u_sub (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v1),      // t+1
        .a_i     (t_d1),    // t snapshot
        .b_i     (nkfp_d1), // -k snapshot
        .sum_o   (sum_w),   // t+2
        .valid_o (add_v)    // t+2
    );

    // ---------------- Outputs (t+2, gated by add_v) ----------------
    reg [DW-1:0] f_r;  // 내부 래치
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            f_r       <= {DW{1'b0}};
            k_o       <= 8'sd0;
            k_fp16_o  <= {DW{1'b0}};
        end else if (add_v) begin
            f_r       <= sum_w;
            k_o       <= k_d_align;      // v1 시점의 k → adder 결과와 정렬(t+2)
            k_fp16_o  <= kfp_d_align;
        end
        // add_v==0이면 유지 → 글리치/플리커 없음
    end

    assign f_o     = f_r;
    
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            valid_o <= 0;
        end else begin
            valid_o <= add_v;
        end
    end

endmodule
