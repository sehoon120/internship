// ============================================================================
// base2_k_picker_v2 (4-cycle total latency: A → B1 → B2 → C(+LAT_ADD))
//  - 목적: Stage-B 과밀 경로(클램프 + LUT + 부호반전) 분리로 타이밍 개선
//  - 변경: Stage-B를 B1(정수 k_floor + clamp) / B2(k→FP16 LUT + -k)로 분리
//  - 출력은 fp16_add의 valid_o(add_v)에 정렬되어 글리치 없음
//  - 기능 동일, 파이프라인 1cycle 증가
// ----------------------------------------------------------------------------
// 타이밍 가이드:
//  * A: 전처리(ipart_abs/frac_nz 등)
//  * B1: k_floor_s1 계산 + 클램프 → k_clamped_r ([-16..16], 8bit 저장)
//  * B2: k2fp16(k_clamped_r) → nkfp_d1 레지스터
//  * C: f = t - k  (fp16_add, latency = LAT_ADD)
//  * k_o/k_fp16_o는 add_v에 맞춰 정렬됨(shift_reg DEPTH = LAT_ADD)
// ============================================================================
module base2_k_picker #(
    parameter integer DW      = 16,
    parameter integer K_MIN   = -16,
    parameter integer K_MAX   =  16,
    parameter integer LAT_ADD =  1   // fp16_add의 싸이클 지연
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] t_i,          // FP16 (1-5-10)
    output reg  signed [7:0] k_o,      // aligned with valid_o
    output reg  [DW-1:0]     k_fp16_o, // aligned with valid_o
    output wire [DW-1:0]     f_o,      // aligned with valid_o
    output reg               valid_o
);

    // ---------------- FP16 fields ----------------
    wire        s = t_i[15];
    wire [4:0]  e = t_i[14:10];
    wire [9:0]  m = t_i[9:0];

    wire is_zero    = (e==5'd0) && (m==10'd0);
    wire is_subnorm = (e==5'd0) && (m!=10'd0);
    wire is_inf_nan = (e==5'd31);
    wire signed [6:0] shift = $signed({1'b0,e}) - 7'sd15;

    // ============================================================
    // Stage A (t + 1): 전처리 결과를 레지스터에 저장
    // ============================================================
    reg                 v0;
    reg  [DW-1:0]       t_d0;

    reg                 s_r;
    reg  signed [6:0]   shift_r;
    reg  [15:0]         ipart_abs_r;
    reg                 frac_nz_r;
    reg                 is_zero_r, is_subnorm_r, is_inf_nan_r;

    // 전처리 콤비: ipart_abs / frac_nz
    reg  [15:0] ipart_abs_c;
    reg         frac_nz_c;

    always @* begin
        ipart_abs_c = 16'd0;
        frac_nz_c   = 1'b0;

        if (!(is_inf_nan || is_zero || is_subnorm)) begin
            // 1.xxx * 2^shift, man16 = {1, m[9:0]} << or >>
            if (shift >= 10) begin
                ipart_abs_c = (16'd1024 | {6'd0, m}) << (shift - 10);
                frac_nz_c   = 1'b0;
            end else if (shift >= 0) begin
                ipart_abs_c = (16'd1024 | {6'd0, m}) >> (10 - shift);
                case (10 - shift)
                    1:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h0001) != 0);
                    2:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h0003) != 0);
                    3:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h0007) != 0);
                    4:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h000F) != 0);
                    5:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h001F) != 0);
                    6:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h003F) != 0);
                    7:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h007F) != 0);
                    8:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h00FF) != 0);
                    9:  frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h01FF) != 0);
                    10: frac_nz_c = (((16'd1024 | {6'd0,m}) & 16'h03FF) != 0);
                    default: frac_nz_c = 1'b0;
                endcase
            end else begin
                ipart_abs_c = 16'd0;
                frac_nz_c   = 1'b1;
            end
        end
    end

    // Stage A 레지스터
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v0           <= 1'b0;
            t_d0         <= {DW{1'b0}};
            s_r          <= 1'b0;
            shift_r      <= 1'b0;
            ipart_abs_r  <= 16'd0;
            frac_nz_r    <= 1'b0;
            is_zero_r    <= 1'b0;
            is_subnorm_r <= 1'b0;
            is_inf_nan_r <= 1'b0;
        end else begin
            v0           <= valid_i;
            t_d0         <= t_i;
            s_r          <= s;
            shift_r      <= shift;
            ipart_abs_r  <= ipart_abs_c;
            frac_nz_r    <= frac_nz_c;
            is_zero_r    <= is_zero;
            is_subnorm_r <= is_subnorm;
            is_inf_nan_r <= is_inf_nan;
        end
    end

    // ============================================================
    // Stage B1a (t + 2): k_floor 결론만 레지스터
    // ============================================================
    reg                 v1a;
    reg  [DW-1:0]       t_d1a;
    reg  signed [15:0]  k_floor_r;     // 16b 보존

    // k_floor_s1 (Stage A 결과만 사용)
    wire signed [15:0] k_floor_s1 =
        is_inf_nan_r ? (s_r ? -16'sd32768 : 16'sd32767) :
        is_zero_r    ? 16'sd0 :
        is_subnorm_r ? (s_r ? -16'sd1 : 16'sd0) :
        (!s_r)       ? (shift_r < 0 ? 16'sd0 : $signed(ipart_abs_r)) :
        (shift_r < 0)? -16'sd1 :
        (frac_nz_r)  ? -$signed(ipart_abs_r + 16'd1) :
                       -$signed(ipart_abs_r);

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v1a       <= 1'b0;
            t_d1a     <= {DW{1'b0}};
            k_floor_r <= 16'sd0;
        end else begin
            v1a       <= v0;
            t_d1a     <= t_d0;
            k_floor_r <= k_floor_s1;
        end
    end

    // ============================================================
    // Stage B1b (t + 3): 클램프만 레지스터 (8b 축소)
    // ============================================================
    reg                 v1b;
    reg  [DW-1:0]       t_d1b;
    reg  signed [7:0]   k_clamped_r;   // [-16..+16]

    // 클램프 (16비트 비교 → 저장은 8비트)
    wire signed [15:0] k_clamped_s1 =
        (k_floor_r < K_MIN) ? K_MIN :
        (k_floor_r > K_MAX) ? K_MAX : k_floor_r;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v1b          <= 1'b0;
            t_d1b        <= {DW{1'b0}};
            k_clamped_r  <= 8'sd0;
        end else begin
            v1b          <= v1a;
            t_d1b        <= t_d1a;
            k_clamped_r  <= k_clamped_s1[7:0]; // sign-preserve
        end
    end

    // ============================================================
    // Stage B2 (t + 4): 정수→FP16 LUT + (-k) 레지스터
    // ============================================================
    reg                 v1c;
    reg  [DW-1:0]       t_d1c;
    reg  [DW-1:0]       nkfp_d1;

    // 정수 → FP16: 작은 LUT (-16..16)
    function [15:0] k2fp16;
        input signed [15:0] kval;
        begin
            case (kval)
                // 음수
                -16: k2fp16 = 16'hCC00; -15: k2fp16 = 16'hCB80; -14: k2fp16 = 16'hCB00;
                -13: k2fp16 = 16'hCA80; -12: k2fp16 = 16'hCA00; -11: k2fp16 = 16'hC980;
                -10: k2fp16 = 16'hC900;  -9: k2fp16 = 16'hC880;  -8: k2fp16 = 16'hC800;
                 -7: k2fp16 = 16'hC700;  -6: k2fp16 = 16'hC600;  -5: k2fp16 = 16'hC500;
                 -4: k2fp16 = 16'hC400;  -3: k2fp16 = 16'hC200;  -2: k2fp16 = 16'hC000;
                 -1: k2fp16 = 16'hBC00;
                 // 영/양수
                  0: k2fp16 = 16'h0000;   1: k2fp16 = 16'h3C00;   2: k2fp16 = 16'h4000;
                  3: k2fp16 = 16'h4200;   4: k2fp16 = 16'h4400;   5: k2fp16 = 16'h4500;
                  6: k2fp16 = 16'h4600;   7: k2fp16 = 16'h4700;   8: k2fp16 = 16'h4800;
                  9: k2fp16 = 16'h4880;  10: k2fp16 = 16'h4900;  11: k2fp16 = 16'h4980;
                 12: k2fp16 = 16'h4A00;  13: k2fp16 = 16'h4A80;  14: k2fp16 = 16'h4B00;
                 15: k2fp16 = 16'h4B80;  16: k2fp16 = 16'h4C00;
                default: k2fp16 = 16'h0000; // K_MIN/K_MAX 밖은 이미 클램프됨
            endcase
        end
    endfunction

    wire [DW-1:0] kfp_s1  = k2fp16({{8{k_clamped_r[7]}}, k_clamped_r});
    wire [DW-1:0] nkfp_s1 = {~kfp_s1[DW-1], kfp_s1[DW-2:0]};
    reg  signed [7:0]   k_clamped_r_d;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v1c      <= 1'b0;
            t_d1c    <= {DW{1'b0}};
            nkfp_d1  <= {DW{1'b0}};
            k_clamped_r_d <= {7{1'b0}};
        end else begin
            v1c      <= v1b;
            t_d1c    <= t_d1b;
            nkfp_d1  <= nkfp_s1;
            k_clamped_r_d <= k_clamped_r;
        end
    end

    // ============================================================
    // Stage C (t + 4 + LAT_ADD): f = t - k
    // ============================================================
    wire [DW-1:0] sum_w;
    wire          add_v;

    fp16_add u_sub (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v1c),      // B2 기준
        .a_i     (t_d1c),    // t 스냅샷 (B2 타이밍 기준)
        .b_i     (nkfp_d1),  // -k 스냅샷
        .sum_o   (sum_w),    // B2 + LAT_ADD
        .valid_o (add_v)
    );

    // k/k_fp16 정렬: B2 타이밍 기준으로 정렬 소스 이동 + LAT_ADD==0 가드
reg signed [7:0] k_clamped_b2_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) k_clamped_b2_r <= 8'sd0; else if (v1c) k_clamped_b2_r <= k_clamped_r_d;
end

wire  signed [7:0] k_d_align;
wire  [DW-1:0]     kfp_d_align;

generate
if (LAT_ADD == 0) begin : G_ALIGN0
    assign k_d_align   = k_clamped_b2_r;
    assign kfp_d_align = kfp_s1;
end else begin : G_ALIGNN
    shift_reg #(.DW(8),  .DEPTH(LAT_ADD-1)) u_align_k (
        .clk(clk), .rstn(rstn), .din(k_clamped_b2_r), .dout(k_d_align)
    );
    shift_reg #(.DW(DW), .DEPTH(LAT_ADD+1)) u_align_kfp (
        .clk(clk), .rstn(rstn), .din(kfp_s1), .dout(kfp_d_align)
    );
end
endgenerate

// 출력 레지스터(글리치 방지)
reg [DW-1:0] f_r;
always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        f_r       <= {DW{1'b0}};
        k_o       <= 8'sd0;
        k_fp16_o  <= {DW{1'b0}};
        valid_o   <= 1'b0;
    end else begin
        valid_o   <= add_v;
        if (add_v) begin
            f_r      <= sum_w;
            k_o      <= k_d_align;     // adder 결과와 정렬
            k_fp16_o <= kfp_d_align;
        end
    end
end

assign f_o = f_r;

endmodule
