// ================================================================
// base2_k_picker (3-cycle total latency, glitch-free f_o)
//  - Stage A: t 파싱/전처리(ipart_abs, frac_nz, flags) 레지스터
//  - Stage B: k_floor 결론 + 클램프 + k→FP16(LUT) + (-k) 레지스터
//  - Stage C: f = t - k  (fp16_add, latency = LAT_ADD)
//  출력은 add_v에 게이트되어 글리치 없음
// ================================================================
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
            // 1.xxx * 2^shift, where man = 1.xxx in 11비트 정수 표현
            // man16 = {1xxxx.x}를 16비트로 확장 (비트10이 암묵적 1)
            // 16'd1024 == (1 << 10)
            // {6'd0, m} 로 하위 10비트에 분수 배치
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
    // Stage B (t + 2): k_floor 결론 + 클램프 + FP16 변환(LUT) + (-k)
    // ============================================================
    reg                 v1;
    reg  [DW-1:0]       t_d1;
    reg  [DW-1:0]       nkfp_d1;

    // k_floor 결론 (Stage A 레지스터 기반)
    wire signed [15:0] k_floor_s1 =
        is_inf_nan_r ? (s_r ? -16'sd32768 : 16'sd32767) :
        is_zero_r    ? 16'sd0 :
        is_subnorm_r ? (s_r ? -16'sd1 : 16'sd0) :
        (!s_r)       ? (shift_r < 0 ? 16'sd0 : $signed(ipart_abs_r)) :
        (shift_r < 0)? -16'sd1 :
        (frac_nz_r)  ? -$signed(ipart_abs_r + 16'd1) :
                       -$signed(ipart_abs_r);

    // 클램프
    wire signed [15:0] k_clamped =
        (k_floor_s1 < K_MIN) ? K_MIN :
        (k_floor_s1 > K_MAX) ? K_MAX : k_floor_s1;

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

    wire [DW-1:0] kfp_s1  = k2fp16(k_clamped);
    wire [DW-1:0] nkfp_s1 = {~kfp_s1[DW-1], kfp_s1[DW-2:0]};

    // Stage B 레지스터
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            v1       <= 1'b0;
            t_d1     <= {DW{1'b0}};
            nkfp_d1  <= {DW{1'b0}};
        end else begin
            v1       <= v0;
            t_d1     <= t_d0;
            nkfp_d1  <= nkfp_s1;
        end
    end

    // k/k_fp16는 adder 지연(LAT_ADD)에 맞춰 정렬
    wire  signed [7:0] k_d_align;
    wire  [DW-1:0]     kfp_d_align;

    // ※ 입력은 Stage-B 시점의 값 (t+1 경계에서 캡처됨)
    shift_reg #(.DW(8),  .DEPTH(LAT_ADD)) u_align_k (
        .clk(clk), .rstn(rstn),
        .din(k_clamped[7:0]),
        .dout(k_d_align)
    );

    shift_reg #(.DW(DW), .DEPTH(LAT_ADD)) u_align_kfp (
        .clk(clk), .rstn(rstn),
        .din(kfp_s1),
        .dout(kfp_d_align)
    );

    // ============================================================
    // Stage C (t + 2 + LAT_ADD): f = t - k
    // ============================================================
    wire [DW-1:0] sum_w;
    wire          add_v;

    fp16_add u_sub (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v1),      // t+1 기준
        .a_i     (t_d1),    // t 스냅샷
        .b_i     (nkfp_d1), // -k 스냅샷
        .sum_o   (sum_w),   // t+1+LAT_ADD
        .valid_o (add_v)
    );

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
