// ============================================================================
// SSMBLOCK_TOP — tile-in → N-accum → (group-accum over TILES_PER_GROUP) → +xD
// y_final_o: size (H_TILE*P_TILE), y_final_valid_o: group 마다 1펄스
// ----------------------------------------------------------------------------
// 가정:
//  - 하위 모듈(delta, sp_dt, dA_mul, dx_mul, dA_exp, dBx_mul, dAh_mul,
//    hnext_add, hC_mul, accum_n, add_hp, xD_mul)은 이미 존재하며
//    포트 폭은 아래 선언과 호환.
//  - backpressure는 사용하지 않아 tile_ready_o=1'b1.
//  - 각 valid는 모듈 latency에 맞춰 내부에서 정렬됨.
// ----------------------------------------------------------------------------

module SSMBLOCK_TOP #(
    parameter integer DW          = 16,
    parameter integer H_TILE      = 1,
    parameter integer P_TILE      = 1,
    parameter integer N_TILE      = 64,
    parameter integer N_TOTAL     = 128,

    // Latency params (IP 설정에 맞춰 조정)
    parameter integer LAT_DX_M    = 6,    // mul latency (dx, dBx, dAh, hC 등에 공통 사용)
    parameter integer LAT_DBX_M   = 6,    // (옵션) dBx 별도 mul latency
    parameter integer LAT_DAH_M   = 6,    // (옵션) dAh 별도 mul latency
    parameter integer LAT_ADD_A   = 11,   // add latency (delta, h_next 등)
    parameter integer LAT_HC_M    = 6,    // hC mul latency
    parameter integer LAT_MUL     = 6
    parameter integer LAT_ADD     = 11
    parameter integer LAT_DIV     = 17
    parameter integer LAT_EXP     = 6 + LAT_MUL * 3 + LAT_ADD * 3,     // exp latency (예시)
    parameter integer LAT_SP      = LAT_EXP + LAT_MUL + LAT_ADD + LAT_DIV + 1    // Softplus latency (예시)
)(
    input  wire                              clk,
    input  wire                              rstn,

    // 타일 유효(연속 타일 스트리밍), 마지막 타일 표시는 TB가 관리
    input  wire                              tile_valid_i,
    output wire                              tile_ready_o,   // backpressure 미사용 → 1

    // Scalars / small vectors
    input  wire [H_TILE*DW-1:0]              dt_i,
    input  wire [H_TILE*DW-1:0]              dt_bias_i,
    input  wire [H_TILE*DW-1:0]              A_i,
    input  wire [H_TILE*P_TILE*DW-1:0]       x_i,
    input  wire [H_TILE*DW-1:0]              D_i,

    // Tile vectors (N_TILE)
    input  wire [N_TILE*DW-1:0]              B_tile_i,
    input  wire [N_TILE*DW-1:0]              C_tile_i,
    input  wire [H_TILE*P_TILE*N_TILE*DW-1:0] hprev_tile_i,

    // 최종 출력: y = sum_{n} hC[n] + x*D
    output wire [H_TILE*P_TILE*DW-1:0]       y_final_o,
    output wire                              y_final_valid_o
);
    // ------------------------------------------------------------
    localparam integer TILES_PER_GROUP = (N_TOTAL + N_TILE - 1) / N_TILE; // 안전 계산

    // 항상 수신 가능 (현재 설계에선 backpressure 없음)
    assign tile_ready_o = 1'b1;

    // ============================================================
    // 0) delta = dt + dt_bias (h)  —— add
    // ============================================================
    wire [H_TILE*DW-1:0] delta_w;
    wire                 v_delta_w;

    delta #(.DW(DW), .H_TILE(H_TILE), .A_LAT(LAT_ADD_A)) u_delta (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i),
        .dt_i    (dt_i),
        .bias_i  (dt_bias_i),
        .sum_o   (delta_w),
        .valid_o (v_delta_w)
    );

    // ============================================================
    // 0’) xD = x * D (hp) —— mul
    //   그룹 끝에서 y_tmp와 더하기 위해 레지스터로 홀드
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] xD_w;
    wire                        v_xD_w;

    xD #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .M_LAT(LAT_DX_M)) u_xD (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i),
        .x_i     (x_i),   // (h*p)
        .D_i     (D_i),   // (h)
        .xD_o    (xD_w),  // (h*p)
        .valid_o (v_xD_w)
    );

    // 그룹마다 xD를 한 번만 쓰도록 래치
    reg [H_TILE*P_TILE*DW-1:0] xD_latched_r;
    reg                        xD_latched_v;
    // 그룹 진행 상태 추적
    reg [$clog2(TILES_PER_GROUP+1)-1:0] group_tile_cnt_r;

    wire group_start  = (group_tile_cnt_r == 0) && tile_valid_i;
    wire group_last   = (group_tile_cnt_r == TILES_PER_GROUP-1) && tile_valid_i;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            group_tile_cnt_r <= 0;
        end else if (tile_valid_i) begin
            if (group_last)
                group_tile_cnt_r <= 0;
            else
                group_tile_cnt_r <= group_tile_cnt_r + 1'b1;
        end
    end

    // xD 값은 그룹의 첫 타일 타이밍에서 유효해지는 v_xD_w를 감지해 래치
    // (x, D가 그룹 내에서 불변이라는 가정)
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            xD_latched_r <= {H_TILE*P_TILE*DW{1'b0}};
            xD_latched_v <= 1'b0;
        end else begin
            // 첫 타일 구간에서 v_xD_w가 뜨면 래치
            if (group_start && v_xD_w) begin
                xD_latched_r <= xD_w;
                xD_latched_v <= 1'b1;
            end
            // 그룹 끝에서 y_out 사용 후 클리어 (y_final_valid_o 발생과 동기)
            if (y_final_valid_o) begin
                xD_latched_v <= 1'b0;
            end
        end
    end

    // ============================================================
    // 1) delta_sp = Softplus(delta) (h)
    // ============================================================
    wire [H_TILE*DW-1:0] delta_sp_w;
    wire                 v_delta_sp_w;

    sp_dt #(.DW(DW), .H_TILE(H_TILE), .SP_LAT(LAT_SP)) u_sp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_w),
        .dt_i    (delta_w),
        .sp_dt_o (delta_sp_w),
        .valid_o (v_delta_sp_w)
    );

    // ============================================================
    // 2) dA_tmp = delta_sp * A (h) —— mul
    // ============================================================
    wire [H_TILE*DW-1:0] dA_tmp_w;
    wire                 v_dA_tmp_w;

    dA_mul #(.DW(DW), .H_TILE(H_TILE), .M_LAT(LAT_DX_M)) u_dA_tmp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_sp_w),
        .lhs_i   (delta_sp_w), // (h)
        .rhs_i   (A_i),        // (h)
        .mul_o   (dA_tmp_w),   // (h)
        .valid_o (v_dA_tmp_w)
    );

    // ============================================================
    // 3) dA = exp(dA_tmp) (h) —— exp
    // ============================================================
    wire [H_TILE*DW-1:0] dA_w;
    wire                 v_dA_w;

    dA_exp #(.DW(DW), .H_TILE(H_TILE), .EXP_LAT(LAT_EXP)) u_dA_exp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dA_tmp_w),
        .in_i    (dA_tmp_w),
        .exp_o   (dA_w),
        .valid_o (v_dA_w)
    );

    // ============================================================
    // 2’) dx = delta_sp * x (hp) —— mul  (broadcast h → hp)
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] dx_w;
    wire                        v_dx_w;

    dx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_sp_w),
        .h_i     (delta_sp_w), // (h)
        .x_i     (x_i),        // (h*p)
        .dx_o    (dx_w),       // (h*p)
        .valid_o (v_dx_w)
    );

    // ============================================================
    // 3’) dBx = dx * B_tile (hpn) —— mul (broadcast B[n] → hp)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w;
    wire                               v_dBx_w;

    dBx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx_w),
        .dx_i    (dx_w),      // (h*p)
        .B_tile_i(B_tile_i),  // (n)
        .dBx_o   (dBx_w),     // (h*p*n)
        .valid_o (v_dBx_w)
    );

    // ============================================================
    // 4) dAh = dA * hprev_tile (hpn) —— mul (broadcast dA[h] → p*n)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dAh_w;
    wire                               v_dAh_w;

    dAh_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DAH_M)) u_dAh (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_dA_w),
        .dA_i      (dA_w),          // (h)
        .hprev_i   (hprev_tile_i),  // (h*p*n)
        .dAh_o     (dAh_w),         // (h*p*n)
        .valid_o   (v_dAh_w)
    );

    // ============================================================
    // 5) h_next = dAh + dBx (hpn) —— add
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hnext_w;
    wire                               v_hnext_w;

    hnext_add #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .A_LAT(LAT_ADD_A)) u_hnext (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dAh_w & v_dBx_w), // 두 경로 정렬 가정 (필요 시 내부 정렬)
        .lhs_i   (dAh_w),
        .rhs_i   (dBx_w),
        .sum_o   (hnext_w),
        .valid_o (v_hnext_w)
    );

    // ============================================================
    // 6) hC = h_next * C_tile (hpn) —— mul (broadcast C[n] → h*p)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hC_w;
    wire                               v_hC_w;

    hC_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_HC_M)) u_hC (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_hnext_w),
        .hnext_i   (hnext_w),
        .C_tile_i  (C_tile_i),
        .hC_o      (hC_w),
        .valid_o   (v_hC_w)
    );

    // ============================================================
    // 7) y_tile = accumulation_n(hC) (hpn → hp) —— reduce over n
    //     한 "타일(n방향 N_TILE개)"에 대해 (h*p*n) → (h*p)
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] y_tile_w;
    wire                        v_y_tile_w;

    accum_n #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE)) u_accum_n (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_hC_w),
        .hC_i      (hC_w),      // (h*p*n)
        .sum_hp_o  (y_tile_w),  // (h*p)
        .valid_o   (v_y_tile_w) // "타일" 단위 완료 펄스
    );

    // ============================================================
    // 7’) group_acc: TILES_PER_GROUP 번 y_tile을 누적 (hp → hp)
    //     그룹 마지막에서 valid 펄스
    // ============================================================
    reg [H_TILE*P_TILE*DW-1:0] group_sum_r;
    reg                        group_sum_v;

    // v_y_tile_w가 뜰 때마다 누적, 그룹 마지막 타일에서 valid 띄움
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            group_sum_r <= {H_TILE*P_TILE*DW{1'b0}};
            group_sum_v <= 1'b0;
        end else begin
            if (v_y_tile_w) begin
                if (group_sum_v)
                    group_sum_r <= add_hp(group_sum_r, y_tile_w); // 함수식 사용(아래 function) or 별도 adder
                else
                    group_sum_r <= y_tile_w;

                // 그룹 누적 시작 표시
                if (!group_sum_v)
                    group_sum_v <= 1'b1;
            end

            // 그룹 종료 시점: 입력 타일이 마지막일 때 (동일 싸이클 v_y_tile_w가 가정)
            if (v_y_tile_w && group_last) begin
                // 다음 싸이클에 y_out에서 사용될 수 있도록 유지
                // 이후 y_final_valid_o 발생과 함께 리셋
            end

            // y_final_valid_o 시점에서 다음 그룹 대비 초기화
            if (y_final_valid_o) begin
                group_sum_v <= 1'b0;
            end
        end
    end

    // 간단한 (h*p) 요소별 더하기 함수 (동일 폭, 2's-complement 가정)
    function [H_TILE*P_TILE*DW-1:0] add_hp;
        input [H_TILE*P_TILE*DW-1:0] a;
        input [H_TILE*P_TILE*DW-1:0] b;
        integer k;
        reg   [DW-1:0] sumk;
        begin
            for (k = 0; k < H_TILE*P_TILE; k = k + 1) begin
                sumk = a[DW*(k+1)-1 -: DW] + b[DW*(k+1)-1 -: DW];
                add_hp[DW*(k+1)-1 -: DW] = sumk;
            end
        end
    endfunction

    // ============================================================
    // 8) y_final = group_sum + xD_latched  —— add (hp)
    //     유효 펄스: 그룹 마지막 타일 처리 완료(v_y_tile_w && group_last)와
    //               xD_latched_v가 둘 다 준비된 시점
    // ============================================================
    // ------------------------------
    // [1] 그룹 완료 펄스(마지막 타일) → 1싸이클 딜레이
    //     group_done_pulse = (v_y_tile_w && group_last)  // 기존 정의 재사용
    // ------------------------------
    wire group_done_pulse;  // 기존에 선언되어 있다면 재사용
    // 예: wire group_done_pulse = v_y_tile_w && group_last;

    reg group_done_pulse_d1;
    always @(posedge clk or negedge rstn) begin
    if (!rstn) group_done_pulse_d1 <= 1'b0;
    else        group_done_pulse_d1 <= group_done_pulse;
    end

    // ------------------------------
    // [2] fire_yout: 두 입력이 같은 싸이클에 준비되었을 때 valid
    //     - group_sum_r : 그룹 누적 최종 합(한 싸이클 전에 레지스터에 갱신됨)
    //     - xD_latched_r: 그룹 초반에 래치해 둔 xD
    // ------------------------------
    wire fire_yout = group_done_pulse_d1 & xD_latched_v;

    // ------------------------------
    // [3] y_out 인스턴스 (FP16 add IP 사용)
    //     A_LAT는 fp16_add_wrapper latency와 맞춰주세요 (보통 10~12)
    // ------------------------------
    wire [H_TILE*P_TILE*DW-1:0] y_sum_hp;
    wire                        v_y_sum_hp;

    y_out #(
    .DW     (DW),
    .H_TILE (H_TILE),
    .P_TILE (P_TILE),
    .A_LAT  (LAT_ADD_A)
    ) u_y_out (
    .clk        (clk),
    .rstn       (rstn),
    .valid_i    (fire_yout),      // 이 싸이클에 두 입력이 모두 유효
    .group_sum_i(group_sum_r),    // (hp) 그룹 누적 최종 합
    .xD_i       (xD_latched_r),   // (hp) x*D (그룹 시작에 래치)
    .y_o        (y_sum_hp),       // (hp)
    .valid_o    (v_y_sum_hp)
    );

    // 최종 출력
    assign y_final_o       = y_sum_hp;
    assign y_final_valid_o = v_y_sum_hp;

    // xD 래치 클리어: 최종 출력이 나간 싸이클에 0으로
    always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        xD_latched_v <= 1'b0;
    end else if (v_y_sum_hp) begin
        xD_latched_v <= 1'b0;
    end
    end


endmodule
