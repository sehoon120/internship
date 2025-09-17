// ============================================================================
// SSMBLOCK_TOP - tile-in → N-accum → (group-accum over TILES_PER_GROUP) → +xD
// y_final_o: size (H_TILE*P_TILE), y_final_valid_o: group 마다 1펄스
// ----------------------------------------------------------------------------
// 가정:
//  - 하위 모듈(delta, sp_dt, dA_mul, dx_mul, dA_exp, dBx_mul, dAh_mul,
//    hnext_add, hC_mul, accum_n, add_hp, xD_mul)은 이미 존재하며
//    포트 폭은 아래 선언과 호환.
//  - backpressure는 사용하지 않아 tile_ready_o=1'b1.
//  - 각 valid는 모듈 latency에 맞춰 내부에서 정렬됨.
// ----------------------------------------------------------------------------
`define SIM 

module SSMBLOCK_TOP #(
    parameter integer DW          = 16,
    parameter integer H_TILE      = 1,
    parameter integer P_TILE      = 1,
    parameter integer N_TILE      = 4,
    parameter integer N_TOTAL     = 128,

    // Latency params (IP 설정에 맞춰 조정)
    parameter integer LAT_DX_M    = 6,    // mul latency (dx, dBx, dAh, hC 등에 공통 사용)
    parameter integer LAT_DBX_M   = 6,    // (옵션) dBx 별도 mul latency
    parameter integer LAT_DAH_M   = 6,    // (옵션) dAh 별도 mul latency
    parameter integer LAT_ADD_A   = 11,   // add latency (delta, h_next 등)
    parameter integer LAT_ACCU    = 11*7,
    parameter integer LAT_HC_M    = 6,    // hC mul latency
    parameter integer LAT_MUL     = 6,
    parameter integer LAT_ADD     = 11,
    parameter integer LAT_DIV     = 15,
    parameter integer LAT_EXP     = 6 + LAT_MUL * 3 + LAT_ADD * 3 + 1,     // exp latency (예시)
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
    // 0) delta = dt + dt_bias (h)  -- add
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
    // 0') xD = x * D (hp) -- mul
    //   그룹 끝에서 y_tmp와 더하기 위해 레지스터로 홀드
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] xD_w;
    wire                        v_xD_w;

    xD_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .M_LAT(LAT_DX_M)) u_xD (
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

    wire group_s_d;
    wire group_l_d;
    
    shift_reg #(.DW(2), .DEPTH(LAT_DX_M)) u_group_s_l_delay (
        .clk(clk), .rstn(rstn),
        .din({group_start, group_last}),
        .dout({group_s_d, group_l_d})
    );
    
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
            if (group_s_d && v_xD_w) begin
                xD_latched_r <= xD_w;
                xD_latched_v <= 1'b1;
            end
            // 그룹 끝에서 y_out 사용 후 클리어 (y_final_valid_o 발생과 동기)
            if (group_l_d) begin
                xD_latched_v <= 1'b0;
            end
        end
    end

    wire [H_TILE*P_TILE*DW-1:0] xD_w_d;
    wire  v_xD_w_d;
    shift_reg #(.DW(H_TILE*P_TILE*DW + 1), .DEPTH(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP + LAT_DAH_M + 2 + LAT_DX_M +  LAT_ACCU + N_TOTAL/N_TILE)) u_xd_delay (
        .clk(clk), .rstn(rstn), .din({xD_latched_r, xD_latched_v}), .dout({xD_w_d, v_xD_w_d})
    );

    // ============================================================
    // 1) delta_sp = Softplus(delta) (h)
    // ============================================================
    wire [H_TILE*DW-1:0] delta_sp_w;
    wire                 v_delta_sp_w;

    sp_dt #(.DW(DW), .H_TILE(H_TILE), .SP_LAT(LAT_SP), .LAT_EXP(LAT_EXP), .LAT_MUL(LAT_MUL), .LAT_ADD(LAT_ADD), .LAT_DIV(LAT_DIV)) u_sp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_w),
        .dt_i    (delta_w),
        .sp_dt_o (delta_sp_w),
        .valid_o (v_delta_sp_w)
    );

    // ============================================================
    // 2) dA_tmp = delta_sp * A (h) -- mul
    // ============================================================
    wire [H_TILE*DW-1:0] dA_tmp_w;
    wire                 v_dA_tmp_w;

    wire [H_TILE*DW-1:0] A_i_d;
    shift_reg #(.DW(H_TILE*DW), .DEPTH(LAT_ADD_A + LAT_SP)) u_zero_flag_delay (
        .clk(clk), .rstn(rstn),
        .din(A_i),
        .dout(A_i_d)
    );

    dA_mul #(.DW(DW), .H_TILE(H_TILE), .M_LAT(LAT_DX_M)) u_dA_tmp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_sp_w),
        .lhs_i   (delta_sp_w), // (h)
        .rhs_i   (A_i_d),        // (h)
        .mul_o   (dA_tmp_w),   // (h)
        .valid_o (v_dA_tmp_w)
    );

    // ============================================================
    // 3) dA = exp(dA_tmp) (h) -- exp
    // ============================================================
    wire [H_TILE*DW-1:0] dA_w;
    wire                 v_dA_w;

    dA_exp #(.DW(DW), .H_TILE(H_TILE), .LAT_EXP(LAT_EXP)) u_dA_exp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dA_tmp_w),
        .in_i    (dA_tmp_w),
        .exp_o   (dA_w),
        .valid_o (v_dA_w)
    );

    // ============================================================
    // 2') dx = delta_sp * x (hp) -- mul  (broadcast h → hp)
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] dx_w;
    wire                        v_dx_w;
    wire [H_TILE*P_TILE*DW-1:0] x_i_d;

    shift_reg #(.DW(H_TILE*P_TILE*DW), .DEPTH(LAT_ADD_A + LAT_SP)) u_x_delay (
        .clk(clk), .rstn(rstn),
        .din(x_i),
        .dout(x_i_d)
    );

    dx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_delta_sp_w),
        .h_i     (delta_sp_w), // (h)
        .x_i     (x_i_d),        // (h*p)
        .dx_o    (dx_w),       // (h*p)
        .valid_o (v_dx_w)
    );

    // ============================================================
    // 3') dBx = dx * B_tile (hpn) -- mul (broadcast B[n] → hp)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w;
    wire                               v_dBx_w;
    wire [N_TILE*DW-1:0] B_tile_i_d;

    shift_reg #(.DW(N_TILE*DW), .DEPTH(LAT_ADD_A + LAT_SP + LAT_DX_M + 1)) u_B_w_delay (
        .clk(clk), .rstn(rstn),
        .din(B_tile_i),
        .dout(B_tile_i_d)
    );

    dBx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx_w),
        .dx_i    (dx_w),      // (h*p)
        .B_tile_i(B_tile_i_d),  // (n)
        .dBx_o   (dBx_w),     // (h*p*n)
        .valid_o (v_dBx_w)
    );

    // ============================================================
    // 4) dAh = dA * hprev_tile (hpn) -- mul (broadcast dA[h] → p*n)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dAh_w;
    wire                               v_dAh_w;
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hprev_tile_i_d;

    shift_reg #(.DW(H_TILE*P_TILE*N_TILE*DW), .DEPTH(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP + 2)) u_hprev_delay (
        .clk(clk), .rstn(rstn),
        .din(hprev_tile_i),
        .dout(hprev_tile_i_d)
    );
    
    dAh_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DAH_M)) u_dAh (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_dA_w),
        .dA_i      (dA_w),          // (h)
        .hprev_i   (hprev_tile_i_d),  // (h*p*n)
        .dAh_o     (dAh_w),         // (h*p*n)
        .valid_o   (v_dAh_w)
    );

    // ============================================================
    // 5) h_next = dAh + dBx (hpn) -- add
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hnext_w;
    wire                               v_hnext_w;
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w_d;
    wire                               v_dBx_w_d;

    shift_reg #(.DW(H_TILE*P_TILE*N_TILE*DW+1), .DEPTH((LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP + LAT_DAH_M + 2) - (LAT_ADD_A + LAT_SP + LAT_DX_M + 1 + LAT_DBX_M))) u_dbx_delay (
        .clk(clk), .rstn(rstn),
        .din({dBx_w, v_dBx_w}),
        .dout({dBx_w_d, v_dBx_w_d})
    );

    hnext_add #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .A_LAT(LAT_ADD_A)) u_hnext (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dAh_w & v_dBx_w_d), // 두 경로 정렬 가정 (필요 시 내부 정렬)
        .dAh_i   (dAh_w),
        .dBx_i   (dBx_w_d),
        .sum_o   (hnext_w),
        .valid_o (v_hnext_w)
    );

    // ============================================================
    // 6) hC = h_next * C_tile (hpn) -- mul (broadcast C[n] → h*p)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hC_w;
    wire                               v_hC_w;
    wire [N_TILE*DW-1:0] C_tile_i_d;

    shift_reg #(.DW(N_TILE*DW), .DEPTH(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP + LAT_DAH_M + 2 + LAT_ADD_A)) u_C_delay (
        .clk(clk), .rstn(rstn),
        .din(C_tile_i),
        .dout(C_tile_i_d)
    );
    
    hC_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_HC_M)) u_hC (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_hnext_w),
        .hnext_i   (hnext_w),
        .C_tile_i  (C_tile_i_d),
        .hC_o      (hC_w),
        .valid_o   (v_hC_w)
    );

    // ============================================================
    // 7) y_tile = accumulation_n(hC) (hpn → hp) -- reduce over n)
    // ============================================================
    reg  [H_TILE*P_TILE*N_TILE*DW-1:0] hC_buf [0:TILES_PER_GROUP-1]; // 8개 타일 버퍼
    reg  [4:0]           tile_ptr;   // 0..7
    reg                  grp_emit;   // 이번 싸이클에 8타일이 모였다는 펄스
    wire                 accept_tile = v_hC_w;
    
    integer ti;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            tile_ptr  <= 5'd0;
            grp_emit  <= 1'b0;
            for (ti=0; ti<TILES_PER_GROUP; ti=ti+1) hC_buf[ti] <= {H_TILE*P_TILE*N_TILE*DW{1'b0}};
        end else begin
            grp_emit <= 1'b0;

            if (accept_tile) begin
                // 현재 타일 저장
                hC_buf[tile_ptr] <= hC_w;

                // 타일 포인터 증가 및 그룹 완료
                if (tile_ptr == TILES_PER_GROUP-1) begin
                    tile_ptr <= 5'd0;
                    grp_emit <= 1'b1;    // 8번째 타일이 막 들어온 싸이클
                end else begin
                    tile_ptr <= tile_ptr + 5'd1;
                end
            end
        end
    end

    // 8개 타일을 128-lane 버스로 평탄화
    wire [H_TILE*P_TILE*N_TOTAL*DW-1:0] hC_128_bus;
    assign hC_128_bus = {
        hC_buf[7], hC_buf[6], hC_buf[5], hC_buf[4],
        hC_buf[3], hC_buf[2], hC_buf[1], hC_buf[0]
    };
    
    wire [H_TILE*P_TILE*DW-1:0] y_tile_w;
    wire                        v_y_tile_w;

    accum_n #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TOTAL(N_TOTAL)) u_accum_n (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (grp_emit),
        .hC_i      (hC_128_bus),      // (h*p*n)
        .sum_hp_o  (y_tile_w),  // (h*p)
        .valid_o   (v_y_tile_w) // "타일" 단위 완료 펄스
    );
    
    // ============================================================
    // 8) 최종 합: group_sum_hp + xD_latched  → y_out
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] y_sum_hp;
    wire                        v_y_sum_hp;

    y_out #(
    .DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .A_LAT(LAT_ADD_A)
    ) u_y_out (
    .clk        (clk),
    .rstn       (rstn),
    .valid_i    (v_y_tile_w & v_xD_w_d),  // 두 입력 동시 준비
    .group_sum_i(y_tile_w),
    .xD_i       (xD_w_d),
    .y_o        (y_sum_hp),
    .valid_o    (v_y_sum_hp)
    );

    assign y_final_o       = y_sum_hp;
    assign y_final_valid_o = v_y_sum_hp;

    // // xD 래치 클리어
    // always @(posedge clk or negedge rstn) begin
    // if (!rstn) xD_latched_v <= 1'b0;
    // else if (v_y_sum_hp) xD_latched_v <= 1'b0;
    // end


    // ============================================================================
    // Debug slice wires for waveform readability (simulation-only)
    // - Verilog-2001 only (no net arrays outside generate scopes)
    // - Exposes hierarchical wires per (h), (h,p), (h,p,n)
    // - View in wave: dut.DBG_<name>.G_H[...].G_P[...].G_N[...].<signal>
    // ============================================================================
    `ifdef SIM   // or: `ifndef SYNTHESIS
    // ---------------------- (h) vectors ----------------------
    genvar __h0;
    generate
      for (__h0 = 0; __h0 < H_TILE; __h0 = __h0 + 1) begin: DBG_h_scalars
        // delta, softplus(delta), dA_tmp, dA
        wire [DW-1:0] delta_h      = delta_w     [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] delta_sp_h   = delta_sp_w  [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dA_tmp_h     = dA_tmp_w    [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dA_h         = dA_w        [DW*(__h0+1)-1 -: DW];
    
        // 입력 스칼라 쪽도 보고 싶다면
        wire [DW-1:0] dt_h         = dt_i        [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dt_bias_h    = dt_bias_i   [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] A_h          = A_i         [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] D_h          = D_i         [DW*(__h0+1)-1 -: DW];
      end
    endgenerate
    
    // ---------------------- (h,p) vectors ----------------------
    genvar __h1, __p1;
    generate
      for (__h1 = 0; __h1 < H_TILE; __h1 = __h1 + 1) begin: DBG_hp_vecs
        for (__p1 = 0; __p1 < P_TILE; __p1 = __p1 + 1) begin: G_P
          localparam integer __HP = __h1*P_TILE + __p1;
    
          // dx, xD, x, 최종 y 타일/최종 y
          wire [DW-1:0] dx_hp        = dx_w          [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] x_hp         = x_i           [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] xD_hp        = xD_w          [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] xD_latched_hp= xD_latched_r  [DW*(__HP+1)-1 -: DW];
    
          wire [DW-1:0] y_tile_hp    = y_tile_w      [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] y_sum_hp_w   = y_sum_hp      [DW*(__HP+1)-1 -: DW]; // 최종 adder 출력
    
          // 중간 hC 타일까지 보고 싶으면 아래 4개(H*P*N 슬라이스) 대신 hp 한 슬라이스만:
          // wire [DW-1:0] hC_hp       = hC_w[DW*((__HP*N_TILE)+0 + 1)-1 -: DW]; // n=0 예시
        end
      end
    endgenerate
    
    // ---------------------- (h,p,n) tensors ----------------------
    // dBx, dAh, hnext, hC : (h,p,n) 전체를 모두 파형에서 계층적으로 볼 수 있음
    genvar __h2, __p2, __n2;
    generate
      for (__h2 = 0; __h2 < H_TILE; __h2 = __h2 + 1) begin: DBG_dBx
        for (__p2 = 0; __p2 < P_TILE; __p2 = __p2 + 1) begin: G_P
          for (__n2 = 0; __n2 < N_TILE; __n2 = __n2 + 1) begin: G_N
            localparam integer __HP_dBx  = __h2*P_TILE + __p2;
            localparam integer __HPN_dBx = __HP_dBx*N_TILE + __n2;
            wire [DW-1:0] dBx_hpn = dBx_w[DW*(__HPN_dBx+1)-1 -: DW];
          end
        end
      end
    endgenerate
    
    genvar __h3, __p3, __n3;
    generate
      for (__h3 = 0; __h3 < H_TILE; __h3 = __h3 + 1) begin: DBG_dAh
        for (__p3 = 0; __p3 < P_TILE; __p3 = __p3 + 1) begin: G_P
          for (__n3 = 0; __n3 < N_TILE; __n3 = __n3 + 1) begin: G_N
            localparam integer __HP_dAh  = __h3*P_TILE + __p3;
            localparam integer __HPN_dAh = __HP_dAh*N_TILE + __n3;
            wire [DW-1:0] dAh_hpn = dAh_w[DW*(__HPN_dAh+1)-1 -: DW];
          end
        end
      end
    endgenerate
    
    genvar __h4, __p4, __n4;
    generate
      for (__h4 = 0; __h4 < H_TILE; __h4 = __h4 + 1) begin: DBG_hnext
        for (__p4 = 0; __p4 < P_TILE; __p4 = __p4 + 1) begin: G_P
          for (__n4 = 0; __n4 < N_TILE; __n4 = __n4 + 1) begin: G_N
            localparam integer __HP_hn  = __h4*P_TILE + __p4;
            localparam integer __HPN_hn = __HP_hn*N_TILE + __n4;
            wire [DW-1:0] hnext_hpn = hnext_w[DW*(__HPN_hn+1)-1 -: DW];
          end
        end
      end
    endgenerate
    
    genvar __h5, __p5, __n5;
    generate
      for (__h5 = 0; __h5 < H_TILE; __h5 = __h5 + 1) begin: DBG_hC
        for (__p5 = 0; __p5 < P_TILE; __p5 = __p5 + 1) begin: G_P
          for (__n5 = 0; __n5 < N_TILE; __n5 = __n5 + 1) begin: G_N
            localparam integer __HP_hC  = __h5*P_TILE + __p5;
            localparam integer __HPN_hC = __HP_hC*N_TILE + __n5;
            wire [DW-1:0] hC_hpn = hC_w[DW*(__HPN_hC+1)-1 -: DW];
          end
        end
      end
    endgenerate
    
    // ---------------------- (optional) hC_buf tiles ----------------------
    // 타일 8장을 모으는 버퍼 내용을 타일/인덱스별로 보고 싶을 때
    genvar __t0, __n6;
    generate
      for (__t0 = 0; __t0 < TILES_PER_GROUP; __t0 = __t0 + 1) begin: DBG_hCbuf
        for (__n6 = 0; __n6 < N_TILE; __n6 = __n6 + 1) begin: G_N
          wire [DW-1:0] hCbuf_tile_n = hC_buf[__t0][DW*(__n6+1)-1 -: DW];
        end
      end
    endgenerate
    
    // ---------------------- handy valids ----------------------
    wire DBG_group_start    = group_start;
    wire DBG_group_last     = group_last;
    wire DBG_group_start_d  = group_s_d;
    wire DBG_group_last_d   = group_l_d;
    wire DBG_v_delta        = v_delta_w;
    wire DBG_v_delta_sp     = v_delta_sp_w;
    wire DBG_v_dA_tmp       = v_dA_tmp_w;
    wire DBG_v_dA           = v_dA_w;
    wire DBG_v_dx           = v_dx_w;
    wire DBG_v_dBx          = v_dBx_w;
    wire DBG_v_dAh          = v_dAh_w;
    wire DBG_v_hnext        = v_hnext_w;
    wire DBG_v_hC           = v_hC_w;
    wire DBG_grp_emit       = grp_emit;
    wire DBG_v_y_tile       = v_y_tile_w;
    wire DBG_xD_latched_v   = xD_latched_v;
    wire DBG_v_y_final      = y_final_valid_o;
    `endif
    
endmodule
