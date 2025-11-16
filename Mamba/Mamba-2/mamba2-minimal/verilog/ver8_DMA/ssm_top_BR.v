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
//`define SIM 

// exp, sp lat -> +2를 +1로 되돌리기
// latched valid 그룹 끝마다 0되는거 없애기.
// 다른 신호들 왜 밀리는지 보기

// ----------------------------------------------------------------------------
// 보호 매크로 / 경계 래치 스테이지
// ----------------------------------------------------------------------------
`define PROTECT   (* DONT_TOUCH="true", KEEP="true" *)
`define NOSRL     (* shreg_extract="no" *)

module br_stage #(
  parameter integer W = 32,
  parameter integer USE_V = 1,
  parameter integer ENABLE = 1   // 1: 1-cycle 래치, 0: bypass
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire [W-1:0]         din,
  input  wire                 vin,     // USE_V=0이면 무시
  output wire [W-1:0]         dout,
  output wire                 vout
);
  generate
    if (ENABLE) begin : G_BR
      `PROTECT reg [W-1:0] r_d;
      `PROTECT reg         r_v;
      always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
          r_d <= {W{1'b0}};
          r_v <= 1'b0;
        end else begin
          r_d <= din;
          r_v <= (USE_V ? vin : 1'b1);
        end
      end
      assign dout = r_d;
      assign vout = (USE_V ? r_v : 1'b1);
    end else begin : G_FWD
      assign dout = din;
      assign vout = (USE_V ? vin : 1'b1);
    end
  endgenerate
endmodule

module SSMBLOCK_TOP_BR #(
    parameter integer DW          = 16,
    parameter integer H_TILE      = 1,
    parameter integer P_TILE      = 1,
    parameter integer N_TILE      = 128,
    parameter integer P_TOTAL     = 64,
    parameter integer N_TOTAL     = 128,

    // 비교 스위치
    parameter USE_FWD        = 0,   // 1: 제안안(직결), 0: Baseline-BR(경계 1-cycle 래치)
    parameter USE_BRAM_DELAY = 1,   // 1: BRAM 딜레이(제안안), 0: SRL/레지스터(베이스라인)
 
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
    parameter integer LAT_EXP     = 6 + LAT_MUL * 3 + LAT_ADD * 3 + 1+1+1,     // exp latency (예시)
    parameter integer LAT_SP      = LAT_EXP + LAT_MUL + LAT_ADD + LAT_DIV + 1 //+ 2  // latching delay   // Softplus latency (예시)
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
    localparam integer GROUPS_PER_H = (P_TOTAL + P_TILE - 1) / P_TILE;  // h당 (h,p) 그룹 수

    // 항상 수신 가능 (현재 설계에선 backpressure 없음)
    assign tile_ready_o = 1'b1;

    // ============================================================
    // 0) delta = dt + dt_bias (h)  -- add
    // ============================================================
    wire [H_TILE*DW-1:0] delta_w_raw;
    wire                 v_delta_w_raw;

    delta #(.DW(DW), .H_TILE(H_TILE), .A_LAT(LAT_ADD_A)) u_delta (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i),
        .dt_i    (dt_i),
        .bias_i  (dt_bias_i),
        .sum_o   (delta_w_raw),
        .valid_o (v_delta_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*DW-1:0] delta_w;
    wire                 v_delta_w;
    br_stage #(.W(H_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_delta (
        .clk(clk), .rstn(rstn),
        .din(delta_w_raw), .vin(v_delta_w_raw),
        .dout(delta_w), .vout(v_delta_w)
    );

    // ============================================================
    // 0') xD = x * D (hp) -- mul  (그룹당 1회 래치)
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] xD_w_raw;
    wire                        v_xD_w_raw;

    xD_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .M_LAT(LAT_DX_M)) u_xD (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i),
        .x_i     (x_i),   // (h*p)
        .D_i     (D_i),   // (h)
        .xD_o    (xD_w_raw),  // (h*p)
        .valid_o (v_xD_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*DW-1:0] xD_w;
    wire                        v_xD_w;
    br_stage #(.W(H_TILE*P_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_xd (
        .clk(clk), .rstn(rstn),
        .din(xD_w_raw), .vin(v_xD_w_raw),
        .dout(xD_w), .vout(v_xD_w)
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
    
    // H group
    reg [$clog2(GROUPS_PER_H):0] hp_group_cnt;
    wire h_start_raw = (hp_group_cnt == 0) && group_start;
    wire h_last_raw  = (hp_group_cnt == GROUPS_PER_H-1) && group_last;
    
    always @(posedge clk or negedge rstn) begin
      if (!rstn) begin
        hp_group_cnt <= 0;
      end else if (tile_valid_i && group_last) begin
        if (hp_group_cnt == GROUPS_PER_H-1)
          hp_group_cnt <= 0;       // 다음 h로 넘어가는 시점
        else
          hp_group_cnt <= hp_group_cnt + 1'b1;
      end
    end

    // softplus(delta) mode select
    wire hstart_to_mode_d, hlast_to_mode_d;
    shift_reg #(.DW(2), .DEPTH(LAT_ADD_A)) u_h_mode_align (  // 
      .clk(clk), .rstn(rstn),
      .din({h_start_raw, h_last_raw}),
      .dout({hstart_to_mode_d, hlast_to_mode_d})
    );
    
    wire hstart_to_sp_d, hlast_to_sp_d;
    shift_reg #(.DW(2), .DEPTH(LAT_SP)) u_h_sp_align (  // 
      .clk(clk), .rstn(rstn),
      .din({hstart_to_mode_d, hlast_to_mode_d}),
      .dout({hstart_to_sp_d, hlast_to_sp_d})
    );
    
    // exp mode select
    wire hstart_to_exp_mode_d, hlast_to_exp_mode_d;
    shift_reg #(.DW(2), .DEPTH(LAT_DX_M+ 1)) u_h_exp_mode_align (  //  
      .clk(clk), .rstn(rstn),
      .din({hstart_to_sp_d, hlast_to_sp_d}),
      .dout({hstart_to_exp_mode_d, hlast_to_exp_mode_d})
    );
    
    // exp 결과(dA) 래치 타이밍 맞춤
    wire hstart_to_exp_d, hlast_to_exp_d;
    shift_reg #(.DW(2), .DEPTH(LAT_EXP+LAT_MUL)) u_h_exp_align (  //  
      .clk(clk), .rstn(rstn),
      .din({hstart_to_exp_mode_d, hlast_to_exp_mode_d}),
      .dout({hstart_to_exp_d, hlast_to_exp_d})
    );

    // xD 값은 그룹의 첫 타일 타이밍에서 유효해지는 v_xD_w를 감지해 래치
    // (x, D가 그룹 내에서 불변이라는 가정)
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            xD_latched_r <= {H_TILE*P_TILE*DW{1'b0}};
            xD_latched_v <= 1'b0;
        end else begin
            if (group_s_d && v_xD_w) begin
                xD_latched_r <= xD_w;
                xD_latched_v <= 1'b1;
            end
            else if (group_l_d) begin
                xD_latched_v <= 1'b0;
            end
        end
    end

    // xD 딜레이선: BRAM vs SRL 분기
    wire [H_TILE*P_TILE*DW-1:0] xD_w_d;
    wire                        v_xD_w_d;
    generate
      if (USE_BRAM_DELAY) begin : G_XD_DLY_BRAM
        pipe_bus_bram #(
            .W(H_TILE*P_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP+1+5 + LAT_DAH_M + 2 + LAT_DX_M +  LAT_ACCU + N_TOTAL/N_TILE + 3 + 1),
            .USE_V(1)
        ) u_xd_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(xD_latched_r), .vin(xD_latched_v),
            .dout(xD_w_d), .vout(v_xD_w_d)
        );
      end else begin : G_XD_DLY_SRL
        // valid는 간단히 동일 딜레이로 시프트(폭1)
        shift_reg #(.DW(H_TILE*P_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+1+5 + LAT_DAH_M+1 + 2 + LAT_DX_M+1 +  LAT_ACCU+1 + N_TOTAL/N_TILE + 3 + 1))
          u_xd_delay_srl (.clk(clk), .rstn(rstn), .din(xD_latched_r), .dout(xD_w_d));
        shift_reg #(.DW(1), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+1+5 + LAT_DAH_M+1 + 2 + LAT_DX_M+1 +  LAT_ACCU+1 + N_TOTAL/N_TILE + 3 + 1))
          u_xd_v_delay_srl (.clk(clk), .rstn(rstn), .din(v_xD_w), .dout(v_xD_w_d));
      end
    endgenerate

    // ============================================================
    // 1) delta_sp = Softplus(delta) (h)  [공유 코어: exp_sp_shared]
    // ============================================================
    wire [H_TILE*DW-1:0] shared_sp_y, shared_exp_y;
    wire                 v_shared_sp, v_shared_exp;
    wire [H_TILE*DW-1:0] dA_tmp_w_pre;
    wire                 v_dA_tmp_w_pre;
    wire                 mode_softplus_o;

    exp_sp_shared #(
      .DW(DW),
      .LAT_EXP(LAT_EXP),
      .LAT_MUL(LAT_MUL), .LAT_ADD(LAT_ADD), .LAT_DIV(LAT_DIV)
    ) u_shared (
      .clk(clk), .rstn(rstn),

      // SP 요청: delta → softplus
      .sp_req_v (v_delta_w),
      .sp_x     (delta_w),
      .v_sp     (hstart_to_mode_d),
      .sp_rsp_v (v_shared_sp),
      .sp_y     (shared_sp_y),
      
      // EXP 요청: dA_tmp → exp
      .exp_req_v (v_dA_tmp_w_pre),
      .exp_x     (dA_tmp_w_pre),
      .v_exp     (hstart_to_exp_mode_d),
      .exp_rsp_v (v_shared_exp),
      .exp_y     (shared_exp_y),
      .mode_softplus_o (mode_softplus_o)
    );

    // [경계 래치 토글: softplus 결과]
    wire [H_TILE*DW-1:0] delta_sp_w;
    wire                 v_delta_sp_w;
    br_stage #(.W(H_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_sp (
      .clk(clk), .rstn(rstn),
      .din(shared_sp_y), .vin(v_shared_sp),
      .dout(delta_sp_w), .vout(v_delta_sp_w)
    );

    // [경계 래치 토글: exp 결과]
    wire [H_TILE*DW-1:0] dA_w;
    wire                 v_dA_w;
    br_stage #(.W(H_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_exp (
      .clk(clk), .rstn(rstn),
      .din(shared_exp_y), .vin(v_shared_exp),
      .dout(dA_w), .vout(v_dA_w)
    );

    // 필요 시점만 래치
    reg  [H_TILE*DW-1:0] delta_sp_latched_r;
    reg                  delta_sp_latched_v;
    always @(posedge clk or negedge rstn) begin
      if (!rstn) begin
        delta_sp_latched_r <= {H_TILE*DW{1'b0}};
        delta_sp_latched_v <= 1'b0;
      end else begin
        if (hstart_to_sp_d && v_delta_sp_w) begin
          delta_sp_latched_r <= delta_sp_w;
          delta_sp_latched_v <= 1'b1;
        end
      end
    end

    // ============================================================
    // 2) dA_tmp = delta_sp * A (h) -- mul
    // ============================================================
    wire [H_TILE*DW-1:0] A_i_d;
    generate
      if (USE_BRAM_DELAY) begin : G_A_DLY_BRAM
        pipe_bus_bram #(
            .W(H_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + 1),
            .USE_V(0)
        ) u_zero_flag_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(A_i), .vin(1'b0),
            .dout(A_i_d), .vout()
        );
      end else begin : G_A_DLY_SRL
        shift_reg #(.DW(H_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + 1))
          u_zero_flag_delay_srl (.clk(clk), .rstn(rstn), .din(A_i), .dout(A_i_d));
      end
    endgenerate

    wire [H_TILE*DW-1:0] dA_tmp_w_raw;
    wire                 v_dA_tmp_w_raw;

    dA_mul #(.DW(DW), .H_TILE(H_TILE), .M_LAT(LAT_DX_M)) u_dA_tmp (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (delta_sp_latched_v),
        .lhs_i   (delta_sp_latched_r), // (h)
        .rhs_i   (A_i_d),              // (h)
        .mul_o   (dA_tmp_w_raw),       // (h)
        .valid_o (v_dA_tmp_w_raw)
    );

    // 경계 래치 토글 (dA_tmp → shared exp 입력으로)
    br_stage #(.W(H_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_dA_tmp_to_shared (
      .clk(clk), .rstn(rstn),
      .din(dA_tmp_w_raw), .vin(v_dA_tmp_w_raw),
      .dout(dA_tmp_w_pre), .vout(v_dA_tmp_w_pre)
    );

    // exp 결과 필요 시점 래치
    reg  [H_TILE*DW-1:0] dA_latched_r;
    reg                  dA_latched_v;
    always @(posedge clk or negedge rstn) begin
      if (!rstn) begin
        dA_latched_r <= {H_TILE*DW{1'b0}};
        dA_latched_v <= 1'b0;
      end else begin
        if (hstart_to_exp_d && v_dA_w) begin
          dA_latched_r <= dA_w;
          dA_latched_v <= 1'b1;
        end 
      end
    end
    
    // ============================================================
    // 2') dx = delta_sp * x (hp) -- mul  (broadcast h → hp)
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] x_i_d;
    generate
      if (USE_BRAM_DELAY) begin : G_X_DLY_BRAM
        pipe_bus_bram #(
            .W(H_TILE*P_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + 1),
            .USE_V(0)
        ) u_x_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(x_i), .vin(1'b0),
            .dout(x_i_d), .vout()
        );
      end else begin : G_X_DLY_SRL
        shift_reg #(.DW(H_TILE*P_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + 1))
          u_x_delay_srl (.clk(clk), .rstn(rstn), .din(x_i), .dout(x_i_d));
      end
    endgenerate

    wire [H_TILE*P_TILE*DW-1:0] dx_w_raw;
    wire                        v_dx_w_raw;

    dx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (delta_sp_latched_v),
        .h_i     (delta_sp_latched_r), // (h)
        .x_i     (x_i_d),              // (h*p)
        .dx_o    (dx_w_raw),           // (h*p)
        .valid_o (v_dx_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*DW-1:0] dx_w;
    wire                        v_dx_w;
    br_stage #(.W(H_TILE*P_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_dx (
      .clk(clk), .rstn(rstn),
      .din(dx_w_raw), .vin(v_dx_w_raw),
      .dout(dx_w), .vout(v_dx_w)
    );

    // ============================================================
    // 3') dBx = dx * B_tile (hpn) -- mul (broadcast B[n] → hp)
    // ============================================================
    wire [N_TILE*DW-1:0] B_tile_i_d;
    generate
      if (USE_BRAM_DELAY) begin : G_B_DLY_BRAM
        pipe_bus_bram #(
            .W(N_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + LAT_DX_M + 1),
            .USE_V(0)
        ) u_B_w_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(B_tile_i), .vin(1'b0),
            .dout(B_tile_i_d), .vout()
        );
      end else begin : G_B_DLY_SRL
        shift_reg #(.DW(N_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + 1))
          u_B_w_delay_srl (.clk(clk), .rstn(rstn), .din(B_tile_i), .dout(B_tile_i_d));
      end
    endgenerate

    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w_raw;
    wire                               v_dBx_w_raw;

    dBx_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx_w),
        .dx_i    (dx_w),         // (h*p)
        .B_tile_i(B_tile_i_d),   // (n)
        .dBx_o   (dBx_w_raw),    // (h*p*n)
        .valid_o (v_dBx_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w;
    wire                               v_dBx_w;
    br_stage #(.W(H_TILE*P_TILE*N_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_dbx (
      .clk(clk), .rstn(rstn),
      .din(dBx_w_raw), .vin(v_dBx_w_raw),
      .dout(dBx_w), .vout(v_dBx_w)
    );

    // ============================================================
    // 4) dAh = dA * hprev_tile (hpn) -- mul (broadcast dA[h] → p*n)
    // ============================================================
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hprev_tile_i_d;
    generate
      if (USE_BRAM_DELAY) begin : G_HPREV_DLY_BRAM
        pipe_bus_bram #(
            .W(H_TILE*P_TILE*N_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP+5+2 + 2),
            .USE_V(0)
        ) u_hprev_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(hprev_tile_i), .vin(1'b0),
            .dout(hprev_tile_i_d), .vout()
        );
      end else begin : G_HPREV_DLY_SRL
        shift_reg #(.DW(H_TILE*P_TILE*N_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+5+2 + 2))
          u_hprev_delay_srl (.clk(clk), .rstn(rstn), .din(hprev_tile_i), .dout(hprev_tile_i_d));
      end
    endgenerate
    
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dAh_w_raw;
    wire                               v_dAh_w_raw;

    dAh_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_DAH_M)) u_dAh (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (dA_latched_v),
        .dA_i      (dA_latched_r),     // (h)
        .hprev_i   (hprev_tile_i_d),   // (h*p*n)
        .dAh_o     (dAh_w_raw),        // (h*p*n)
        .valid_o   (v_dAh_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dAh_w;
    wire                               v_dAh_w;
    br_stage #(.W(H_TILE*P_TILE*N_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_dah (
      .clk(clk), .rstn(rstn),
      .din(dAh_w_raw), .vin(v_dAh_w_raw),
      .dout(dAh_w), .vout(v_dAh_w)
    );

    // ============================================================
    // 5) h_next = dAh + dBx (hpn) -- add
    // ============================================================
    // dBx aligned to dAh: BRAM vs SRL 분기
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] dBx_w_d;
    wire                               v_dBx_w_d;
    generate
      if (USE_BRAM_DELAY) begin : G_DBX_DLY_BRAM
        pipe_bus_bram #(
            .W(H_TILE*P_TILE*N_TILE*DW), 
            .D((LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP+2+5 + LAT_DAH_M + 1) - (LAT_ADD_A + LAT_SP + LAT_DX_M + 1 + LAT_DBX_M)),
            .USE_V(1)
        ) u_dbx_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(dBx_w), .vin(v_dBx_w),
            .dout(dBx_w_d), .vout(v_dBx_w_d)
        );
      end else begin : G_DBX_DLY_SRL
        shift_reg #(.DW(H_TILE*P_TILE*N_TILE*DW), .DEPTH((LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+2+5 + LAT_DAH_M+1 + 1) - (LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + 1 + LAT_DBX_M+1)))
          u_dbx_delay_srl (.clk(clk), .rstn(rstn), .din(dBx_w), .dout(dBx_w_d));
        shift_reg #(.DW(1), .DEPTH((LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+2+5 + LAT_DAH_M+1 + 1) - (LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + 1 + LAT_DBX_M+1)))
          u_dbx_v_delay_srl (.clk(clk), .rstn(rstn), .din(v_dBx_w), .dout(v_dBx_w_d));
      end
    endgenerate

    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hnext_w_raw;
    wire                               v_hnext_w_raw;

    hnext_add #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .A_LAT(LAT_ADD_A)) u_hnext (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dAh_w & v_dBx_w_d), // 두 경로 정렬 가정
        .dAh_i   (dAh_w),
        .dBx_i   (dBx_w_d),
        .sum_o   (hnext_w_raw),
        .valid_o (v_hnext_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hnext_w;
    wire                               v_hnext_w;
    br_stage #(.W(H_TILE*P_TILE*N_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_hn (
      .clk(clk), .rstn(rstn),
      .din(hnext_w_raw), .vin(v_hnext_w_raw),
      .dout(hnext_w), .vout(v_hnext_w)
    );

    // ============================================================
    // 6) hC = h_next * C_tile (hpn) -- mul (broadcast C[n] → h*p)
    // ============================================================
    wire [N_TILE*DW-1:0] C_tile_i_d;
    generate
      if (USE_BRAM_DELAY) begin : G_C_DLY_BRAM
        pipe_bus_bram #(
            .W(N_TILE*DW), 
            .D(LAT_ADD_A + LAT_SP + LAT_DX_M + LAT_EXP+2+5 + LAT_DAH_M + 2 + LAT_ADD_A),
            .USE_V(0)
        ) u_C_delay_bram (
            .clk(clk), .rstn(rstn),
            .din(C_tile_i), .vin(1'b0),
            .dout(C_tile_i_d), .vout()
        );
      end else begin : G_C_DLY_SRL
        shift_reg #(.DW(N_TILE*DW), .DEPTH(LAT_ADD_A+1 + LAT_SP+1 + LAT_DX_M+1 + LAT_EXP+1+2+5 + LAT_DAH_M+1 + 2 + LAT_ADD_A+1))
          u_C_delay_srl (.clk(clk), .rstn(rstn), .din(C_tile_i), .dout(C_tile_i_d));
      end
    endgenerate
    
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hC_w_raw;
    wire                               v_hC_w_raw;

    hC_mul #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .M_LAT(LAT_HC_M)) u_hC (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (v_hnext_w),
        .hnext_i   (hnext_w),
        .C_tile_i  (C_tile_i_d),
        .hC_o      (hC_w_raw),
        .valid_o   (v_hC_w_raw)
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*N_TILE*DW-1:0] hC_w;
    wire                               v_hC_w;
    br_stage #(.W(H_TILE*P_TILE*N_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_hC (
      .clk(clk), .rstn(rstn),
      .din(hC_w_raw), .vin(v_hC_w_raw),
      .dout(hC_w), .vout(v_hC_w)
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
    
    wire [H_TILE*P_TILE*DW-1:0] y_tile_w_raw;
    wire                        v_y_tile_w_raw;

    accum_n #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TOTAL(N_TOTAL)) u_accum_n (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (grp_emit),
        .hC_i      (hC_128_bus),      // (h*p*n)
        .sum_hp_o  (y_tile_w_raw),    // (h*p)
        .valid_o   (v_y_tile_w_raw)   // "타일" 단위 완료 펄스
    );

    // 경계 래치 토글
    wire [H_TILE*P_TILE*DW-1:0] y_tile_w;
    wire                        v_y_tile_w;
    br_stage #(.W(H_TILE*P_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_y_tile (
      .clk(clk), .rstn(rstn),
      .din(y_tile_w_raw), .vin(v_y_tile_w_raw),
      .dout(y_tile_w), .vout(v_y_tile_w)
    );
    
    // ============================================================
    // 8) 최종 합: group_sum_hp + xD_latched  → y_out
    // ============================================================
    wire [H_TILE*P_TILE*DW-1:0] y_sum_hp_raw;
    wire                        v_y_sum_hp_raw;

    y_out #(
      .DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .A_LAT(LAT_ADD_A)
    ) u_y_out (
      .clk        (clk),
      .rstn       (rstn),
      .valid_i    (v_y_tile_w & v_xD_w_d),  // 두 입력 동시 준비
      .group_sum_i(y_tile_w),
      .xD_i       (xD_w_d),
      .y_o        (y_sum_hp_raw),
      .valid_o    (v_y_sum_hp_raw)
    );

    // 경계 래치 토글 (최종)
    wire [H_TILE*P_TILE*DW-1:0] y_sum_hp;
    wire                        v_y_sum_hp;
    br_stage #(.W(H_TILE*P_TILE*DW), .USE_V(1), .ENABLE(!USE_FWD)) br_y_out (
      .clk(clk), .rstn(rstn),
      .din(y_sum_hp_raw), .vin(v_y_sum_hp_raw),
      .dout(y_sum_hp), .vout(v_y_sum_hp)
    );

    assign y_final_o       = y_sum_hp;
    assign y_final_valid_o = v_y_sum_hp;

    // ========================================================================
    // Debug slice wires (simulation-only)
    // ========================================================================
    `ifdef SIM
    genvar __h0;
    generate
      for (__h0 = 0; __h0 < H_TILE; __h0 = __h0 + 1) begin: DBG_h_scalars
        wire [DW-1:0] delta_h      = delta_w     [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] delta_sp_h   = delta_sp_w  [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dA_tmp_h     = dA_tmp_w_pre[DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dA_h         = dA_w        [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dt_h         = dt_i        [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] dt_bias_h    = dt_bias_i   [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] A_h          = A_i         [DW*(__h0+1)-1 -: DW];
        wire [DW-1:0] D_h          = D_i         [DW*(__h0+1)-1 -: DW];
      end
    endgenerate
    
    genvar __h1, __p1;
    generate
      for (__h1 = 0; __h1 < H_TILE; __h1 = __h1 + 1) begin: DBG_hp_vecs
        for (__p1 = 0; __p1 < P_TILE; __p1 = __p1 + 1) begin: G_P
          localparam integer __HP = __h1*P_TILE + __p1;
          wire [DW-1:0] dx_hp        = dx_w          [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] x_hp         = x_i           [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] xD_hp        = xD_w          [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] xD_latched_hp= xD_latched_r  [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] y_tile_hp    = y_tile_w      [DW*(__HP+1)-1 -: DW];
          wire [DW-1:0] y_sum_hp_w   = y_sum_hp      [DW*(__HP+1)-1 -: DW];
        end
      end
    endgenerate
    
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

    wire DBG_group_start    = group_start;
    wire DBG_group_last     = group_last;
    wire DBG_group_start_d  = group_s_d;
    wire DBG_group_last_d   = group_l_d;
    wire DBG_v_delta        = v_delta_w;
    wire DBG_v_delta_sp     = v_delta_sp_w;
    wire DBG_v_dA_tmp       = v_dA_tmp_w_pre;
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
