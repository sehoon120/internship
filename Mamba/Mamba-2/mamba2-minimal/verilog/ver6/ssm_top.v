// ================================================================
// SSMBLOCK_TOP — tile-in → tile-out(hC), + Acc across tiles, + y_out (= y_tmp + xD)
// Assumptions:
//  - B=H=P=1 (스칼라 3개: dt, dA, x; 스칼라 1개: D; 타일 벡터 3개: B_tile, C_tile, hprev_tile)
//  - 매 클록 1타일(N_TILE) 입력 가능 (II=1), 내부 IP도 throughput=1 필요
//  - tile_last_i=1 이 마지막 타일임을 의미 (TB에서 제공)
//  - 하위 모듈: dx, dBx, dAh, h_next, hC, fp16_mult_wrapper, fp16_add_wrapper 존재 가정
// ================================================================
module SSMBLOCK_TOP #(
    parameter integer DW        = 16,
    parameter integer N_TILE    = 16,
    // Latency params (IP 설정에 맞춰 조정)
    parameter integer LAT_DX_M  = 6,  // dx: dt*x (mul)
    parameter integer LAT_DBX_M = 6,  // dBx: dx*B (mul)
    parameter integer LAT_DAH_M = 6,  // dAh: dA*hprev (mul)
    parameter integer LAT_ADD_A = 11,  // h_next: add
    parameter integer LAT_HC_M  = 6,  // hC: h_next*C (mul)
    parameter integer LAT_ACC_A = 11   // adder latency used inside Acc (fp16 add)
)(
    input  wire                   clk,
    input  wire                   rstn,

    // 타일 유효/마지막 표시 (TB에서 분배/종료 관리)
    input  wire                   tile_valid_i,
    input  wire                   tile_last_i,
    output wire                   tile_ready_o,   // 필요시 backpressure 연결, 기본 항상 1

    // Scalars
    input  wire [DW-1:0]          dt_i,
    input  wire [DW-1:0]          dA_i,
    input  wire [DW-1:0]          x_i,
    input  wire [DW-1:0]          D_i,

    // Tile vectors (N_TILE)
    input  wire [N_TILE*DW-1:0]   B_tile_i,
    input  wire [N_TILE*DW-1:0]   C_tile_i,
    input  wire [N_TILE*DW-1:0]   hprev_tile_i,

    // Tile output: hC[n] (중간값 관찰용)
    output wire [N_TILE*DW-1:0]   hC_tile_o,
    output wire                   hC_tile_valid_o,

    // 최종 출력: y = sum_over_all_tiles(hC) + x*D  (프레임당 1회 펄스)
    output wire [DW-1:0]          y_final_o,
    output wire                   y_final_valid_o
);

    // ---------------------------------------------
    // 1) dx = dt * x   (scalar)
    // ---------------------------------------------
    wire [DW-1:0] dx_w;
    wire          v_dx;
    dx #(.DW(DW), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i), // 처음 타일 들어오는 시점에 발사(지속 입력도 OK)
        .dt_i    (dt_i),
        .x_i     (x_i),
        .dx_o    (dx_w),
        .valid_o (v_dx)
    );

    // ---------------------------------------------
    // 2) dBx = dx * B[n] (N_TILE 병렬)
    // ---------------------------------------------
    wire [N_TILE*DW-1:0] dBx_w;
    wire                 v_dBx;
    dBx #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx),
        .dx_i    (dx_w),
        .Bmat_i  (B_tile_i),  // B LAT_M 만큼 delay
        .dBx_o   (dBx_w),
        .valid_o (v_dBx)
    );

    // ---------------------------------------------
    // 3) dAh = dA * hprev[n] (N_TILE 병렬)
    //    dBx 경로와 정렬: (LAT_DX_M + LAT_DBX_M - LAT_DAH_M)
    // ---------------------------------------------
    wire [N_TILE*DW-1:0] dAh_raw_w, dAh_w;
    wire                 v_dAh_raw,  v_dAh;

    dAh #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DAH_M)) u_dAh (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (tile_valid_i),
        .dA_i     (dA_i),
        .hprev_i  (hprev_tile_i),
        .dAh_o    (dAh_raw_w),  // dBx 끝날때까지 결과 delay -> pipe_bus
        .valid_o  (v_dAh_raw)
    );

    localparam integer DLY_DAH_ALIGN = (LAT_DX_M + LAT_DBX_M) - LAT_DAH_M;
    wire [N_TILE*DW-1:0] dAh_dly_w;
    wire                 v_dAh_dly;
    pipe_bus #(.W(N_TILE*DW), .D((DLY_DAH_ALIGN>0)?DLY_DAH_ALIGN:0)) u_dly_dAh_bus (
        .clk   (clk), .rstn(rstn),
        .din   (dAh_raw_w), .vin(v_dAh_raw),
        .dout  (dAh_dly_w), .vout(v_dAh_dly)
    );
    assign dAh_w = (DLY_DAH_ALIGN>0) ? dAh_dly_w : dAh_raw_w;
    assign v_dAh = (DLY_DAH_ALIGN>0) ? v_dAh_dly : v_dAh_raw;

    // ---------------------------------------------
    // 4) h_next = dBx + dAh (lane-wise)
    // ---------------------------------------------
    wire [N_TILE*DW-1:0] hnext_w;
    wire                 v_hnext;
    h_next #(.DW(DW), .N_TILE(N_TILE), .ADD_LAT(LAT_ADD_A)) u_hnext (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (v_dBx & v_dAh),
        .dBx_i    (dBx_w),
        .dAh_i    (dAh_w),
        .hnext_o  (hnext_w),
        .valid_o  (v_hnext)
    );

    // ---------------------------------------------
    // 5) hC = h_next * C[n] (lane-wise) → 타일 출력
    // ---------------------------------------------
    wire                 v_hC;
    hC #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_HC_M)) u_hC (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (v_hnext),
        .hnext_i  (hnext_w),
        .C_i      (C_tile_i),  // 앞쪽 지연만큼 delay
        .hC_o     (hC_tile_o),
        .valid_o  (v_hC)
    );
    assign hC_tile_valid_o = v_hC;

    // ---------------------------------------------
    // 6) 타일 합산(Σ lanes) → tile_sum (DW)
    //    16 → 8 → 4 → 2 → 1 의 tree, 각 단계 fp16_add LAT_ACC_A
    // ---------------------------------------------
    wire [DW-1:0] tile_sum_w;
    wire          tile_sum_v;
    AccTileSum16 #(.DW(DW), .ADD_LAT(LAT_ACC_A)) u_tile_sum (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_hC),
        .hC_i    (hC_tile_o),
        .sum_o   (tile_sum_w),
        .valid_o (tile_sum_v)
    );

    // ---------------------------------------------
    // 7) 전체 타일 누적(Σ over tiles) — 스트리밍 파이프 누산기
    //    clear는 "프레임 시작"에서 TB가 tile_last_i 이전 사이클에 넣어주는 식으로 구현,
    //    여기서는 간단히 '첫 유효 타일'에서 내부 초기화.
    // ---------------------------------------------
    wire [DW-1:0] y_tmp_w;
    wire          y_tmp_v;
    wire          last_sum_v;  // 마지막 타일의 누적 결과가 나오는 싸이클 표시

    AccTilesStream #(.DW(DW), .ADD_LAT(LAT_ACC_A)) u_acc_stream (
        .clk        (clk),
        .rstn       (rstn),
        .sum_i      (tile_sum_w),
        .sum_valid_i(tile_sum_v),
        .last_i     (tile_sum_v & tile_last_i),  // 마지막 타일 sum에 태깅
        .y_tmp_o    (y_tmp_w),                   // running sum (파이프 딜레이 포함)
        .y_tmp_valid_o(y_tmp_v),
        .last_o     (last_sum_v)                 // y_tmp_o가 "최종"일 때 1
    );

    // ---------------------------------------------
    // 8) xD = x * D  (미리 계산해 보관)
    // ---------------------------------------------
    wire [DW-1:0] xD_w;
    wire          v_xD;
    xD_scalar #(.DW(DW)) u_xD (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i), // 첫 타일에서 1이면 충분(연속 타일도 OK)
        .x_i     (x_i),
        .D_i     (D_i),
        .xD_o    (xD_w),  // 결과 딜레이
        .valid_o (v_xD)
    );

    // xD 보관 (최종 합산 때까지 유지)
    reg  [DW-1:0] xD_hold;
    reg           xD_hold_v;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            xD_hold   <= {DW{1'b0}};
            xD_hold_v <= 1'b0;
        end else if (v_xD) begin
            xD_hold   = xD_w;
            xD_hold_v = 1'b1;
        end
        // 프레임 경계 관리는 TB에서 tile_last_i로 하므로 여기선 유지
    end

    // ---------------------------------------------
    // 9) 최종 y = y_tmp + xD (마지막 타일 누적이 나온 싸이클에 트리거)
    // ---------------------------------------------
    wire [DW-1:0] y_final_w;
    wire          v_y_final_w;

    fp16_add_wrapper u_add_yfinal (
        .clk       (clk),
        .valid_in  (last_sum_v & y_tmp_v & xD_hold_v),
        .a         (y_tmp_w),
        .b         (xD_hold),
        .result    (y_final_w),
        .valid_out (v_y_final_w)
    );

    assign y_final_o        = y_final_w;
    assign y_final_valid_o  = v_y_final_w;

    // 타일 입력 항상 수락 (필요시 backpressure로 교체)
    assign tile_ready_o = 1'b1;

endmodule


// ------------------------------------------------------------
// 16-lane tile sum (adder tree)
// ------------------------------------------------------------
module AccTileSum16 #(
  parameter integer DW = 16,
  parameter integer ADD_LAT = 7
)(
  input  wire               clk,
  input  wire               rstn,
  input  wire               valid_i,
  input  wire [16*DW-1:0]   hC_i,
  output wire [DW-1:0]      sum_o,
  output wire               valid_o
);
  // 16→8
  wire [DW-1:0] s1 [0:7];
  genvar i;
  generate
    for (i=0;i<8;i=i+1) begin: L1
      fp16_add_wrapper u_add1 (
        .clk(clk), .valid_in(valid_i),
        .a(hC_i[(2*i+0)*DW +: DW]),
        .b(hC_i[(2*i+1)*DW +: DW]),
        .result(s1[i]), .valid_out(/*unused*/)
      );
    end
  endgenerate
  wire v1 = valid_i;

  // 8→4
  wire [DW-1:0] s2 [0:3];
  generate
    for (i=0;i<4;i=i+1) begin: L2
      fp16_add_wrapper u_add2 (
        .clk(clk), .valid_in(v1),
        .a(s1[2*i+0]), .b(s1[2*i+1]),
        .result(s2[i]), .valid_out()
      );
    end
  endgenerate
  wire v2 = v1;

  // 4→2
  wire [DW-1:0] s3 [0:1];
  generate
    for (i=0;i<2;i=i+1) begin: L3
      fp16_add_wrapper u_add3 (
        .clk(clk), .valid_in(v2),
        .a(s2[2*i+0]), .b(s2[2*i+1]),
        .result(s3[i]), .valid_out()
      );
    end
  endgenerate
  wire v3 = v2;

  // 2→1
  fp16_add_wrapper u_add4 (
    .clk(clk), .valid_in(v3),
    .a(s3[0]), .b(s3[1]),
    .result(sum_o), .valid_out(valid_o)
  );
endmodule


// ------------------------------------------------------------
// Streaming accumulator across tiles with pipelined FP adder
//  - Throughput 1 유지: 매 클록 tile_sum + (과거 누적값) 을 더함
//  - ADD_LAT 단계 파이프라인 피드백 (지연선) 사용
// ------------------------------------------------------------
module AccTilesStream #(
  parameter integer DW = 16,
  parameter integer ADD_LAT = 7
)(
  input  wire         clk,
  input  wire         rstn,
  input  wire [DW-1:0] sum_i,
  input  wire         sum_valid_i,
  input  wire         last_i,        // 이 sum_i가 마지막 타일임을 표시
  output wire [DW-1:0] y_tmp_o,      // 누적 합(파이프 지연 포함)
  output wire         y_tmp_valid_o,
  output wire         last_o         // 위 출력이 최종임을 표시
);
  // 누적값 파이프 (지연선)
  reg  [DW-1:0] acc_pipe [0:ADD_LAT-1];
  reg           vld_pipe [0:ADD_LAT-1];
  reg           last_pipe[0:ADD_LAT-1];

  // 피드백: 가장 오래된 누적값 + 현재 sum_i
  wire [DW-1:0] adder_out;
  wire          adder_vout;
  fp16_add_wrapper u_add_acc (
    .clk       (clk),
    .valid_in  (sum_valid_i),
    .a         (acc_pipe[ADD_LAT-1]),  // c-ADD_LAT 시점 누적값
    .b         (sum_i),
    .result    (adder_out),
    .valid_out (adder_vout)
  );

  integer k;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      for (k=0;k<ADD_LAT;k=k+1) begin
        acc_pipe[k]  <= {DW{1'b0}};
        vld_pipe[k]  <= 1'b0;
        last_pipe[k] <= 1'b0;
      end
    end else begin
      // 파이프 시프트 (유효할 때만 진행)
      if (adder_vout) begin
        // 최신 누적 결과를 pipe[0]에 적재
        acc_pipe[0]  <= adder_out;
        vld_pipe[0]  <= 1'b1;
        last_pipe[0] <= last_pipe[ADD_LAT-1]; // last 플래그도 동일 지연으로 전달

        for (k=ADD_LAT-1; k>0; k=k-1) begin
          acc_pipe[k]  <= acc_pipe[k-1];
          vld_pipe[k]  <= vld_pipe[k-1];
          last_pipe[k] <= last_pipe[k-1];
        end
      end

      // last_i를 adder 입력 경로의 지연에 맞춰 전달
      if (sum_valid_i) begin
        // sum_i가 adder에 들어가는 사이클에 last_i를 ADD_LAT 지연 뒤로 보냄
        // 여기서는 간단히 last_pipe의 꼬리를 덮어쓰기 위한 준비로,
        // adder_vout 시점에 위에서 last_pipe[0]에 acc_pipe[ADD_LAT-1]의 last를 복사한다.
      end
    end
  end

  // sum_valid_i가 들어오는 사이클에 last_i를 ADD_LAT 후에 맞춰주기 위한 쉬프트 레지스터
  reg [ADD_LAT-1:0] last_sr;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) last_sr <= 'b0;
    else begin
      last_sr <= {last_sr[ADD_LAT-2:0], (sum_valid_i ? last_i : 1'b0)};
      // ADD_LAT 뒤에 last_sr[ADD_LAT-1] == adder_vout에 해당
      if (adder_vout) begin
        // 위 파이프 갱신과 동시에 꼬리에 last 플래그 주입
        last_pipe[0] <= last_sr[ADD_LAT-1];
      end
    end
  end

  assign y_tmp_o       = acc_pipe[ADD_LAT-1];
  assign y_tmp_valid_o = vld_pipe[ADD_LAT-1];
  assign last_o        = last_pipe[ADD_LAT-1];

endmodule


// ------------------------------------------------------------
// xD = x * D (scalar) — 간단 래퍼
// ------------------------------------------------------------
module xD_scalar #(
  parameter integer DW = 16
)(
  input  wire         clk,
  input  wire         rstn,
  input  wire         valid_i,
  input  wire [DW-1:0] x_i,
  input  wire [DW-1:0] D_i,
  output wire [DW-1:0] xD_o,
  output wire         valid_o
);
  fp16_mult_wrapper u_mul (
    .clk       (clk),
    .valid_in  (valid_i),
    .a         (x_i),
    .b         (D_i),
    .result    (xD_o),
    .valid_out (valid_o)
  );
endmodule


// ------------------------------------------------------------
// Data+valid pipeline utility
// ------------------------------------------------------------
module pipe_bus #(
    parameter integer W = 16,
    parameter integer D = 0
)(
    input  wire             clk,
    input  wire             rstn,
    input  wire [W-1:0]     din,
    input  wire             vin,
    output wire [W-1:0]     dout,
    output wire             vout
);
    generate
        if (D == 0) begin : G_D0
            assign dout = din;
            assign vout = vin;
        end else begin : G_DN
            reg [W-1:0] q   [0:D-1];
            reg         qv  [0:D-1];
            integer i;
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    for (i=0;i<D;i=i+1) begin
                        q[i]  <= {W{1'b0}};
                        qv[i] <= 1'b0;
                    end
                end else begin
                    q [0] <= din;  qv[0] <= vin;
                    for (i=1;i<D;i=i+1) begin
                        q [i] <= q [i-1];
                        qv[i] <= qv[i-1];
                    end
                end
            end
            assign dout = q [D-1];
            assign vout = qv[D-1];
        end
    endgenerate
endmodule
