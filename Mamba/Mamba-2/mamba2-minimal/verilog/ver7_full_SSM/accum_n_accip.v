// ============================================================================
// accum_n_accip : Reduce over n with FP16 accumulator IP (streaming)
//   - 각 (h,p)마다 fp16_accum_wrapper 1개 사용
//   - 내부에서 n=0..N_TILE-1 순차로 데이터를 공급
//   - 출력 valid_o는 누산기 IP의 valid_out을 그대로 사용(마지막 입력 후)
// ----------------------------------------------------------------------------
// 가정: fp16_accum_wrapper 포트 예시
//   .clk, .rstn
//   .valid_in, .data_in (DW)
//   .last_in  (마지막 샘플 표시)
//   .sum_out (DW)
//   .valid_out (1펄스)
// ============================================================================
module accum_n_accip #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,     // 입력 타일 시작 트리거(1펄스)
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hC_i,        // (h*p*n) 병렬
  output wire [H_TILE*P_TILE*DW-1:0]         sum_hp_o,    // (h*p)
  output wire                                valid_o      // 모든 (h,p) 완료 "AND" 펄스
);

  // 입력 타일 래치 (큰 용량 → 실제론 BRAM 권장)
  reg [DW-1:0] hC_reg [0:H_TILE*P_TILE*N_TILE-1];
  integer li;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      // no-op
    end else if (valid_i) begin
      for (li = 0; li < H_TILE*P_TILE*N_TILE; li = li + 1)
        hC_reg[li] <= hC_i[DW*(li+1)-1 -: DW];
    end
  end

  // 스트리밍 제어 (모든 (h,p)가 같은 n_idx로 동시에 진행)
  localparam integer CNTW = (N_TILE > 1) ? $clog2(N_TILE) : 1;
  reg                  running_r;
  reg [CNTW-1:0]       n_idx_r;
  wire                 last_n = (n_idx_r == N_TILE-1);

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      running_r <= 1'b0;
      n_idx_r   <= {CNTW{1'b0}};
    end else begin
      if (valid_i && !running_r) begin
        running_r <= 1'b1;
        n_idx_r   <= {CNTW{1'b0}};
      end else if (running_r) begin
        if (!last_n) n_idx_r <= n_idx_r + 1'b1;
        else         running_r <= 1'b0;
      end
    end
  end

  // 각 (h,p) 누산기 IP
  wire [DW-1:0] sum_lane  [0:H_TILE*P_TILE-1];
  wire          vout_lane [0:H_TILE*P_TILE-1];

  genvar h, p;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        localparam int IDX_HP = h*P_TILE + p;

        // 현재 (h,p,n_idx) 요소 선택
        wire [DW-1:0] cur_sample;
        assign cur_sample = hC_reg[(IDX_HP*N_TILE) + n_idx_r];

        fp16_accum_wrapper u_acc (
          .clk       (clk),
          .rstn      (rstn),
          .valid_in  (running_r),      // 누산 중 매 싸이클 1
          .data_in   (cur_sample),
          .last_in   (running_r && last_n),
          .sum_out   (sum_lane[IDX_HP]),
          .valid_out (vout_lane[IDX_HP])
        );

        // 최종 합계를 연결(유효 싸이클은 vout_lane=1)
        assign sum_hp_o[DW*(IDX_HP+1)-1 -: DW] = sum_lane[IDX_HP];
      end
    end
  endgenerate

  // 모든 (h,p) 누산 완료 동시성 가정 → AND로 1펄스
  assign valid_o = &vout_lane;

`ifdef SIM
  // 디버그: 각 (h,p) valid 동기성 경고
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE; k = k + 1)
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(accum_n_accip): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
  end
`endif

endmodule
