// ============================================================================
// sp_dt : per-lane Softplus for (H_TILE) lanes using softplus_or_exp16
// - 입력  : dt_i (H_TILE×DW)
// - 출력  : sp_dt_o (H_TILE×DW) = softplus(dt_i lane-wise)
// - valid : 모든 lane의 valid_o_S가 1일 때 1 (동일 latency 가정)
// ----------------------------------------------------------------------------
// 주의:
//  - softplus_or_exp16은 exp/softplus 겸용이므로 mode_softplus_i=1로 고정
//  - y_o_S/valid_o_S만 사용 (y_o_e는 무시)
//  - SP_LAT 파라미터는 상위 설계 편의를 위한 문서용이며, 내부 지연은
//    softplus_or_exp16 인스턴스의 LAT_* 조합으로 결정됨.
// ============================================================================
module sp_dt #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer SP_LAT  = 33, // 문서용(파이프 전체 지연), 실제 지연은 softplus_or_exp16 파라미터에 따름

  // 아래 4개는 softplus_or_exp16 내부 IP 지연에 맞춰 조정하세요.
  parameter integer LAT_MUL = 1,
  parameter integer LAT_ADD = 1,
  parameter integer LAT_DIV = 1,
  parameter integer LAT_EXP = 6 + 3*LAT_MUL + 3*LAT_ADD // 예시: 코멘트의 기본식
)(
  input  wire                         clk,
  input  wire                         rstn,
  input  wire                         valid_i,
  input  wire [H_TILE*DW-1:0]         dt_i,       // (h)
  output wire [H_TILE*DW-1:0]         sp_dt_o,    // (h)
  output wire                         valid_o
);

  // lane별 신호
  wire [DW-1:0] x_lane     [0:H_TILE-1];
  wire [DW-1:0] yS_lane    [0:H_TILE-1];
  wire          vS_lane    [0:H_TILE-1];

  // (필요 없지만 포트 규격상 받게 되는 exp 출력 무시용 와이어)
  wire [DW-1:0] yE_lane    [0:H_TILE-1];
  wire          vE_lane    [0:H_TILE-1];
  wire          mode_back  [0:H_TILE-1];

  genvar h;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_softplus
      assign x_lane[h] = dt_i[DW*(h+1)-1 -: DW];

      softplus_or_exp16 #(
        .DW      (DW),
        .LAT_MUL (LAT_MUL),
        .LAT_ADD (LAT_ADD),
        .LAT_DIV (LAT_DIV),
        .LAT_EXP (LAT_EXP)
      ) u_soft_or_exp (
        .clk              (clk),
        .rstn             (rstn),
        .valid_i          (valid_i),
        .mode_softplus_i  (1'b1),       // <<— 항상 Softplus 모드
        .x_i              (x_lane[h]),
        .y_o_S            (yS_lane[h]), // Softplus 결과만 사용
        .valid_o_S        (vS_lane[h]),
        .y_o_e            (yE_lane[h]), // exp 출력은 사용 안 함
        .valid_o_e        (vE_lane[h]),
        .mode_softplus_o  (mode_back[h])
      );

      assign sp_dt_o[DW*(h+1)-1 -: DW] = yS_lane[h];
    end
  endgenerate

  // lane 동시성 가정 → AND 결합
  assign valid_o = vS_lane[0];  // &vS_lane;

`ifdef SIM
  // (옵션) 검증용: 모든 lane의 valid 동기 확인
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE; k = k + 1) begin
      if (vS_lane[k] !== vS_lane[0])
        $display("[%0t] WARN(sp_dt): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vS_lane[k], vS_lane[0]);
    end
  end
`endif

endmodule
