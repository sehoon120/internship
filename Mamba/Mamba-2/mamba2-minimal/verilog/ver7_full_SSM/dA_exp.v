// ============================================================================
// dA_exp : per-lane exp for (H_TILE) lanes using softplus_or_exp16 (mode=exp)
// - 입력  : in_i (H_TILE×DW)
// - 출력  : exp_o (H_TILE×DW) = exp(in_i lane-wise)
// - valid : 모든 lane의 valid_o_e가 1일 때 1
// ----------------------------------------------------------------------------
module dA_exp #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  // 아래 4개는 softplus_or_exp16 내부 IP 지연에 맞춰 조정하세요.
  parameter integer LAT_MUL = 1,
  parameter integer LAT_ADD = 1,
  parameter integer LAT_DIV = 1,
  parameter integer LAT_EXP = 6 + 3*LAT_MUL + 3*LAT_ADD // softplus_or_exp16 코멘트 기본식
)(
  input  wire                         clk,
  input  wire                         rstn,
  input  wire                         valid_i,
  input  wire [H_TILE*DW-1:0]         in_i,     // (h)
  output wire [H_TILE*DW-1:0]         exp_o,    // (h)
  output wire                         valid_o
);

  // lane별 신호
  wire [DW-1:0] x_lane    [0:H_TILE-1];
  wire [DW-1:0] yE_lane   [0:H_TILE-1];
  wire          vE_lane   [0:H_TILE-1];

  // 사용하지 않는 Softplus 경로
  wire [DW-1:0] yS_lane   [0:H_TILE-1];
  wire          vS_lane   [0:H_TILE-1];
  wire          mode_back [0:H_TILE-1];

  genvar h;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_exp
      assign x_lane[h] = in_i[DW*(h+1)-1 -: DW];

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
        .mode_softplus_i  (1'b0),       // <<— exp 모드
        .x_i              (x_lane[h]),
        .y_o_S            (yS_lane[h]), // 미사용
        .valid_o_S        (vS_lane[h]), // 미사용
        .y_o_e            (yE_lane[h]), // exp 결과 사용
        .valid_o_e        (vE_lane[h]),
        .mode_softplus_o  (mode_back[h])
      );

      assign exp_o[DW*(h+1)-1 -: DW] = yE_lane[h];
    end
  endgenerate

  // lane valid 동시성 가정 → AND 결합
  assign valid_o = vE_lane[0];  // &vE_lane;

`ifdef SIM
  // (옵션) 검증용: lane valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE; k = k + 1)
      if (vE_lane[k] !== vE_lane[0])
        $display("[%0t] WARN(dA_exp): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vE_lane[k], vE_lane[0]);
  end
`endif

endmodule
