// ============================================================================
// dA_mul : per-lane FP16 mul for (H_TILE) lanes
//   dA = delta_sp(h) * A(h)
// ----------------------------------------------------------------------------
module dA_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer M_LAT   = 6   // 참고용(내부 wrapper latency)
)(
  input  wire                         clk,
  input  wire                         rstn,
  input  wire                         valid_i,
  input  wire [H_TILE*DW-1:0]         lhs_i,   // delta_sp (h)
  input  wire [H_TILE*DW-1:0]         rhs_i,   // A (h)
  output wire [H_TILE*DW-1:0]         mul_o,   // dA (h)
  output wire                         valid_o
);

  // lane 배열
  wire [DW-1:0] lhs_lane   [0:H_TILE-1];
  wire [DW-1:0] rhs_lane   [0:H_TILE-1];
  wire [DW-1:0] prod_lane  [0:H_TILE-1];
  wire          vout_lane  [0:H_TILE-1];

  genvar h;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_lane
      // 벡터 → 스칼라 슬라이스
      assign lhs_lane[h] = lhs_i[DW*(h+1)-1 -: DW];
      assign rhs_lane[h] = rhs_i[DW*(h+1)-1 -: DW];

      // FP16 곱
      fp16_mul_wrapper u_mul (
        .clk       (clk),
        .valid_in  (valid_i),
        .a         (lhs_lane[h]),
        .b         (rhs_lane[h]),
        .result    (prod_lane[h]),
        .valid_out (vout_lane[h])
      );

      // 스칼라 → 벡터 재패킹
      assign mul_o[DW*(h+1)-1 -: DW] = prod_lane[h];
    end
  endgenerate

  // lane 동시성 가정 → AND 결합
  assign valid_o = vout_lane[0];  // &vout_lane;

`ifdef SIM
  // (옵션) lane valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE; k = k + 1)
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(dA_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
  end
`endif

endmodule
