// ============================================================================
// hnext_add : per-(h,p,n) FP16 add over H_TILE × P_TILE × N_TILE lanes
//   hnext(h,p,n) = dAh(h,p,n) + dBx(h,p,n)
//   입력  : dAh_i (h*p*n), dBx_i (h*p*n)
//   출력  : sum_o (h*p*n)
// ----------------------------------------------------------------------------
module hnext_add #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128,
  parameter integer A_LAT   = 11   // 참고용(내부 fp16_add_wrapper latency)
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  dAh_i,   // (h*p*n)
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  dBx_i,   // (h*p*n)
  output wire [H_TILE*P_TILE*N_TILE*DW-1:0]  sum_o,   // (h*p*n)
  output wire                                valid_o
);

  // 슬라이스 배열
  wire [DW-1:0] dAh_lane   [0:H_TILE*P_TILE*N_TILE-1];
  wire [DW-1:0] dBx_lane   [0:H_TILE*P_TILE*N_TILE-1];
  wire [DW-1:0] sum_lane   [0:H_TILE*P_TILE*N_TILE-1];
  wire          vout_lane  [0:H_TILE*P_TILE*N_TILE-1];

  genvar h, p, n;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        for (n = 0; n < N_TILE; n = n + 1) begin : g_n
          localparam int IDX_HPN = (h*P_TILE + p)*N_TILE + n;

          // 벡터 → 스칼라 슬라이스
          assign dAh_lane[IDX_HPN] = dAh_i[DW*(IDX_HPN+1)-1 -: DW];
          assign dBx_lane[IDX_HPN] = dBx_i[DW*(IDX_HPN+1)-1 -: DW];

          // FP16 가산기 1개/lane
          fp16_add_wrapper u_add (
            .clk       (clk),
            .valid_in  (valid_i),
            .a         (dAh_lane[IDX_HPN]),
            .b         (dBx_lane[IDX_HPN]),
            .result    (sum_lane[IDX_HPN]),
            .valid_out (vout_lane[IDX_HPN])
          );

          // 스칼라 → 벡터 재패킹
          assign sum_o[DW*(IDX_HPN+1)-1 -: DW] = sum_lane[IDX_HPN];
        end
      end
    end
  endgenerate

  // lane 동시성 가정 → AND 결합
  assign valid_o = vout_lane[0];  // &vout_lane;

`ifdef SIM
  // (옵션) lane valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE*N_TILE; k = k + 1)
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(hnext_add): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
  end
`endif

endmodule
