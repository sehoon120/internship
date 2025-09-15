// ============================================================================
// dx_mul : per-(h,p) FP16 mul over H_TILE × P_TILE lanes
//   dx(h,p) = delta_sp(h) * x(h,p)
// ----------------------------------------------------------------------------
module dx_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer MUL_LAT = 6   // 참고용(내부 wrapper latency)
)(
  input  wire                          clk,
  input  wire                          rstn,
  input  wire                          valid_i,
  input  wire [H_TILE*DW-1:0]          h_i,     // delta_sp (h)
  input  wire [H_TILE*P_TILE*DW-1:0]   x_i,     // x (h*p)
  output wire [H_TILE*P_TILE*DW-1:0]   dx_o,    // dx (h*p)
  output wire                          valid_o
);

  // 슬라이스 배열
  wire [DW-1:0] h_lane    [0:H_TILE-1];
  wire [DW-1:0] x_lane    [0:H_TILE*P_TILE-1];
  wire [DW-1:0] dx_lane   [0:H_TILE*P_TILE-1];
  wire          vout_lane [0:H_TILE*P_TILE-1];

  genvar h, p;
  generate
    // h 벡터 슬라이스
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h_slice
      assign h_lane[h] = h_i[DW*(h+1)-1 -: DW];
    end

    // (h,p)별 곱: x(h,p) * h(h)
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        localparam int IDX = h*P_TILE + p;

        assign x_lane[IDX] = x_i[DW*(IDX+1)-1 -: DW];

        fp16_mul_wrapper u_mul (
          .clk       (clk),
          .valid_in  (valid_i),
          .a         (x_lane[IDX]),
          .b         (h_lane[h]),     // 같은 h에 대해 모든 p에 브로드캐스트
          .result    (dx_lane[IDX]),
          .valid_out (vout_lane[IDX])
        );

        assign dx_o[DW*(IDX+1)-1 -: DW] = dx_lane[IDX];
      end
    end
  endgenerate

  // 모든 lane이 동일 latency라고 가정 → AND 결합
  assign valid_o = vout_lane[0];  // &vout_lane;

`ifdef SIM
  // (옵션) lane valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE; k = k + 1) begin
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(dx_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
    end
  end
`endif

endmodule
