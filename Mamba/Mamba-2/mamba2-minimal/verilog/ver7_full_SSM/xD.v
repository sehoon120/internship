// xD[b*H*P + h*H*P + p] = D[h] × x[b*H*P + h*H*P + p];
// xD.v
// xD = x * D

module xD #(
  parameter integer DW       = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire          clk,
  input  wire          rstn,
  input  wire          valid_i,
  input  wire [DW-1:0] x_i,
  input  wire [DW-1:0] D_i,
  output wire [DW-1:0] xD_o,
  output wire          valid_o
);
    fp16_mult_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(x_i),
        .b(D_i),
        .result(xD_o),
        .valid_out(valid_o)
    );
endmodule

// xD_mul : per-(h,p) FP16 mul over H_TILE × P_TILE lanes
module xD_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer M_LAT   = 6    // 참고용 (wrapper 내부 latency)
)(
  input  wire                          clk,
  input  wire                          rstn,
  input  wire                          valid_i,
  // x: (h*p) packed vector
  input  wire [H_TILE*P_TILE*DW-1:0]   x_i,
  // D: (h) packed vector (브로드캐스트 대상)
  input  wire [H_TILE*DW-1:0]          D_i,
  // xD: (h*p) packed vector
  output wire [H_TILE*P_TILE*DW-1:0]   xD_o,
  output wire                          valid_o
);

  // ----- 슬라이스 배열 -----
  wire [DW-1:0] x_lane   [0:H_TILE*P_TILE-1];
  wire [DW-1:0] D_h      [0:H_TILE-1];
  wire [DW-1:0] xD_lane  [0:H_TILE*P_TILE-1];
  wire          vout_lane[0:H_TILE*P_TILE-1];

  genvar h, p;
  generate
    // D_i → D_h[h]
    for (h = 0; h < H_TILE; h = h + 1) begin : g_D_slice
      assign D_h[h] = D_i[DW*(h+1)-1 -: DW];
    end

    // (h,p)별 곱: x(h,p) * D(h)
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        localparam int IDX = h*P_TILE + p;

        // x_i → x_lane[IDX]
        assign x_lane[IDX] = x_i[DW*(IDX+1)-1 -: DW];

        // FP16 multiplier 1개/lane
        fp16_mul_wrapper u_mul (
          .clk       (clk),
          .valid_in  (valid_i),
          .a         (x_lane[IDX]),
          .b         (D_h[h]),        // 같은 h에 대해 모든 p에 브로드캐스트
          .result    (xD_lane[IDX]),
          .valid_out (vout_lane[IDX])
        );

        // xD_lane → xD_o
        assign xD_o[DW*(IDX+1)-1 -: DW] = xD_lane[IDX];
      end
    end
  endgenerate

  // 모든 lane이 동일 latency라고 가정 → AND 결합
  assign valid_o = vout_lane[0];  // &vout_lane;

`ifdef SIM
  // (옵션) 래인 valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE; k = k + 1)
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(xD_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
  end
`endif

endmodule
