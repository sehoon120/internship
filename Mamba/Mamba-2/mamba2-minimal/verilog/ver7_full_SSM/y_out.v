// ============================================================================
// y_out : (h*p) 벡터끼리 FP16 요소별 덧셈
//   y(h,p) = group_sum(h,p) + xD(h,p)
// ----------------------------------------------------------------------------
// - 내부: 각 (h,p) 래인마다 fp16_add_wrapper 1개 사용 (throughput=1 가정)
// - 입력 valid_i 한 싸이클에 두 벡터가 동시에 유효하다는 전제
// - 필요하면 아래 "두 입력 valid 분리형" 버전을 사용하세요.
// ============================================================================
module y_out #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer A_LAT   = 11  // 참고용(내부 FP16 add IP latency)
)(
  input  wire                        clk,
  input  wire                        rstn,
  input  wire                        valid_i,
  input  wire [H_TILE*P_TILE*DW-1:0] group_sum_i, // (h*p)
  input  wire [H_TILE*P_TILE*DW-1:0] xD_i,        // (h*p)
  output wire [H_TILE*P_TILE*DW-1:0] y_o,         // (h*p)
  output wire                        valid_o
);

  // 래인 슬라이스
  wire [DW-1:0] a_lane     [0:H_TILE*P_TILE-1];
  wire [DW-1:0] b_lane     [0:H_TILE*P_TILE-1];
  wire [DW-1:0] sum_lane   [0:H_TILE*P_TILE-1];
  wire          vout_lane  [0:H_TILE*P_TILE-1];

  genvar hp;
  generate
    for (hp = 0; hp < H_TILE*P_TILE; hp = hp + 1) begin : g_lane
      assign a_lane[hp] = group_sum_i[DW*(hp+1)-1 -: DW];
      assign b_lane[hp] = xD_i       [DW*(hp+1)-1 -: DW];

      // FP16 add IP (throughput=1, latency=A_LAT)
      fp16_add_wrapper u_add (
        .clk       (clk),
        .valid_in  (valid_i),
        .a         (a_lane[hp]),
        .b         (b_lane[hp]),
        .result    (sum_lane[hp]),
        .valid_out (vout_lane[hp])
      );

      assign y_o[DW*(hp+1)-1 -: DW] = sum_lane[hp];
    end
  endgenerate

  // 모든 래인이 같은 사이클에 valid_out이 뜬다고 가정 → AND 결합
  assign valid_o = &vout_lane;

`ifdef SIM
  // (옵션) 래인 valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE; k = k + 1)
      if (vout_lane[k] !== vout_lane[0])
        $display("[%0t] WARN(y_out): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
  end
`endif

endmodule
