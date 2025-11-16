// dt + dt_bias

// delta: per-lane FP16 add over H_TILE lanes
module delta #(
  parameter integer DW     = 16,
  parameter integer H_TILE = 1,
  parameter integer A_LAT  = 11   // 참고용(현재 모듈 내부에서 별도 사용 X, wrapper가 valid_out 제공)
)(
  input  wire                         clk,
  input  wire                         rstn,
  input  wire                         valid_i,
  input  wire [H_TILE*DW-1:0]         dt_i,    // [lane(H_TILE)-wise][DW]
  input  wire [H_TILE*DW-1:0]         bias_i,  // [lane(H_TILE)-wise][DW]
  output wire [H_TILE*DW-1:0]         sum_o,   // [lane(H_TILE)-wise][DW]
  output wire                         valid_o
);

  // Lane별 신호
  wire [DW-1:0] a_lane   [0:H_TILE-1];
  wire [DW-1:0] b_lane   [0:H_TILE-1];
  wire [DW-1:0] sum_lane [0:H_TILE-1];
  wire          vout_lane[0:H_TILE-1];

  genvar i;
  generate
    for (i = 0; i < H_TILE; i = i + 1) begin : g_lanes
      // 벡터 -> 스칼라 슬라이스 (LSB 쪽이 lane 0)
      assign a_lane[i] = dt_i  [DW*(i+1)-1 -: DW];
      assign b_lane[i] = bias_i[DW*(i+1)-1 -: DW];

      // 각 lane에 스칼라 FP16 가산기 인스턴스
      fp16_add_wrapper u_add (
        .clk       (clk),
        .valid_in  (valid_i),
        .a         (a_lane[i]),
        .b         (b_lane[i]),
        .result    (sum_lane[i]),
        .valid_out (vout_lane[i])
      );

      // 스칼라 -> 벡터 재묶기
      assign sum_o[DW*(i+1)-1 -: DW] = sum_lane[i];
    end
  endgenerate

  // 모든 lane의 valid_out이 동일 타이밍이라고 가정.
  // 안전하게는 AND 묶기(모두 1일 때만 1). wrapper가 동질이면 vout_lane[0]만 써도 무방.
  assign valid_o = vout_lane[0];  // &vout_lane;

  // (옵션) 시뮬레이션에서 valid 동기성 체크
`ifdef SIM
  generate
    if (H_TILE > 1) begin : g_assert
      integer k;
      always @(posedge clk) if (rstn) begin
        for (k = 1; k < H_TILE; k = k + 1) begin
          if (vout_lane[k] !== vout_lane[0]) begin
            $display("[%0t] WARN(delta): lane valid mismatch: lane%0d=%b lane0=%b",
                     $time, k, vout_lane[k], vout_lane[0]);
          end
        end
      end
    end
  endgenerate
`endif

endmodule
