// ============================================================================
// accum_n : Reduce over n with a pipelined adder tree (h*p*n → h*p)
// - Verilog-2001 only
// - fp16_adder_tree_128 : 128-to-1 FP16 트리(throughput=1) 사용
// - N_TOTAL은 128로 가정 (다르면 제네릭 트리 또는 zero-pad 필요)
// ============================================================================
module accum_n #(
  parameter integer DW       = 16,
  parameter integer H_TILE   = 1,
  parameter integer P_TILE   = 1,
  parameter integer N_TOTAL  = 128   // == 128 가정
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,      // 입력 준비 1펄스(모든 n이 hC_i에 이미 펼쳐져 있음)
  input  wire [H_TILE*P_TILE*N_TOTAL*DW-1:0] hC_i,         // (h*p*n) 플랫 버스
  output wire [H_TILE*P_TILE*DW-1:0]         sum_hp_o,     // (h*p)
  output wire                                valid_o
);

  genvar h, p, n;
  wire v0;  // lane0 valid tap

  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : G_H
      for (p = 0; p < P_TILE; p = p + 1) begin : G_P
        localparam integer HP = h*P_TILE + p;

        // (1) 이 (hp) 래인의 N_TOTAL개를 in_bus_hp로 평탄화
        wire [N_TOTAL*DW-1:0] in_bus_hp;

        for (n = 0; n < N_TOTAL; n = n + 1) begin : G_NMAP
          localparam integer HPN = HP*N_TOTAL + n;
          assign in_bus_hp[DW*(n+1)-1 -: DW] = hC_i[DW*(HPN+1)-1 -: DW];
        end

        // (2) 128→1 트리 1개/래인
        wire [DW-1:0] sum_lane_w;
        wire          v_lane_w;

        fp16_adder_tree_128 #(.DW(DW)) u_tree (
          .clk      (clk),
          .rst     (rstn),        // 포트명이 rstn인 버전 기준
          .valid_in (valid_i),
          .in_flat  (in_bus_hp),   // N_TOTAL*DW (=128*DW)
          .sum      (sum_lane_w),
          .valid_out(v_lane_w)
        );

        // (3) 출력 패킹
        assign sum_hp_o[DW*(HP+1)-1 -: DW] = sum_lane_w;

        // (4) lane0에서 valid 탭 (모든 래인 동일 지연 가정)
        if (HP == 0) begin : G_V0
          assign v0 = v_lane_w;
        end
      end
    end
  endgenerate

  assign valid_o = v0;

`ifdef SIM
  // (옵션) 모든 래인 valid 동기성 체크 (Vivado 호환)
  integer k;
  reg v_ref;
  always @(posedge clk) if (rstn && valid_i) begin
    v_ref = v0;
    for (k = 1; k < H_TILE*P_TILE; k = k + 1) begin
      // 각 래인의 v_lane_w는 지역선이라 직접 접근 못함 → 필요한 경우 디버그 신호 뽑아두세요.
    end
  end
`endif

endmodule
