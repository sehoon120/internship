// ============================================================================
// accum_n : Reduce over n with a pipelined adder tree (h*p*n → h*p)
//  - Verilog-2001 only
//  - stage 수: STAGES = ceil_log2(N_TILE)
//  - 각 stage는 reduce_stage로 구성 (pairwise 합산)
//  - valid_o = 마지막 stage의 valid_out
// ============================================================================
module accum_n #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,     // 입력 타일 준비 1펄스
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hC_i,        // (h*p*n)
  output wire [H_TILE*P_TILE*DW-1:0]         sum_hp_o,    // (h*p)
  output wire                                valid_o
);

  // --------- 유틸 함수들 (정수, 정적 평가) ---------
  function integer ceil_log2;
    input integer val;
    integer i;
    begin
      if (val <= 1) begin ceil_log2 = 0; end
      else begin
        i = 0;
        while ((1 << i) < val) i = i + 1;
        ceil_log2 = i;
      end
    end
  endfunction

  function integer ceil_div_pow2;
    input integer n;
    input integer k; // divide by 2^k, ceil
    integer denom;
    begin
      denom = (1 << k);
      ceil_div_pow2 = (n + denom - 1) / denom;
    end
  endfunction

  localparam integer STAGES = ceil_log2(N_TILE);

  genvar h, p, n, s;

  // (special) N_TILE == 1이면 트리 없음: 바로 전달
  generate if (STAGES == 0) begin : G_BYPASS_ALL
    for (h = 0; h < H_TILE; h = h + 1) begin : G_H0
      for (p = 0; p < P_TILE; p = p + 1) begin : G_P0
        localparam integer HP  = h*P_TILE + p;
        localparam integer HPN = HP*N_TILE; // n=0만 존재

        // n=0 슬라이스 그대로 전달
        assign sum_hp_o[DW*(HP+1)-1 -: DW] = hC_i[DW*(HPN+1)-1 -: DW];
      end
    end
    assign valid_o = valid_i;
  end else begin : G_TREE
    // 각 (h,p)마다 독립적으로 N_TILE을 트리로 reduce
    wire v_last_any; // 마지막 stage valid (어느 (h,p)와도 동일한 레이턴시 가정)
    // 첫 (h=0,p=0)의 마지막 stage valid을 탭
    wire v_last_hp0;

    for (h = 0; h < H_TILE; h = h + 1) begin : G_H
      for (p = 0; p < P_TILE; p = p + 1) begin : G_P
        localparam integer HP = h*P_TILE + p;

        // ---- Stage0 입력 버스: N_TILE개의 항목을 1차 버스로 맵 ----
        wire [DW*N_TILE-1:0] stage0_bus;

        for (n = 0; n < N_TILE; n = n + 1) begin : G_MAP0
          // hC_i 인덱스: ((h*P_TILE + p)*N_TILE + n)
          localparam integer HPN = HP*N_TILE + n;
          assign stage0_bus[DW*(n+1)-1 -: DW] = hC_i[DW*(HPN+1)-1 -: DW];
        end

        // ---- Stage 1..STAGES 체인 ----
        // g_stage[s] : s-th stage (1-based)
        for (s = 1; s <= STAGES; s = s + 1) begin : g_stage
          localparam integer PREV_N = ceil_div_pow2(N_TILE, s-1);
          localparam integer CUR_N  = ceil_div_pow2(N_TILE, s);

          wire [DW*CUR_N-1:0] cur_bus_s;
          wire                 v_stage_s;

          if (s == 1) begin : FROM_S0
            reduce_stage #(.DW(DW), .PREV_N(PREV_N)) u_rs (
              .clk      (clk),
              .rstn     (rstn),
              .valid_in (valid_i),
              .prev_bus (stage0_bus),
              .cur_bus  (cur_bus_s),
              .valid_out(v_stage_s)
            );
          end else begin : FROM_PREV
            reduce_stage #(.DW(DW), .PREV_N(PREV_N)) u_rs (
              .clk      (clk),
              .rstn     (rstn),
              .valid_in (g_stage[s-1].v_stage_s),
              .prev_bus (g_stage[s-1].cur_bus_s),
              .cur_bus  (cur_bus_s),
              .valid_out(v_stage_s)
            );
          end
        end

        // 마지막 stage 결과(항상 1개 요소) → sum_hp_o(해당 hp)
        assign sum_hp_o[DW*(HP+1)-1 -: DW] = g_stage[STAGES].cur_bus_s[DW*1-1 -: DW];

        // valid 탭: (h=0,p=0)에서만 valid_o 원천을 탭
        if (HP == 0) begin : G_VTAP
          assign v_last_hp0 = g_stage[STAGES].v_stage_s;
        end

      end
    end

    assign v_last_any = v_last_hp0;
    assign valid_o    = v_last_any;
  end endgenerate

endmodule





// ============================================================================
// reduce_stage : pairwise reduce (PREV_N -> ceil(PREV_N/2))
//  - Verilog-2001 only (no net arrays, no SV)
//  - 각 합산은 fp16_add_wrapper(throughput=1, 고정 레이턴시) 사용
//  - valid_out은 첫 adder의 valid_out을 탭(모두 동일 레이턴시 가정)
// ============================================================================
module reduce_stage #(
  parameter integer DW      = 16,
  parameter integer PREV_N  = 2
)(
  input  wire                     clk,
  input  wire                     rstn,
  input  wire                     valid_in,
  input  wire [DW*PREV_N-1:0]     prev_bus,  // [0]가 LSB 슬라이스
  output wire [DW*((PREV_N+1)/2)-1:0] cur_bus,
  output wire                     valid_out
);
  localparam integer CUR_N = (PREV_N+1)/2;

  wire v0;  // 첫 래인의 valid_out 탭
  genvar j;
  generate
    for (j = 0; j < CUR_N; j = j + 1) begin : G
      // 인덱스 2*j, 2*j+1
      wire [DW-1:0] a_w;
      wire [DW-1:0] b_w;
      wire [DW-1:0] s_w;
      wire          v_w;

      assign a_w = prev_bus[DW*(2*j+1)-1 -: DW];
      assign b_w = ( (2*j+1) < PREV_N ) ? prev_bus[DW*(2*j+2)-1 -: DW] : {DW{1'b0}};

      fp16_add_wrapper u_add (
        .clk       (clk),
        .valid_in  (valid_in),
        .a         (a_w),
        .b         (b_w),
        .result    (s_w),
        .valid_out (v_w)
      );

      assign cur_bus[DW*(j+1)-1 -: DW] = s_w;

      if (j == 0) begin : GV
        assign v0 = v_w;
      end
    end
  endgenerate

  assign valid_out = v0;

endmodule
