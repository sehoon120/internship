// ============================================================================
// accum_n_tree : Reduce over n with a pipelined adder tree (h*p*n → h*p)
//   - 입력  : hC_i(h*p*n), valid_i (1펄스)
//   - 출력  : sum_hp_o(h*p), valid_o (트리 파이프라인 딜레이 후 1펄스)
//   - 트리 단계 수 = ceil_log2(N_TILE)
//   - 각 단계는 fp16_add_wrapper(throughput=1)로 파이프라인
// ----------------------------------------------------------------------------
module accum_n #(  // _tree #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128,
  // adder latency (각 wrapper의 파이프 지연)
  parameter integer A_LAT   = 11
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,     // 입력 타일 준비 1펄스
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hC_i,        // (h*p*n)
  output wire [H_TILE*P_TILE*DW-1:0]         sum_hp_o,    // (h*p)
  output wire                                valid_o
);

  // ---------- 유틸: ceil_log2 ----------
  function integer ceil_log2;
    input integer val;
    integer i;
    begin
      i = 0;
      while ((1 << i) < val) i = i + 1;
      ceil_log2 = (val <= 1) ? 1 : i;
    end
  endfunction

  localparam integer STAGES = ceil_log2(N_TILE);

  // stage s에서 (h*p)마다 node 수: ceil(N_TILE / 2^(s+1)) (마지막에 1이 됨)
  // 데이터 배열: stage_data[s][hp_idx][node_idx]
  // 구현 편의를 위해 1차원으로 평탄화
  // 최대 노드 수: ceil(N_TILE / 2)
  // valid 파이프: STAGES 단계 + 각 단계 adder 내부 latency를 맞추는 시프트

  // 입력을 stage0로 매핑
  // stage0_nodes = N_TILE
  // 이후 stage마다 약 절반으로 감소(홀수면 +1)

  // ------- valid 파이프 -------
  // 각 stage마다 adder latency A_LAT 만큼 valid를 지연.
  reg [STAGES*A_LAT:0] vpipe;
  integer vp;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) vpipe <= 'b0;
    else       vpipe <= {vpipe[STAGES*A_LAT-1:0], valid_i};
  end
  assign valid_o = vpipe[STAGES*A_LAT]; // 최종 단계 딜레이 반영

  // ------- 데이터 파이프 -------
  genvar h, p, s, n;
  generate
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        localparam int IDX_HP = h*P_TILE + p;

        // stage 0: 입력 펼치기 (n=0..N_TILE-1)
        wire [DW-1:0] stage0 [0:N_TILE-1];
        for (n = 0; n < N_TILE; n = n + 1) begin : g_s0
          localparam int IDX_HPN = (IDX_HP*N_TILE) + n;
          assign stage0[n] = hC_i[DW*(IDX_HPN+1)-1 -: DW];
        end

        // stage별 와이어(동적 크기라 generate마다 지역 와이어 선언 불가 → 최대치로 선언)
        // 간단화를 위해 각 stage마다 최대 크기 배열을 선언하고, 유효 인덱스만 사용
        // stage s의 노드 수 계산 함수
        function integer nodes_at;
          input integer sidx;
          integer denom, base, rem;
          begin
            denom = (1 << (sidx+1));
            base  = N_TILE / denom;
            rem   = (N_TILE % denom) ? 1 : 0;
            nodes_at = base + rem; // ceil(N_TILE / 2^(sidx+1))
          end
        endfunction

        // stage 데이터 버스
        // s=0은 stage0[]로 이미 있음 → s=1..STAGES에 대해 생성
        // 이전 stage의 소스 배열을 가리키기 위한 시스템버스
        // 구현: 각 stage마다 레지스터 배열 stage_s[]
        // 더 깔끔히 하려면 2D packed array를 사용할 수 있으나, 합성용으로 아래처럼 전개
        // stage 1..STAGES
        // 입력: prev[j*2], prev[j*2+1(optional)]
        // 출력: cur[j]
        // 홀수 노드는 바이패스(두 번째 항이 없으면 첫 번째를 그대로 전달)

        // stage 1
        if (STAGES >= 1) begin : g_stage1
          localparam int N1 = nodes_at(0);
          wire [DW-1:0] cur [0:N1-1];
          genvar j1;
          for (j1 = 0; j1 < N1; j1 = j1 + 1) begin : g_j1
            localparam int L = j1*2;
            localparam int R = j1*2 + 1;
            wire [DW-1:0] a = stage0[L];
            wire [DW-1:0] b = (R < N_TILE) ? stage0[R] : {DW{1'b0}}; // 없는 경우 0
            wire [DW-1:0] s_out;

            // 홀수 노드(짝 없음)도 adder를 통일 사용(0 더하기). 자원 줄이려면 바이패스로 바꿔도 됨.
            fp16_add_wrapper u_add (
              .clk       (clk),
              .valid_in  (valid_i),
              .a         (a),
              .b         (b),
              .result    (s_out),
              .valid_out (/*unused*/)
            );
            assign cur[j1] = s_out;
          end
        end

        // stage 2..STAGES
        // 매 stage의 입력은 직전 stage의 cur[]로 가정
        // 매크로 같은 전개
        genvar ss;
        for (ss = 2; ss <= STAGES; ss = ss + 1) begin : g_stages_dyn
          localparam int PREV_N = nodes_at(ss-2); // 이전 stage 노드 수
          localparam int CUR_N  = nodes_at(ss-1); // 현재 stage 노드 수
          wire [DW-1:0] prev [0:PREV_N-1];
          wire [DW-1:0] cur  [0:CUR_N-1];

          // prev를 앞 stage의 cur로 연결
          genvar pj;
          for (pj = 0; pj < PREV_N; pj = pj + 1) begin : g_prev_bind
            if (ss == 2) begin : from_s1
              // stage1.cur → prev
              assign prev[pj] = g_stage1.cur[pj];
            end else begin : from_prev
              // g_stages_dyn[ss-1].cur → prev
              assign prev[pj] = g_stages_dyn[ss-1].cur[pj];
            end
          end

          // 현재 stage adder
          genvar cj;
          for (cj = 0; cj < CUR_N; cj = cj + 1) begin : g_cj
            localparam int L = cj*2;
            localparam int R = cj*2 + 1;
            wire [DW-1:0] a = prev[L];
            wire [DW-1:0] b = (R < PREV_N) ? prev[R] : {DW{1'b0}};
            wire [DW-1:0] s_out;

            fp16_add_wrapper u_add (
              .clk       (clk),
              .valid_in  (vpipe[(ss-1)*A_LAT]), // 한 stage 뒤로 valid를 넘깁니다
              .a         (a),
              .b         (b),
              .result    (s_out),
              .valid_out (/*unused*/)
            );
            assign cur[cj] = s_out;
          end
        end

        // 최종 결과 배선
        wire [DW-1:0] final_sum =
          (STAGES == 0) ? stage0[0] :
          g_stages_dyn[STAGES].cur[0];

        assign sum_hp_o[DW*(IDX_HP+1)-1 -: DW] = final_sum;

      end
    end
  endgenerate

endmodule
