`timescale 1ns / 1ps

// SSMBLOCK_TOP 스트리밍 TB (Sticky-valid, II=1)
//  - (h,p)마다 8타일(=128/16) 연속 전송
//  - y_final_valid_o 시점에 FIFO로 (h,p) 인덱스 매칭 저장
module testbench_ssmblock_top_stream;

  // -----------------------------
  // 문제 크기 (필요에 맞게 조정: N_TOTAL=128 고정 가정)
  // -----------------------------
  localparam integer B        = 1;
  localparam integer H        = 24;       // 예: 16
  localparam integer P        = 64;        // 예: 4
  localparam integer N_TOTAL  = 128;      // 탑 내부 합트리 128-lane 가정
  localparam integer N_TILE   = 16;       // 포트 폭 (lane)
  localparam integer TILES    = N_TOTAL / N_TILE; // 8

  localparam integer DW       = 16;

  // -----------------------------
  // DUT I/O
  // -----------------------------
  reg                      clk;
  reg                      rstn;

  reg                      tile_valid_i;           // sticky-high
  wire                     tile_ready_o;

  reg  [DW-1:0]            dt_i, dA_i, x_i, D_i;

  reg  [N_TILE*DW-1:0]     B_tile_i;
  reg  [N_TILE*DW-1:0]     C_tile_i;
  reg  [N_TILE*DW-1:0]     hprev_tile_i;

  wire [DW-1:0]            y_final_o;
  wire                     y_final_valid_o;

  // -----------------------------
  // Memories (입력 데이터)
  // -----------------------------
  reg [DW-1:0] dt_mem   [0:B*H-1];                // dt[h]
  reg [DW-1:0] dA_mem   [0:B*H-1];                // dA[h]
  reg [DW-1:0] D_mem    [0:H-1];                  // D[h]
  reg [DW-1:0] x_mem    [0:B*H*P-1];              // x[h*P + p]
  reg [DW-1:0] B_mem    [0:B*N_TOTAL-1];          // B[n]
  reg [DW-1:0] C_mem    [0:B*N_TOTAL-1];          // C[n]
  reg [DW-1:0] h_mem    [0:B*H*P*N_TOTAL-1];      // h_prev[((h*P)+p)*N + n]

  // 결과 버퍼 (H*P 개의 스칼라) + 플랫
  reg [DW-1:0]            y_out_mem [0:H*P-1];
  reg [B*H*P*DW-1:0]      y_flat_out;

  integer fout, i, j;

  // -----------------------------
  // DUT
  // -----------------------------
  SSMBLOCK_TOP #(
      .DW(DW),
      .H_TILE(1),
      .P_TILE(1),
      .N_TILE(N_TILE),
      .N_TOTAL(N_TOTAL),
      .LAT_DX_M (6),
      .LAT_DBX_M(6),
      .LAT_DAH_M(6),
      .LAT_ADD_A(11),
      .LAT_HC_M (6)
  ) dut (
      .clk(clk),
      .rstn(rstn),

      .tile_valid_i(tile_valid_i),
      .tile_ready_o(tile_ready_o),

      .dt_i(dt_i),
      .dA_i(dA_i),
      .x_i(x_i),
      .D_i(D_i),

      .B_tile_i(B_tile_i),
      .C_tile_i(C_tile_i),
      .hprev_tile_i(hprev_tile_i),

      .y_final_o(y_final_o),
      .y_final_valid_o(y_final_valid_o)
  );

  // -----------------------------
  // 100 MHz clock
  // -----------------------------
  initial clk = 1'b0;
  always #5 clk = ~clk;

  // -----------------------------
  // 결과 인덱스 매칭용 FIFO
  // -----------------------------
//  localparam integer QDEPTH = H*P + 64;
  localparam integer QDEPTH = (H*P) + 2048;
  localparam integer QW     = $clog2(QDEPTH);
  

  reg [31:0]   idx_q   [0:QDEPTH-1];
  reg [QW-1:0] q_head, q_tail;
  reg [31:0]   q_count;

  // enqueue 1클럭 펄스
  reg          enq_req;
  reg [31:0]   enq_idx;

  // 큐 처리 + 결과 기록기
  integer deq_idx;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      q_head <= 0; q_tail <= 0; q_count <= 0;
    end else begin
      // Enqueue
      if (enq_req) begin
        if (q_count == QDEPTH) begin
          $display("[TB][%0t] ERROR: idx queue overflow", $time); $finish;
        end
        idx_q[q_tail] <= enq_idx;
        q_tail        <= (q_tail == QDEPTH-1) ? 0 : (q_tail + 1'b1);
        q_count       <= q_count + 1;
      end
      // Dequeue on y_valid → y_out_mem에 저장
      if (y_final_valid_o) begin
        if (q_count == 0) begin
          $display("[TB][%0t] ERROR: idx queue underflow", $time); $finish;
        end else begin
          
          deq_idx  = idx_q[q_head];
          q_head   <= (q_head == QDEPTH-1) ? 0 : (q_head + 1'b1);
          q_count  <= q_count - 1;
          y_out_mem[deq_idx] <= y_final_o;
        end
      end
    end
  end

  // -----------------------------
  // Payload packer
  // -----------------------------
  task pack_tile_payload(input integer h_a, input integer p_a, input integer n_base);
    begin
      // B, C
      for (j = 0; j < N_TILE; j = j + 1) begin
        B_tile_i[DW*j +: DW] = B_mem[n_base + j];
        C_tile_i[DW*j +: DW] = C_mem[n_base + j];
      end
      // h_prev
      for (j = 0; j < N_TILE; j = j + 1) begin
        hprev_tile_i[DW*j +: DW] = h_mem[((h_a*P) + p_a)*N_TOTAL + (n_base + j)];
      end
    end
  endtask

  // -----------------------------
  // Sticky-valid 스트리밍 엔진
  //   - 전체 스캔 동안 tile_valid_i=1 유지 (끝에서만 0)
  //   - ready=1 싸이클에서만 타일 전환/카운트
  //   - 8번째 타일 수락 시 (h,p) 인덱스 enqueue
  // -----------------------------
  integer cur_h, cur_p;        // 현재 (h,p)
  integer n_sent;              // 현재 (h,p)에서 전송된 타일 수: 0..TILES-1
  integer groups_done;         // 완료된 (h,p) 개수
  integer next_h, next_p;
  integer idx_now;

  task automatic set_scalars(input integer h_a, input integer p_a);
  begin
    dt_i <= dt_mem[h_a];
    dA_i <= dA_mem[h_a];
    D_i  <= D_mem[h_a];
    x_i  <= x_mem[h_a*P + p_a];
  end
  endtask

  task automatic next_hp(input integer h_a, input integer p_a,
                         output integer h_b, output integer p_b);
  begin
    if (p_a == P-1) begin h_b = h_a + 1; p_b = 0; end
    else             begin h_b = h_a;     p_b = p_a + 1; end
  end
  endtask

  // -----------------------------
  // Reset / Load / Stream all
  // -----------------------------
  initial begin
    // 초기화
    rstn = 1'b0;
    tile_valid_i = 1'b0;
    dt_i=0; dA_i=0; D_i=0; x_i=0;
    B_tile_i=0; C_tile_i=0; hprev_tile_i=0;
    q_head=0; q_tail=0; q_count=0; enq_req=1'b0; enq_idx=0;

    // 파일 경로 (Windows 예시: 네 경로에 맞게 조정)
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",         dt_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",         dA_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",          D_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",          x_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",          B_mem);   // 길이 128 필요
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",          C_mem);   // 길이 128 필요
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex",  h_mem);   // H*P*128 길이 필요

    // 결과 버퍼 클리어
    for (i = 0; i < H*P; i = i + 1) y_out_mem[i] = 0;

    // 리셋 해제
    repeat (10) @(posedge clk);
    rstn = 1'b1;
    repeat (2) @(posedge clk);

    $display("==== Stream start: H=%0d, P=%0d, N_TOTAL=%0d, N_TILE=%0d, TILES=%0d ====",
             H, P, N_TOTAL, N_TILE, TILES);

    // 스트리밍 상태 초기화
    cur_h = 0; cur_p = 0; n_sent = 0; groups_done = 0;

    // 첫 (h,p) 스칼라 & 첫 타일 준비
    set_scalars(cur_h, cur_p);
    pack_tile_payload(cur_h, cur_p, 0);

    // ★ valid를 전체 스트림 동안 유지 (sticky)
    tile_valid_i <= 1'b1;

    // 전체 (h,p) 처리
    while (groups_done < H*P) begin
      @(posedge clk);
      if (tile_ready_o) begin
        if (n_sent == TILES-1) begin
          // 이번 싸이클에 8번째 타일이 수락됨 → (h,p) 완료
          idx_now <= cur_h*P + cur_p;
          enq_idx <= idx_now; enq_req <= 1'b1;  // 결과 인덱스 enqueue (1클럭)

          // 다음 (h,p) 결정
          groups_done = groups_done + 1;
          if (groups_done == H*P) begin
            // 전체 종료 → 다음 싸이클에 valid 내림
            tile_valid_i <= 1'b0;
          end else begin
            next_hp(cur_h, cur_p, next_h, next_p);
            cur_h = next_h; cur_p = next_p;

            // 다음 (h,p) 첫 타일 준비 (버블 없이 이어감)
            set_scalars(cur_h, cur_p);
            pack_tile_payload(cur_h, cur_p, 0);
            n_sent = 0;
          end
        end else begin
          // 중간 타일 → 다음 타일 프리페치
          n_sent = n_sent + 1;
          pack_tile_payload(cur_h, cur_p, n_sent*N_TILE);
        end
      end

      // enqueue 펄스 다운 (정확히 1클럭)
      if (enq_req) enq_req <= 1'b0;
    end

    // 남은 결과 드레인
    $display("[%0t] Draining %0d pending results...", $time, q_count);
    while (q_count > 0) @(posedge clk);

    // 플랫화 & 저장
    for (i = 0; i < H*P; i = i + 1)
      y_flat_out[DW*i +: DW] = y_out_mem[i];

    fout = $fopen("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out.hex", "w");
    for (i = 0; i < B*H*P; i = i + 1)
      $fdisplay(fout, "%04h", y_flat_out[DW*i +: DW]);
    $fclose(fout);

    $display("✅ Done. Output saved to 0_y_out.hex");
    #20 $finish;
  end

endmodule
