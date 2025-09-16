`timescale 1ns/1ps

// ===== File paths (edit here) =====
`define PATH_PREFIX "/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/"
`define F_DT        {`PATH_PREFIX, "0_dt_full_full_SSM.hex"}
`define F_DTBIAS    {`PATH_PREFIX, "0_dt_bias_full_SSM.hex"}
`define F_A         {`PATH_PREFIX, "0_A_full_SSM.hex"}
`define F_X         {`PATH_PREFIX, "0_x_full_SSM.hex"}
`define F_D         {`PATH_PREFIX, "0_D_full_SSM.hex"}
`define F_B         {`PATH_PREFIX, "0_B_full_SSM.hex"}
`define F_C         {`PATH_PREFIX, "0_C_full_SSM.hex"}
`define F_HPREV     {`PATH_PREFIX, "0_ssm_state_full_SSM.hex"}
`define F_YOUT      {`PATH_PREFIX, "0_y_out_full_SSM.hex"}

module tb_ssmblock_fullscan;
  // -----------------------------
  // Parameters
  // -----------------------------
  localparam integer B       = 1;
  localparam integer H       = 24;
  localparam integer P       = 64;
  localparam integer N       = 128;
  localparam integer H_tile  = 1;
  localparam integer P_tile  = 1;
  localparam integer DW      = 16;
  localparam integer N_TILE  = 16;         // 16씩 타일 → 8타일
  localparam integer TILES   = N / N_TILE; // 8

  // 유효성 체크
  initial begin
    if (H % H_tile != 0) begin
      $display("ERROR: H(%0d) %% H_tile(%0d) != 0", H, H_tile); $finish;
    end
    if (P % P_tile != 0) begin
      $display("ERROR: P(%0d) %% P_tile(%0d) != 0", P, P_tile); $finish;
    end
    if (N % N_TILE != 0) begin
      $display("ERROR: N(%0d) %% N_TILE(%0d) != 0", N, N_TILE); $finish;
    end
  end

  // -----------------------------
  // DUT I/O
  // -----------------------------
  reg                      clk;
  reg                      rstn;

  reg                      tile_valid_i;
  wire                     tile_ready_o;

  reg  [DW-1:0]            dt_i;
  reg  [DW-1:0]            dt_bias_i;   // NEW
  reg  [DW-1:0]            A_i;         // NEW
  reg  [DW-1:0]            x_i;
  reg  [DW-1:0]            D_i;

  reg  [N_TILE*DW-1:0]     B_tile_i;
  reg  [N_TILE*DW-1:0]     C_tile_i;
  reg  [H_tile*P_tile*N_TILE*DW-1:0] hprev_tile_i; // (hp*n) 폭

  wire [DW-1:0]            y_final_o;
  wire                     y_final_valid_o;

  // -----------------------------
  // Memories
  // -----------------------------
  reg [DW-1:0] dt_mem       [0:B*H-1];          // dt[h]
  reg [DW-1:0] dt_bias_mem  [0:B*H-1];          // dt_bias[h]
  reg [DW-1:0] A_mem        [0:B*H-1];          // A[h]
  reg [DW-1:0] D_mem        [0:H-1];            // D[h]
  reg [DW-1:0] x_mem        [0:B*H*P-1];        // x[h*P + p]
  reg [DW-1:0] B_mem        [0:B*N-1];          // B[n]
  reg [DW-1:0] C_mem        [0:B*N-1];          // C[n]
  reg [DW-1:0] h_mem        [0:B*H*P*N-1];      // h_prev[((h*P)+p)*N + n]

  // 결과 버퍼 (H*P 개의 스칼라)
  reg [DW-1:0] y_out_mem [0:H*P-1];

  integer h_blk, p_blk, h_rel, p_rel, h_abs, p_abs;
  integer t, j, base, fout;

  // -----------------------------
  // DUT
  // -----------------------------
  SSMBLOCK_TOP #(
      .DW(DW), .H_TILE(H_tile), .P_TILE(P_tile),
      .N_TILE(N_TILE), .N_TOTAL(N),
      .LAT_DX_M(6), .LAT_DBX_M(6), .LAT_DAH_M(6),
      .LAT_ADD_A(11), .LAT_HC_M(6)
  ) dut (
      .clk(clk),
      .rstn(rstn),

      .tile_valid_i(tile_valid_i),
      .tile_ready_o(tile_ready_o),

      .dt_i(dt_i),
      .dt_bias_i(dt_bias_i),   // NEW
      .A_i(A_i),               // NEW
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
  // (h,p) 결과 매칭용 인덱스 FIFO
  // -----------------------------
  integer idx_fifo [0:H*P-1];
  integer head, tail, fifo_count;
  integer pop_idx;

  task fifo_reset;
    integer k;
    begin
      head = 0; tail = 0; fifo_count = 0;
      for (k = 0; k < H*P; k = k + 1) idx_fifo[k] = 0;
    end
  endtask

  task fifo_push(input integer val);
    begin
      idx_fifo[tail] = val;
      tail = tail + 1;
      fifo_count = fifo_count + 1;
    end
  endtask

  task fifo_pop(output integer val);
    begin
      val = idx_fifo[head];
      head = head + 1;
      fifo_count = fifo_count - 1;
    end
  endtask

  // 출력 수집기
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      // nothing
    end else begin
      if (y_final_valid_o) begin
        if (fifo_count == 0) begin
          $display("[%0t] WARN: y_final_valid_o with empty FIFO!", $time);
        end else begin
          fifo_pop(pop_idx);
          y_out_mem[pop_idx] <= y_final_o;
          // $display("[%0t] y_out_mem[%0d] = 0x%04h", $time, pop_idx, y_final_o);
        end
      end
    end
  end

  // -----------------------------
  // Helpers
  // -----------------------------
  // 타일 payload 패킹
  task pack_tile_payload(input integer h_a, input integer p_a, input integer n_base);
    begin
      // B, C : n_base ~ n_base+N_TILE-1
      for (j = 0; j < N_TILE; j = j + 1) begin
        B_tile_i[DW*(j+1)-1 -: DW] = B_mem[n_base + j];
        C_tile_i[DW*(j+1)-1 -: DW] = C_mem[n_base + j];
      end
      // hprev(h,p,n)
      for (j = 0; j < N_TILE; j = j + 1) begin
        hprev_tile_i[DW*(j+1)-1 -: DW] =
          h_mem[((h_a*P) + p_a)*N + (n_base + j)];
      end
    end
  endtask

  // 단일 (h_abs, p_abs) 그룹 처리: 스칼라 세팅 → 8타일 전송(ready에 1싸이클 valid) → 인덱스 FIFO push
  task process_one_hp(input integer h_a, input integer p_a);
    integer t_local, base_local;
    begin
      // 스칼라 고정 (그룹 내 불변)
      dt_i      <= dt_mem     [h_a];
      dt_bias_i <= dt_bias_mem[h_a];
      A_i       <= A_mem      [h_a];
      D_i       <= D_mem      [h_a];
      x_i       <= x_mem      [h_a*P + p_a];

      // 8개 타일 송신
      for (t_local = 0; t_local < TILES; t_local = t_local + 1) begin
        base_local = t_local * N_TILE;

        // payload 준비
        pack_tile_payload(h_a, p_a, base_local);

        // ready 관찰 → 그 싸이클에 딱 1싸이클 valid 펄스
        @(posedge clk);
        while (tile_ready_o == 1'b0) @(posedge clk);
        tile_valid_i <= 1'b1;
        @(posedge clk);
        tile_valid_i <= 1'b0;
      end

      // 해당 (h,p)의 결과는 나중에 올라오므로, 인덱스를 FIFO에 적재
      fifo_push(h_a*P + p_a);
    end
  endtask

  // -----------------------------
  // Reset / Load / Full Scan
  // -----------------------------
  integer idx;
  initial begin
    rstn = 1'b0;
    tile_valid_i = 1'b0;
    dt_i = 0; dt_bias_i = 0; A_i = 0; D_i = 0; x_i = 0;
    B_tile_i = 0; C_tile_i = 0; hprev_tile_i = 0;
    fifo_reset();

    // 입력 로드 (상단 define 경로 사용)
    $readmemh(`F_DT,     dt_mem);
    $readmemh(`F_DTBIAS, dt_bias_mem);
    $readmemh(`F_A,      A_mem);
    $readmemh(`F_X,      x_mem);
    $readmemh(`F_D,      D_mem);
    $readmemh(`F_B,      B_mem);
    $readmemh(`F_C,      C_mem);
    $readmemh(`F_HPREV,  h_mem);

    // Reset release
    #100 rstn = 1'b1;
    @(posedge clk); @(posedge clk);

    $display("==== Full scan start: H=%0d, P=%0d, N=%0d (H_tile=%0d, P_tile=%0d, N_TILE=%0d, TILES=%0d) ====",
             H, P, N, H_tile, P_tile, N_TILE, TILES);

    // 블로킹 순회: (h_blk, p_blk) 타일 블록 → 내부에서 (h_rel, p_rel) 순회
    for (h_blk = 0; h_blk < H; h_blk = h_blk + H_tile) begin
      for (p_blk = 0; p_blk < P; p_blk = p_blk + P_tile) begin
        for (h_rel = 0; h_rel < H_tile; h_rel = h_rel + 1) begin
          for (p_rel = 0; p_rel < P_tile; p_rel = p_rel + 1) begin : LP_REL
            h_abs = h_blk + h_rel;
            p_abs = p_blk + p_rel;
            process_one_hp(h_abs, p_abs);
          end
        end
      end
    end

    // 남아있는 결과 드레인
    $display("[%0t] Draining %0d pending results...", $time, fifo_count);
    while (fifo_count > 0) @(posedge clk);
    #100;

    // 결과 저장
    fout = $fopen(`F_YOUT, "w");
    for (idx = 0; idx < H*P; idx = idx + 1)
      $fdisplay(fout, "%04h", y_out_mem[idx]);
    $fclose(fout);

    $display("✅ Full scan completed. Results written: %s", `F_YOUT);

    #50 $finish;
  end

endmodule
