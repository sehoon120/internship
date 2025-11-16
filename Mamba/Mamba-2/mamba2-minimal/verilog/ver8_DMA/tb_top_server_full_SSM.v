`timescale 1ns/1ps
`define ORDER_P_MAJOR

// ===== File paths (edit here) =====
`define PATH_PREFIX "/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/"
`define F_DT        {`PATH_PREFIX, "0_dt_full_SSM.hex"}
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
  localparam integer N_TILE  = 128;
  localparam integer TILES   = N / N_TILE; // =1 (여기선 128/128)

  // -----------------------------
  // DUT I/O
  // -----------------------------
  reg                      clk;
  reg                      rstn;

  reg                      tile_valid_i;
  wire                     tile_ready_o;

  reg  [DW-1:0]            dt_i;
  reg  [DW-1:0]            dt_bias_i;
  reg  [DW-1:0]            A_i;
  reg  [DW-1:0]            x_i;
  reg  [DW-1:0]            D_i;

  reg  [N_TILE*DW-1:0]     B_tile_i;
  reg  [N_TILE*DW-1:0]     C_tile_i;
  reg  [H_tile*P_tile*N_TILE*DW-1:0] hprev_tile_i;

  wire [DW-1:0]            y_final_o;
  wire                     y_final_valid_o;

  // -----------------------------
  // Memories
  // -----------------------------
  reg [DW-1:0] dt_mem       [0:B*H-1];
  reg [DW-1:0] dt_bias_mem  [0:B*H-1];
  reg [DW-1:0] A_mem        [0:B*H-1];
  reg [DW-1:0] D_mem        [0:H-1];
  reg [DW-1:0] x_mem        [0:B*H*P-1];
  reg [DW-1:0] B_mem        [0:B*N-1];
  reg [DW-1:0] C_mem        [0:B*N-1];
  reg [DW-1:0] h_mem        [0:B*H*P*N-1];

  // 결과 버퍼 (H*P 개의 스칼라)
  reg [DW-1:0] y_out_mem [0:H*P-1];

  integer h_blk, p_blk, h_rel, p_rel, h_abs, p_abs;
  integer t, j, base, fout;

  // -----------------------------
  // DUT (당신의 SSMBLOCK_TOP 존재 가정)
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
      .dt_bias_i(dt_bias_i),
      .A_i(A_i),
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

  task fifo_push; input integer val; begin
      idx_fifo[tail] = val;
      tail = tail + 1;
      fifo_count = fifo_count + 1;
  end endtask

  task fifo_pop; output integer val; begin
      val = idx_fifo[head];
      head = head + 1;
      fifo_count = fifo_count - 1;
  end endtask

  // -----------------------------
  // 모니터 연결 신호 (WRITE 이벤트)
  // -----------------------------
  localparam [63:0] BASE_Y = 64'h0; // DRAM 베이스 주소 가정
  reg        wr_ev_valid,   wr_ev_valid_q;
  reg [63:0] wr_ev_addr,    wr_ev_addr_q;

  // 출력 수집기 + 모니터 트리거
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      wr_ev_valid   <= 1'b0;  wr_ev_addr   <= 64'd0;
      wr_ev_valid_q <= 1'b0;  wr_ev_addr_q <= 64'd0;
    end else begin
      wr_ev_valid <= 1'b0; // 기본 0
      if (y_final_valid_o) begin
        if (fifo_count == 0) begin
          $display("[%0t] WARN: y_final_valid_o with empty FIFO!", $time);
        end else begin
          fifo_pop(pop_idx);
          y_out_mem[pop_idx] <= y_final_o;
          wr_ev_addr  <= BASE_Y + (pop_idx << 1); // FP16=2B
          wr_ev_valid <= 1'b1;
        end
      end
      // 1-cycle stage
      wr_ev_valid_q <= wr_ev_valid;
      wr_ev_addr_q  <= wr_ev_addr;
    end
  end

  // -----------------------------
  // Helpers
  // -----------------------------
  task process_one_hp; input integer h_a, p_a;
    integer t_local, base_local, j;
    begin
      // 스칼라
      dt_i      <= dt_mem     [h_a];
      dt_bias_i <= dt_bias_mem[h_a];
      A_i       <= A_mem      [h_a];
      D_i       <= D_mem      [h_a];
      x_i       <= x_mem      [h_a*P + p_a];

      // 타일 송신
      for (t_local = 0; t_local < TILES; t_local = t_local + 1) begin
        base_local = t_local * N_TILE;
        // payload
        for (j = 0; j < N_TILE; j = j + 1) begin
          B_tile_i[DW*(j+1)-1 -: DW] = B_mem[base_local + j];
          C_tile_i[DW*(j+1)-1 -: DW] = C_mem[base_local + j];
          hprev_tile_i[DW*(j+1)-1 -: DW] =
            h_mem[((h_a*P) + p_a)*N + (base_local + j)];
        end
        tile_valid_i <= 1'b1;
        @(posedge clk);
        while (tile_ready_o == 1'b0) @(posedge clk);
      end
      // 결과 인덱스 push
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

    // 입력 로드
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

//    // HP-major 순서 권장: (h_rel 먼저, p_rel 나중)
//    for (h_blk = 0; h_blk < H; h_blk = h_blk + H_tile) begin
//      for (p_blk = 0; p_blk < P; p_blk = p_blk + P_tile) begin
//        for (h_rel = 0; h_rel < H_tile; h_rel = h_rel + 1) begin
//          for (p_rel = 0; p_rel < P_tile; p_rel = p_rel + 1) begin : LP_REL
//            h_abs = h_blk + h_rel;
//            p_abs = p_blk + p_rel;
//            process_one_hp(h_abs, p_abs);
//          end
//        end
//      end
//    end
    // ==== 스캔 루프 ====
    `ifndef ORDER_P_MAJOR
        // ----- HP-major (기존) : h 먼저, p 나중 -----
        for (h_blk = 0; h_blk < H; h_blk = h_blk + H_tile) begin
          for (p_blk = 0; p_blk < P; p_blk = p_blk + P_tile) begin
            for (h_rel = 0; h_rel < H_tile; h_rel = h_rel + 1) begin
              for (p_rel = 0; p_rel < P_tile; p_rel = p_rel + 1) begin
                h_abs = h_blk + h_rel;  p_abs = p_blk + p_rel;
                process_one_hp(h_abs, p_abs);
              end
            end
          end
        end
    `else
        // ----- P-major : p 먼저, h 나중 -----
        for (p_blk = 0; p_blk < P; p_blk = p_blk + P_tile) begin
          for (h_blk = 0; h_blk < H; h_blk = h_blk + H_tile) begin
            for (p_rel = 0; p_rel < P_tile; p_rel = p_rel + 1) begin
              for (h_rel = 0; h_rel < H_tile; h_rel = h_rel + 1) begin
                h_abs = h_blk + h_rel;  p_abs = p_blk + p_rel;
                process_one_hp(h_abs, p_abs);
              end
            end
          end
        end
    `endif

    // 남은 결과 드레인
    $display("[%0t] Draining %0d pending results...", $time, fifo_count);
    while (fifo_count > 0) @(posedge clk);
    #100;

    // 결과 저장
    fout = $fopen(`F_YOUT, "w");
    for (idx = 0; idx < H*P; idx = idx + 1)
      $fdisplay(fout, "%04h", y_out_mem[idx]);
    $fclose(fout);
    $display("✅ Full scan completed. Results written: %s", `F_YOUT);

//    // ---- 모니터 리포트 덤프 ----
//    u_wr_mon.dump_report();
    // ---- 리포트 덤프 (파일명 다르게) ----
    `ifndef ORDER_P_MAJOR
        u_wr_mon.dump_report(); // hp_major_wr_HP.csv 로 저장되도록 아래 인스턴스에서 설정
    `else
        u_wr_mon.dump_report(); // hp_major_wr_P.csv
    `endif

    #50 $finish;
  end

//  // -----------------------------
//  // 모니터 인스턴스 (WRITE 전용)
//  // -----------------------------
//  hp_major_monitor #(
//    .DATA_BYTES(2),
//    .BUS_BYTES(32),     // 필요시 16(=128b) 등으로 변경
//    .RUN_HIST_MAX(64)
//  ) u_wr_mon (
//    .clk(clk),
//    .rstn(rstn),
//    .ev_valid(wr_ev_valid),
//    .ev_addr(wr_ev_addr),
//    .stream_start(1'b0),
//    .stream_end(1'b0)
//  );
  // -----------------------------
  // 모니터 인스턴스 (파일명 다르게)
  // -----------------------------
    `ifndef ORDER_P_MAJOR
      // HP-major run
      hp_major_monitor #(
        .DATA_BYTES(2), .BUS_BYTES(32), .RUN_HIST_MAX(64),
        .FNAME("hp_major_wr_HP.csv"), .TAG("WR_HP")
      ) u_wr_mon (
        .clk(clk), .rstn(rstn),
        .ev_valid(wr_ev_valid_q),
        .ev_addr (wr_ev_addr_q),
        .stream_start(1'b0),
        .stream_end  (1'b0)
      );
    `else
      // P-major run
      hp_major_monitor #(
        .DATA_BYTES(2), .BUS_BYTES(32), .RUN_HIST_MAX(64),
        .FNAME("hp_major_wr_P.csv"), .TAG("WR_P")
      ) u_wr_mon (
        .clk(clk), .rstn(rstn),
        .ev_valid(wr_ev_valid_q),
        .ev_addr (wr_ev_addr_q),
        .stream_start(1'b0),
        .stream_end  (1'b0)
      );
    `endif

endmodule
