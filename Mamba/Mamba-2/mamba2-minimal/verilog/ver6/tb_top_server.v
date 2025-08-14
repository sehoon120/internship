`timescale 1ns / 1ps

// 전체 데이터(B=1,H=24,P=64,N=128)를 H_tile×P_tile 블로킹 순서로 모두 순회하여
// (h,p)마다 N=128을 16씩(=8타일) 스트리밍 → SSMBLOCK_TOP → y_final 수집 TB
// 고쳐야할것: 결과 인덱스에 맞춰서 저장하기. 지금은 그 과정이 없음
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
  localparam integer N_TILE  = 16;
  localparam integer TILES   = N / N_TILE;  // 8

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
  reg  [DW-1:0]            dA_i;
  reg  [DW-1:0]            x_i;
  reg  [DW-1:0]            D_i;

  reg  [N_TILE*DW-1:0]     B_tile_i;
  reg  [N_TILE*DW-1:0]     C_tile_i;
  reg  [N_TILE*DW-1:0]     hprev_tile_i;

  wire [DW-1:0]            y_final_o;
  wire                     y_final_valid_o;

  // -----------------------------
  // Memories
  // -----------------------------
  reg [DW-1:0] dt_mem   [0:B*H-1];         // dt[h]
  reg [DW-1:0] dA_mem   [0:B*H-1];         // dA[h]
  reg [DW-1:0] D_mem    [0:H-1];           // D[h]
  reg [DW-1:0] x_mem    [0:B*H*P-1];       // x[h*P + p]
  reg [DW-1:0] B_mem    [0:B*N-1];         // B[n]
  reg [DW-1:0] C_mem    [0:B*N-1];         // C[n]
  reg [DW-1:0] h_mem    [0:B*H*P*N-1];     // h_prev[((h*P)+p)*N + n]

  // 결과 버퍼 (H*P 개의 스칼라)
  reg [DW-1:0] y_out_mem [0:H*P-1];

  integer h_blk, p_blk, h_rel, p_rel, h_abs, p_abs;
  integer t, j, base, fout;

  // -----------------------------
  // DUT
  // -----------------------------
  SSMBLOCK_TOP #(
      .DW(DW), .N_TILE(N_TILE), .N_TOTAL(N),
      .LAT_DX_M(6), .LAT_DBX_M(6), .LAT_DAH_M(6),
      .LAT_ADD_A(11), .LAT_HC_M(6)
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
  // Helpers
  // -----------------------------
  // 타일 한 장(N_TILE)을 버스로 패킹해서 1사이클 펄스로 전송
  task send_one_tile(input integer n_base);
    begin
      for (j = 0; j < N_TILE; j = j + 1) begin
        B_tile_i     [DW*j +: DW] = B_mem[n_base + j];
        C_tile_i     [DW*j +: DW] = C_mem[n_base + j];
        hprev_tile_i [DW*j +: DW] = h_mem[((h_abs*P) + p_abs)*N + (n_base + j)];
      end
      // 핸드셰이크
      @(posedge clk);
      while (tile_ready_o == 1'b0) @(posedge clk);
      tile_valid_i <= 1'b1;
      @(posedge clk);
      tile_valid_i <= 1'b0;
    end
  endtask

  // (h_abs, p_abs) 그룹 처리: 스칼라 세팅 → 8타일 연속 전송 → y_final 수신
    task process_one_hp(input integer h_a, input integer p_a);
      integer t, j, base;
      begin
        // 스칼라 세팅 (바뀌자마자 첫 타일과 함께 사용됨)
        dt_i <= dt_mem[h_a];
        dA_i <= dA_mem[h_a];
        D_i  <= D_mem[h_a];
        x_i  <= x_mem[h_a*P + p_a];
    
        // === 8사이클 연속 주입 (II=1) ===
        tile_valid_i <= 1'b1;
        for (t = 0; t < (N/N_TILE); t = t + 1) begin
          base = t * N_TILE;
    
          // 필요시 백프레셔: while (!tile_ready_o) @(posedge clk);
    
          // 이번 타일 payload 패킹
          for (j = 0; j < N_TILE; j = j + 1) begin
            B_tile_i     [DW*j +: DW] = B_mem[base + j];
            C_tile_i     [DW*j +: DW] = C_mem[base + j];
            hprev_tile_i [DW*j +: DW] = h_mem[((h_a*P) + p_a)*N + (base + j)];
          end
    
          @(posedge clk); // 다음 타일로 즉시 진행 (valid 유지)
        end
        tile_valid_i <= 1'b0;
      end
    endtask
    


  // -----------------------------
  // Reset / Load / Full Scan
  // -----------------------------
  initial begin
    rstn = 1'b0;
    tile_valid_i = 1'b0;
    dt_i = 0; dA_i = 0; D_i = 0; x_i = 0;
    B_tile_i = 0; C_tile_i = 0; hprev_tile_i = 0;

    // 파일 경로는 환경에 맞게 수정
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",        dt_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",        dA_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",         D_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",         x_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",         B_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",         C_mem);
    $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex", h_mem);

    // Reset release
    #100 rstn = 1'b1;
    @(posedge clk); @(posedge clk);

    $display("==== Full scan start: H=%0d, P=%0d, N=%0d (H_tile=%0d, P_tile=%0d, N_TILE=%0d) ====",
             H, P, N, H_tile, P_tile, N_TILE);

    // 블로킹 순회: (h_blk, p_blk) 타일 블록 → 내부에서 (h_rel, p_rel) 순회
    for (h_blk = 0; h_blk < H; h_blk = h_blk + H_tile) begin
      for (p_blk = 0; p_blk < P; p_blk = p_blk + P_tile) begin
        for (h_rel = 0; h_rel < H_tile; h_rel = h_rel + 1) begin
          for (p_rel = 0; p_rel < P_tile; p_rel = p_rel + 1) begin
            h_abs = h_blk + h_rel;
            p_abs = p_blk + p_rel;
            process_one_hp(h_abs, p_abs);
          end
        end
      end
    end

    // 결과 저장
    fout = $fopen("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out_full.hex", "w");
    for (integer idx = 0; idx < H*P; idx = idx + 1)
      $fdisplay(fout, "%04h", y_out_mem[idx]);
    $fclose(fout);
    $display("✅ Full scan completed. Results written.");

    #50 $finish;
  end

endmodule
