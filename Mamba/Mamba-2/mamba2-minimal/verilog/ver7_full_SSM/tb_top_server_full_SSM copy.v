// tb_ssmblock_top.v  (Pure Verilog-2001)
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

module tb_ssmblock_top;

  // ===== DUT params =====
  parameter integer DW       = 16;
  parameter integer H_TILE   = 1;
  parameter integer P_TILE   = 1;
  parameter integer N_TILE   = 64;
  parameter integer N_TOTAL  = 128;

  // Latencies (DUT와 동일하게 맞추세요)
  parameter integer LAT_DX_M  = 6;
  parameter integer LAT_DBX_M = 6;
  parameter integer LAT_DAH_M = 6;
  parameter integer LAT_ADD_A = 11;
  parameter integer LAT_HC_M  = 6;
  parameter integer LAT_MUL   = 6;
  parameter integer LAT_ADD   = 11;
  parameter integer LAT_DIV   = 17;
  parameter integer LAT_EXP   = 6 + LAT_MUL*3 + LAT_ADD*3;
  parameter integer LAT_SP    = LAT_EXP + LAT_MUL + LAT_ADD + LAT_DIV + 1;

  // Derived
  parameter integer TILES_PER_GROUP = (N_TOTAL + N_TILE - 1) / N_TILE;

  // ===== Clock / Reset =====
  reg clk = 0;
  always #5 clk = ~clk; // 100 MHz
  reg rstn;

  // ===== DUT I/O =====
  reg                               tile_valid_i;
  wire                              tile_ready_o; // 1'b1 가정
  reg  [H_TILE*DW-1:0]              dt_i;
  reg  [H_TILE*DW-1:0]              dt_bias_i;
  reg  [H_TILE*DW-1:0]              A_i;
  reg  [H_TILE*P_TILE*DW-1:0]       x_i;
  reg  [H_TILE*DW-1:0]              D_i;

  reg  [N_TILE*DW-1:0]              B_tile_i;
  reg  [N_TILE*DW-1:0]              C_tile_i;
  reg  [H_TILE*P_TILE*N_TILE*DW-1:0] hprev_tile_i;

  wire [H_TILE*P_TILE*DW-1:0]       y_final_o;
  wire                              y_final_valid_o;

  // ===== Memories for inputs =====
  reg [DW-1:0] mem_dt     [0:H_TILE-1];
  reg [DW-1:0] mem_dt_b   [0:H_TILE-1];
  reg [DW-1:0] mem_A      [0:H_TILE-1];
  reg [DW-1:0] mem_x      [0:H_TILE*P_TILE-1];
  reg [DW-1:0] mem_D      [0:H_TILE-1];
  reg [DW-1:0] mem_B      [0:N_TOTAL-1];
  reg [DW-1:0] mem_C      [0:N_TOTAL-1];
  reg [DW-1:0] mem_hprev  [0:H_TILE*P_TILE*N_TOTAL-1];

  // ===== DUT inst =====
  SSMBLOCK_TOP #(
    .DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE), .N_TILE(N_TILE), .N_TOTAL(N_TOTAL),
    .LAT_DX_M(LAT_DX_M), .LAT_DBX_M(LAT_DBX_M), .LAT_DAH_M(LAT_DAH_M),
    .LAT_ADD_A(LAT_ADD_A), .LAT_HC_M(LAT_HC_M),
    .LAT_MUL(LAT_MUL), .LAT_ADD(LAT_ADD), .LAT_DIV(LAT_DIV),
    .LAT_EXP(LAT_EXP), .LAT_SP(LAT_SP)
  ) dut (
    .clk(clk), .rstn(rstn),
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

  // ===== Index helpers (pure Verilog) =====
  function integer idx_hpn_full;
    input integer h, p, n;
    begin
      idx_hpn_full = ((h*P_TILE)+p)*N_TOTAL + n;
    end
  endfunction

  function integer idx_hpn_tile;
    input integer h, p, n;
    begin
      idx_hpn_tile = ((h*P_TILE)+p)*N_TILE + n;
    end
  endfunction

  // ===== File I/O =====
  integer fout;

  // ===== Tasks =====
  task load_hex_files;
    begin
      $display("[TB] Loading hex files ...");
      $readmemh(`F_DT,     mem_dt);
      $readmemh(`F_DTBIAS, mem_dt_b);
      $readmemh(`F_A,      mem_A);
      $readmemh(`F_X,      mem_x);
      $readmemh(`F_D,      mem_D);
      $readmemh(`F_B,      mem_B);
      $readmemh(`F_C,      mem_C);
      $readmemh(`F_HPREV,  mem_hprev);
      $display("[TB] Done.");
    end
  endtask

  task pack_scalar_vectors; // (h), (hp)
    integer h, p, hp;
    begin
      // (h)
      for (h=0; h<H_TILE; h=h+1) begin
        dt_i     [DW*(h+1)-1 -: DW] = mem_dt  [h];
        dt_bias_i[DW*(h+1)-1 -: DW] = mem_dt_b[h];
        A_i      [DW*(h+1)-1 -: DW] = mem_A   [h];
        D_i      [DW*(h+1)-1 -: DW] = mem_D   [h];
      end
      // (hp)
      for (h=0; h<H_TILE; h=h+1) begin
        for (p=0; p<P_TILE; p=p+1) begin
          hp = h*P_TILE + p;
          x_i[DW*(hp+1)-1 -: DW] = mem_x[hp];
        end
      end
    end
  endtask

  task pack_one_tile;
    input integer tile_idx; // 0..TILES_PER_GROUP-1
    integer t, h, p, n, n_global, hp, idx_full, idx_tile;
    reg [DW-1:0] v;
    begin
      // B,C: (n=0..N_TILE-1) with global offset
      for (t=0; t<N_TILE; t=t+1) begin
        n_global = tile_idx*N_TILE + t;
        if (n_global < N_TOTAL) begin
          B_tile_i[DW*(t+1)-1 -: DW] = mem_B[n_global];
          C_tile_i[DW*(t+1)-1 -: DW] = mem_C[n_global];
        end else begin
          B_tile_i[DW*(t+1)-1 -: DW] = {DW{1'b0}};
          C_tile_i[DW*(t+1)-1 -: DW] = {DW{1'b0}};
        end
      end

      // hprev_tile_i: (h,p,n=0..N_TILE-1) with same global offset
      for (h=0; h<H_TILE; h=h+1) begin
        for (p=0; p<P_TILE; p=p+1) begin
          for (n=0; n<N_TILE; n=n+1) begin
            n_global = tile_idx*N_TILE + n;
            idx_tile = idx_hpn_tile(h,p,n);
            if (n_global < N_TOTAL) begin
              idx_full = idx_hpn_full(h,p,n_global);
              v = mem_hprev[idx_full];
            end else begin
              v = {DW{1'b0}};
            end
            hprev_tile_i[DW*(idx_tile+1)-1 -: DW] = v;
          end
        end
      end
    end
  endtask

  task write_y_final_to_file;
    integer h, p, hp;
    reg [DW-1:0] word;
    begin
      for (h=0; h<H_TILE; h=h+1) begin
        for (p=0; p<P_TILE; p=p+1) begin
          hp = h*P_TILE + p;
          word = y_final_o[DW*(hp+1)-1 -: DW];
          $fdisplay(fout, "%04h", word);
        end
      end
    end
  endtask

  // ===== Main =====
  integer g, t;
  initial begin
    // init
    rstn = 0;
    tile_valid_i = 0;
    dt_i = {H_TILE*DW{1'b0}};
    dt_bias_i = {H_TILE*DW{1'b0}};
    A_i = {H_TILE*DW{1'b0}};
    x_i = {H_TILE*P_TILE*DW{1'b0}};
    D_i = {H_TILE*DW{1'b0}};
    B_tile_i = {N_TILE*DW{1'b0}};
    C_tile_i = {N_TILE*DW{1'b0}};
    hprev_tile_i = {H_TILE*P_TILE*N_TILE*DW{1'b0}};

    // load & open
    load_hex_files();
    fout = $fopen(`F_YOUT, "w");
    if (fout == 0) begin
      $display("[TB][ERROR] Failed to open output file.");
      $finish;
    end

    // reset
    repeat (10) @(posedge clk);
    rstn = 1;
    repeat (2) @(posedge clk);

    // (h),(hp) 고정 파라미터 패킹 (그룹 내 불변 가정)
    pack_scalar_vectors();

    // ===== 타일별 순회 =====
    // 한 그룹 (= N_TOTAL 전 범위)을 tile 단위로 쪼개어 순회
    for (t = 0; t < TILES_PER_GROUP; t = t + 1) begin
      pack_one_tile(t);
      // 이 타일을 한 싸이클 동안 valid
      @(posedge clk);
      tile_valid_i <= 1'b1;
      @(posedge clk);
      tile_valid_i <= 1'b0;
      // 필요하면 타일 사이 idle 삽입 가능:
      // @(posedge clk);
    end

    // 마지막 타일 처리 후 파이프라인 지연 동안 대기 → valid 잡기
    wait (y_final_valid_o === 1'b1);
    write_y_final_to_file();
    $display("[TB] y_final captured and written to file.");

    $fclose(fout);
    repeat (20) @(posedge clk);
    $finish;
  end

endmodule
