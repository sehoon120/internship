`timescale 1ns/1ps

module tb_softplus_or_exp16_pure;

  // -----------------------------
  // Parameters (DUT 파라미터와 동일하게)
  // -----------------------------
  localparam integer DW       = 16;
  localparam integer LAT_MUL  = 1;
  localparam integer LAT_ADD  = 1;
  localparam integer LAT_DIV  = 1;
  localparam integer LAT_EXP  = 12;

  localparam [DW-1:0] FP16_ONE     = 16'h3C00; // 1.0
  localparam [DW-1:0] FP16_LN2     = 16'h398C; // ln(2) ≈ 0.693147
  localparam [DW-1:0] FP16_INV_LN2 = 16'h3DC5; // 1/ln(2) ≈ 1.442695

  // -----------------------------
  // DUT I/O
  // -----------------------------
  reg                  clk;
  reg                  rstn;
  reg                  valid_i;
  reg                  mode_softplus_i; // 1: softplus, 0: exp
  reg  [DW-1:0]       x_i;

  wire [DW-1:0]       y_o_S;
  wire                valid_o_S;
  wire [DW-1:0]       y_o_e;
  wire                valid_o_e;
  wire                mode_softplus_o;

  // 에러 카운터
  integer error_cnt;

  // -----------------------------
  // Clock / Reset
  // -----------------------------
  initial begin
    clk  = 1'b0;
    forever #5 clk = ~clk;  // 100 MHz
  end

  initial begin
    rstn  = 1'b0;
    valid_i = 1'b0;
    mode_softplus_i = 1'b0;
    x_i = {DW{1'b0}};
    error_cnt = 0;
    
    repeat (5) @(posedge clk);
    rstn = 1'b1;
  end

  // -----------------------------
  // DUT
  // -----------------------------
  softplus_or_exp16 #(
    .DW(DW),
    .LAT_MUL(LAT_MUL),
    .LAT_ADD(LAT_ADD),
    .LAT_DIV(LAT_DIV),
    .LAT_EXP(LAT_EXP),
    .FP16_ONE(FP16_ONE),
    .FP16_LN2(FP16_LN2),
    .FP16_INV_LN2(FP16_INV_LN2)
  ) dut (
    .clk(clk),
    .rstn(rstn),
    .valid_i(valid_i),
    .mode_softplus_i(mode_softplus_i),
    .x_i(x_i),
    .y_o_S(y_o_S),
    .valid_o_S(valid_o_S),
    .y_o_e(y_o_e),
    .valid_o_e(valid_o_e),
    .mode_softplus_o(mode_softplus_o)
  );

  // -----------------------------
  // 출력 모니터링 (디버그 표시)
  // -----------------------------
  always @(posedge clk) begin
    if (valid_o_e) begin
      $display("[%0t] EXP  OUT: y=0x%h", $time, y_o_e);
    end
    if (valid_o_S) begin
      $display("[%0t] SOFT OUT: y=0x%h", $time, y_o_S);
    end
  end

  // -----------------------------
  // 입력 드라이브 태스크 (순수 Verilog)
  // -----------------------------
  task drive_one;
    input mode_soft;          // 1: softplus, 0: exp
    input [DW-1:0] xin;
    begin
      @(posedge clk);
      mode_softplus_i <= mode_soft;
      x_i             <= xin;
      valid_i         <= 1'b1;
      @(posedge clk);
      valid_i         <= 1'b0;  // 단발 입력
    end
  endtask

  // -----------------------------
  // 테스트 시퀀스
  // -----------------------------
  initial begin
    // 리셋 해제 대기
    wait (rstn == 1'b1);
    repeat (2) @(posedge clk);

    // 1) EXP 모드: x = +0 (0x0000) → exp(0) ≈ 1.0 근방
    $display("=== TEST 1: EXP x=+0 ===");
    drive_one(1'b0, 16'h2c04);//403f);

    // 2) SOFT 모드: x = +0 → ln2(0x398C) 정확 체크
    $display("=== TEST 2: SOFT x=+0 (expect ln2) ===");
    drive_one(1'b1, 16'h2c04);//403f);
    // 결과 대기 후 체크
    fork
      begin
        // EXP 결과는 오지 않아야 함(이 토큰은 SOFT)
        // 그냥 대기 없이 지나감
      end
      begin : wait_soft_ln2
        @(posedge clk);
        // valid 뜰 때까지 폴링
        while (valid_o_S !== 1'b1) @(posedge clk);
        if (y_o_S !== FP16_LN2) begin
          $display("[%0t] ERROR: SOFT x=0 expected ln2=0x%h, got 0x%h",
                   $time, FP16_LN2, y_o_S);
          error_cnt = error_cnt + 1;
        end else begin
          $display("[%0t] PASS : SOFT x=0 -> ln2 OK (0x%h)", $time, y_o_S);
        end
      end
    join

    // 3) EXP 모드: x = +1.0 (0x3C00) → exp(1) ≈ 2.718 (근사값이므로 값은 프린트만)
    $display("=== TEST 3: EXP x=+1.0 ===");
    drive_one(1'b0, 16'h3C00);

    // 4) SOFT 모드: x = -1.0 (0xBC00) → softplus(-1) ≈ 0.313… (값은 프린트만)
    $display("=== TEST 4: SOFT x=-1.0 ===");
    drive_one(1'b1, 16'hBC00);

    // 5) II=1 연속 주입: EXP, SOFT, EXP, SOFT (4사이클 연속)
    $display("=== TEST 5: Burst (EXP,SOFT,EXP,SOFT) ===");
    @(posedge clk);
    mode_softplus_i <= 1'b0; x_i <= 16'h3800; valid_i <= 1'b1; // EXP x≈0.5
    @(posedge clk);
    mode_softplus_i <= 1'b1; x_i <= 16'h3800;                 // SOFT x≈0.5
    @(posedge clk);
    mode_softplus_i <= 1'b0; x_i <= 16'hC000;                 // EXP x≈-2.0
    @(posedge clk);
    mode_softplus_i <= 1'b1; x_i <= 16'h3C00;                 // SOFT x=+1.0
    @(posedge clk);
    valid_i <= 1'b0;

    // 충분히 대기 후 종료
    repeat (300) @(posedge clk);

    if (error_cnt == 0) begin
      $display("=== TB RESULT: PASS (no errors) ===");
    end else begin
      $display("=== TB RESULT: FAIL (%0d errors) ===", error_cnt);
    end

    $finish;
  end

endmodule
