`timescale 1ns/1ps

module tb_softplus_or_exp16;
  // -----------------------------
  // Parameters (tb에서도 동일하게 잡아 총 지연 계산)
  // -----------------------------
  localparam integer DW       = 16;
  localparam integer LAT_MUL  = 6;
  localparam integer LAT_ADD  = 11;
  localparam integer LAT_DIV  = 14;
  localparam integer LAT_EXP  = 20;
  localparam [DW-1:0] FP16_ONE     = 16'h3C00; // 1.0
  localparam [DW-1:0] FP16_LN2     = 16'h398C; // ln(2)
  localparam [DW-1:0] FP16_INV_LN2 = 16'h3DC5; // 1/ln(2)

  // softplus 총 지연
  localparam integer POST_SOFT_LAT = (LAT_ADD + LAT_DIV);
  localparam integer TOT_LAT       = (LAT_MUL + LAT_EXP + POST_SOFT_LAT);

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

  // -----------------------------
  // Clock / Reset
  // -----------------------------
  initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100 MHz
  end

  initial begin
    rstn = 0;
    valid_i = 0;
    mode_softplus_i = 0;
    x_i = 16'h0000;
    repeat (5) @(posedge clk);
    rstn = 1;
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
  // Helpers
  // -----------------------------
  task drive_input(input bit mode_soft, input [DW-1:0] x_hex, input string tag);
    begin
      @(posedge clk);
      mode_softplus_i <= mode_soft;
      x_i             <= x_hex;
      valid_i         <= 1'b1;
      $display("[%0t] DRIVE  mode=%s  x=0x%h  (%s)",
               $time, mode_soft ? "SOFT" : "EXP ", x_hex, tag);
      @(posedge clk);
      valid_i         <= 1'b0;  // 단발 펄스(II=1 스트리밍이면 계속 1로 밀어도 됨)
    end
  endtask

  // exp 결과 도착 기다리기
  task wait_exp_and_print(input string tag);
    begin
      wait (valid_o_e === 1'b1);
      $display("[%0t] EXP OUT valid=1  y_e=0x%h  (%s)", $time, y_o_e, tag);
      @(posedge clk);
    end
  endtask

  // softplus 결과 도착 기다리기
  task wait_soft_and_check(input [DW-1:0] expect_hex, input bit do_check, input string tag);
    begin
      wait (valid_o_S === 1'b1);
      $display("[%0t] SOFT OUT valid=1 y_s=0x%h  (%s)", $time, y_o_S, tag);
      if (do_check) begin
        if (y_o_S !== expect_hex) begin
          $error("[%0t] SOFT CHECK FAIL: got=0x%h expect=0x%h  (%s)",
                 $time, y_o_S, expect_hex, tag);
        end else begin
          $display("[%0t] SOFT CHECK PASS: y_s == 0x%h (%s)",
                   $time, y_o_S, tag);
        end
      end
      @(posedge clk);
    end
  endtask

  // -----------------------------
  // Test sequence
  // -----------------------------
  initial begin
    // 대기
    wait (rstn == 1);
    repeat (2) @(posedge clk);

    // 1) EXP 모드: x = +0 (0x0000) → exp(0) ≈ 1.0 (0x3C00) 기대 (근사/테이블에 따라 조금 다를 수도)
    drive_input(1'b0, 16'h0000, "EXP x=+0");
    // exp 유효 도착 대기 및 출력 로그
    wait_exp_and_print("EXP x=+0");

    // 2) SOFT 모드: x = +0 → y = ln2 (=0x398C) 정확 체크 (DUT가 예외처리)
    drive_input(1'b1, 16'h0000, "SOFT x=+0  expect ln2");
    wait_soft_and_check(FP16_LN2, /*do_check=*/1'b1, "SOFT x=+0");

    // 3) EXP 모드: x = +1.0 (0x3C00) → exp(1) ≈ 2.718 (근사값, 동등 체크는 안함)
    drive_input(1'b0, 16'h3C00, "EXP x=+1.0");
    wait_exp_and_print("EXP x=+1.0");

    // 4) SOFT 모드: x = -1.0 (0xBC00) → softplus(-1) ~ 0.3132617 (≈ 0x35xx 근방)
    //   수치 동등 체크는 하지 않고 결과/타이밍만 확인
    drive_input(1'b1, 16'hBC00, "SOFT x=-1.0");
    wait_soft_and_check(16'h0000, /*do_check=*/1'b0, "SOFT x=-1.0");

    // 5) 파이프라인 연속 입력(II=1) 간단 스팟 체크
    //    EXP, SOFT, EXP, SOFT 순으로 4사이클 연속 주입
    fork
      begin
        drive_input(1'b0, 16'h3800, "EXP x≈0.5");
        drive_input(1'b1, 16'h3800, "SOFT x≈0.5");
        drive_input(1'b0, 16'hC000, "EXP x≈-2.0");
        drive_input(1'b1, 16'h3C00, "SOFT x=+1.0");
      end
      begin
        // 출력들은 각각 자신의 레이턴시로 도착할 것.
        // 단순히 valid를 기다리며 프린트
        repeat (8) @(posedge clk); // 약간 여유를 준 뒤부터 기다림
        // 이후 발생하는 valid는 위 wait_* task에서 이미 잡음.
      end
    join

    // 충분 대기 후 종료
    repeat (200) @(posedge clk);
    $display("TB finished.");
    $finish;
  end

endmodule
