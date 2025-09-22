`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/22/2025 05:00:56 PM
// Design Name: 
// Module Name: tb_top_dma_path
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



// tb_top_dma_path.v
`timescale 1ns/1ps
module tb_top_dma_path;
  reg clk=0, rstn=0;
  always #5 clk=~clk; // 100MHz

  localparam ADDR_W=8;  // 예: 256 words * 16B = 4KB
  reg        go_mm2s, go_s2mm;
  reg [31:0] byte_len;
  reg [ADDR_W-1:0] in_base, out_base;
  wire mm2s_done, s2mm_done;

  top_dma_path #(.ADDR_W(ADDR_W)) dut(
    .clk(clk), .rstn(rstn),
    .go_mm2s(go_mm2s), .go_s2mm(go_s2mm),
    .byte_len(byte_len), .in_base(in_base), .out_base(out_base),
    .mm2s_done(mm2s_done), .s2mm_done(s2mm_done)
  );

  // BRAM에 초기 데이터 써넣기: dut.u_bram.mem[...] 직접 접근 (시뮬 전용)
  integer i;
  initial begin
    rstn=0; go_mm2s=0; go_s2mm=0;
    byte_len = 32'd256; // 16B * 16 beats
    in_base  = 'h00;
    out_base = 'h80;
    repeat(10) @(posedge clk);
    rstn=1;

    // 입력 채우기 (FP16 0..7 반복)
    for (i=0;i< (byte_len/16); i=i+1) begin
      dut.u_bram.mem[in_base + i] = {
        16'h0007,16'h0006,16'h0005,16'h0004,
        16'h0003,16'h0002,16'h0001,16'h0000
      };
    end

    // S2MM 먼저 Go, 그다음 MM2S Go (일반 DMA 시퀀스)
    @(posedge clk); go_s2mm<=1; @(posedge clk); go_s2mm<=0;
    @(posedge clk); go_mm2s<=1; @(posedge clk); go_mm2s<=0;

    wait(mm2s_done && s2mm_done);
    // 출력 확인
    $display("Check OUT...");
    for (i=0;i< (byte_len/16); i=i+1) begin
      $display("%0d: %h", i, dut.u_bram.mem[out_base + i]);
    end
    $finish;
  end
endmodule
