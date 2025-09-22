`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/22/2025 05:00:35 PM
// Design Name: 
// Module Name: axis_passthru_128
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


// axis_fp16_add1p0_128.v : 128b(= FP16×8) 스트림 입력에 대해 각 요소 +1.0
module axis_fp16_add1p0_128(
  input  wire        aclk, aresetn,
  // in
  input  wire [127:0] s_tdata,
  input  wire         s_tvalid,
  output wire         s_tready,
  input  wire         s_tlast,
  // out
  output reg  [127:0] m_tdata,
  output reg          m_tvalid,
  input  wire         m_tready,
  output reg          m_tlast
);
  wire [15:0] xin [0:7];
  reg  [15:0] xreg[0:7];
  wire [15:0] yout[0:7];

  genvar i;
  generate
    for (i=0;i<8;i=i+1) begin : UNPACK
      assign xin[i] = s_tdata[i*16 +: 16];
    end
  endgenerate

  // 한 사이클 파이프라인: 입력 래치 → fp16_add1 조합 → 출력 래치
  // (fp16_add1이 조합이므로 타이밍 안나오면 파이프 더 넣기)
  always @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      xreg[0]<=0; xreg[1]<=0; xreg[2]<=0; xreg[3]<=0;
      xreg[4]<=0; xreg[5]<=0; xreg[6]<=0; xreg[7]<=0;
    end else if (s_tvalid && s_tready) begin
      xreg[0]<=xin[0]; xreg[1]<=xin[1]; xreg[2]<=xin[2]; xreg[3]<=xin[3];
      xreg[4]<=xin[4]; xreg[5]<=xin[5]; xreg[6]<=xin[6]; xreg[7]<=xin[7];
    end
  end

    wire v_in = s_tvalid && s_tready;
    wire v_out;
  // 조합 가산기 8개
  generate
    for (i=0;i<8;i=i+1) begin : ADDERS
      fp16_add_wrapper add(
        .clk(aclk),
        .a(xreg[i]),
        .b(16'h3C00),
        .valid_in(v_in),
        .result(yout[i]),
        .valid_out(v_out)
      );
    end
  endgenerate

  // 핸드셰이크: 1-stage throughput=1(ready直結), latency=1 beat
  assign s_tready = m_tready;

  integer k;
  always @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      m_tdata<=0; m_tvalid<=0; m_tlast<=0;
    end else begin
      if (s_tvalid && s_tready) begin
        for (k=0;k<8;k=k+1) m_tdata[k*16 +: 16] <= yout[k];
        m_tvalid <= 1'b1;
        m_tlast  <= s_tlast;
      end else if (m_tready) begin
        m_tvalid <= 1'b0;
      end
    end
  end
endmodule

