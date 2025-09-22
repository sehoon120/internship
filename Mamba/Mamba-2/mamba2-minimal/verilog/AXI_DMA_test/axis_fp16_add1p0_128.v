`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/22/2025 04:53:37 PM
// Design Name: 
// Module Name: axis_fp16_add1p0_128
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


module axis_fp16_add1p0_128 #
(
  parameter W = 128
)(
  input  wire           aclk,
  input  wire           aresetn,
  // slave
  input  wire [W-1:0]   s_tdata,
  input  wire           s_tvalid,
  output wire           s_tready,
  input  wire           s_tlast,
  // master
  output reg  [W-1:0]   m_tdata,
  output reg            m_tvalid,
  input  wire           m_tready,
  output reg            m_tlast
);
  // 간단: 단일 사이클 파이프(ready=1) - 실제론 FP16 adder IP latency 고려해 파이프라인
  assign s_tready = m_tready;  // back-to-back 통과

  // FP16 +1.0 연산기 (자리표시자)
  function [15:0] f16_add1 (input [15:0] x);
    // 실제로는 FP16 Add IP 인스턴스 권장.
    // 시뮬 전용 근사: NaN/Inf 무시, 정수 0..N 범위만 테스트라면 LUT 변환 테이블 가능.
    // 여기선 간단히 0x3C00 더하기가 안 되므로, TB 단계에선 "미리 +1 된 기대값"과 비교만 권장.
    f16_add1 = 16'hDEAD; // placeholder
  endfunction

  integer i;
  reg [15:0] half [0:7];
  reg [15:0] half_o [0:7];

  always @(*) begin
    for (i=0;i<8;i=i+1) begin
      half[i] = s_tdata[i*16 +: 16];
      // 연산기 대신 TB 초기 버전은 bypass 해도 좋음. 일단 패스스루:
      half_o[i] = half[i]; // 나중에 FP16 add IP 연결로 교체
    end
  end

  always @(posedge aclk) begin
    if (!aresetn) begin
      m_tdata  <= {W{1'b0}};
      m_tvalid <= 1'b0;
      m_tlast  <= 1'b0;
    end else begin
      if (s_tvalid && m_tready) begin
        for (i=0;i<8;i=i+1) m_tdata[i*16 +: 16] <= half_o[i];
        m_tvalid <= 1'b1;
        m_tlast  <= s_tlast;
      end else if (m_tready) begin
        m_tvalid <= 1'b0;
      end
    end
  end
endmodule


