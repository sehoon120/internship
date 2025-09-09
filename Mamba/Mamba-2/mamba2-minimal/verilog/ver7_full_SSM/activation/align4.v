module align2 #(parameter integer DW=16)(
  input wire clk, input wire rstn,
  input wire [DW-1:0] a_i, input wire va_i,
  input wire [DW-1:0] b_i, input wire vb_i,
  output reg [DW-1:0] a_o, output reg [DW-1:0] b_o,
  output reg v_o
);
  // 간단히 한쪽을 1-2단 지연시켜 맞추는 용도로 수정해 사용하세요.
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin a_o<=0; b_o<=0; v_o<=1'b0; end
    else begin
      a_o <= a_i; b_o <= b_i; v_o <= va_i & vb_i;
    end
  end
endmodule

module align4 #(parameter integer DW=16)(
  input wire clk, input wire rstn,
  input wire [DW-1:0] a_i, input wire va_i,
  input wire [DW-1:0] b_i, input wire vb_i,
  input wire [DW-1:0] c_i, input wire vc_i,
  input wire [DW-1:0] d_i, input wire vd_i,
  output reg [DW-1:0] a_o, output reg [DW-1:0] b_o,
  output reg [DW-1:0] c_o, output reg [DW-1:0] d_o,
  output reg v_o
);
  always @(posedge clk or negedge rstn) begin
    if(!rstn) begin a_o<=0; b_o<=0; c_o<=0; d_o<=0; v_o<=1'b0; end
    else begin
      a_o<=a_i; b_o<=b_i; c_o<=c_i; d_o<=d_i; v_o<=va_i & vb_i & vc_i & vd_i;
    end
  end
endmodule
