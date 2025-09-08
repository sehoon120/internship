// xD[b*H*P + h*H*P + p] = D[h] × x[b*H*P + h*H*P + p];
// xD.v
// xD = x * D

module xD #(
  parameter integer DW       = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire          clk,
  input  wire          rstn,
  input  wire          valid_i,
  input  wire [DW-1:0] x_i,
  input  wire [DW-1:0] D_i,
  output wire [DW-1:0] xD_o,
  output wire          valid_o
);
    fp16_mult_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(x_i),
        .b(D_i),
        .result(xD_o),
        .valid_out(valid_o)
    );
endmodule
