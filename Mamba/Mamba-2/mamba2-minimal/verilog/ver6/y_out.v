// y_out[b*H*P + h*H*P + p] = y_in[b*H*P + h*H*P + p] + D[h] × xD[b*H*P + h*H*P + p];
// y_out.v
// y_out = y_in + xD

module y_out #(
  parameter integer DW       = 16,
  parameter integer ADD_LAT  = 11
)(
  input  wire          clk,
  input  wire          rstn,
  input  wire          valid_i,
  input  wire [DW-1:0] ytmp_i,
  input  wire [DW-1:0] xD_i,
  output wire [DW-1:0] y_o,
  output wire          valid_o
);
    fp16_add_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(ytmp_i),
        .b(xD_i),
        .result(y_o),
        .valid_out(valid_o)
    );
endmodule
