// dt*x

// N 방향 없음

module dx #(
  parameter integer DW       = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire          clk,
  input  wire          rstn,
  input  wire          valid_i,
  input  wire [DW-1:0] dt_i,
  input  wire [DW-1:0] x_i,
  output wire [DW-1:0] dx_o,
  output wire          valid_o
);
    fp16_mult_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(dt_i),
        .b(x_i),
        .result(dx_o),
        .valid_out(valid_o)
    );
endmodule
