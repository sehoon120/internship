// dt + dt_bias

// N 방향 없음

module delta #(
  parameter integer DW       = 16,
  parameter integer ADD_LAT  = 11
)(
  input  wire          clk,
  input  wire          rstn,
  input  wire          valid_i,
  input  wire [DW-1:0] dt_i,
  input  wire [DW-1:0] dt_bias_i,
  output wire [DW-1:0] delta_o,
  output wire          valid_o
);
    fp16_add_wrapper u_add (
        .clk(clk),
        .valid_in(valid_i),
        .a(dt_i),
        .b(dt_bias_i),
        .result(delta_o),
        .valid_out(valid_o)
    );
endmodule
