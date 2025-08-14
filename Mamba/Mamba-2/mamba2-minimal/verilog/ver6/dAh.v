// 16 병렬

module dAh #(
  parameter integer DW       = 16,
  parameter integer N_TILE   = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 valid_i,
  input  wire [DW-1:0]        dA_i,
  input  wire [N_TILE*DW-1:0] hprev_i,
  output wire [N_TILE*DW-1:0] dAh_o,
  output wire                 valid_o
);
  wire [DW-1:0] y   [N_TILE];
  wire          vld [N_TILE];

  genvar n;
  generate
    for (n=0; n<N_TILE; n++) begin : g_mul
      fp16_mult_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(dA_i),
        .b(hprev_i[n*DW +: DW]),
        .result(y[n]),
        .valid_out(vld[n])
      );
      assign dBx_o[n*DW +: DW] = y[n];
    end
  endgenerate

  assign valid_o = vld[0];
endmodule
