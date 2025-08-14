// 16 병렬

module h_next #(
  parameter integer DW       = 16,
  parameter integer N_TILE   = 16,
  parameter integer ADD_LAT  = 11
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 valid_i,
  input  wire [N_TILE*DW-1:0] dBx_i,
  input  wire [N_TILE*DW-1:0] dAh_i,
  output wire [N_TILE*DW-1:0] hnext_o,
  output wire                 valid_o
);
  wire [DW-1:0] y   [0:N_TILE-1];
  wire          vld [0:N_TILE-1];

  genvar n;
  generate
    for (n=0; n<N_TILE; n=n+1) begin : g_add
      fp16_add_wrapper u_mul (
        .clk(clk),
        .valid_in(valid_i),
        .a(dBx_i[n*DW +: DW]),
        .b(dAh_i[n*DW +: DW]),
        .result(y[n]),
        .valid_out(vld[n])
      );
      assign hnext_o[n*DW +: DW] = y[n];
    end
  endgenerate

  assign valid_o = vld[0];
endmodule
