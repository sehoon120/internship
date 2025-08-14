// 16 병렬

module hC #(
  parameter integer DW       = 16,
  parameter integer N_TILE   = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 valid_i,
  input  wire [N_TILE*DW-1:0] hnext_i,
  input  wire [N_TILE*DW-1:0] C_i,
  output wire [N_TILE*DW-1:0] hC_o,
  output wire                 valid_o
);
  wire [DW-1:0] y   [0:N_TILE-1];
  wire          vld [0:N_TILE-1];

  genvar n;
  generate
    for (n=0; n<N_TILE; n=n+1) begin : g_mul
      fp16_mult_wrapper u_mul (
        .clk(clk), 
        .valid_in(valid_i),
        .a(hnext_i[n*DW +: DW]),
        .b(C_i[n*DW +: DW]),
        .result(y[n]),
        .valid_out(vld[n])
      );
      assign hC_o[n*DW +: DW] = y[n];
    end
  endgenerate

  assign valid_o = vld[0];
endmodule
