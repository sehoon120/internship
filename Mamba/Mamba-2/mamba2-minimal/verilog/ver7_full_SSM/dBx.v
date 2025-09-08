// 1*1*1*16*16으로 들어와서 N 방향 parallel 처리


module dBx #(
  parameter integer DW       = 16,
  parameter integer N_TILE   = 16,
  parameter integer MUL_LAT  = 6
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 valid_i,
  input  wire [DW-1:0]        dx_i,
  input  wire [N_TILE*DW-1:0] Bmat_i,   // [n][DW]
  output wire [N_TILE*DW-1:0] dBx_o,    // [n][DW]
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
        .a(dx_i),
        .b(Bmat_i[n*DW +: DW]),
        .result(y[n]),
        .valid_out(vld[n])
      );
      assign dBx_o[n*DW +: DW] = y[n];
    end
  endgenerate

  // 모든 lane이 동일 LAT를 가지므로 하나만 사용
  assign valid_o = vld[0];
endmodule
