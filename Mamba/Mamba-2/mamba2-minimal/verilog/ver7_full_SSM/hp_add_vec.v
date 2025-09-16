// hp_add_vec : element-wise FP16 add over (H_TILE*P_TILE) lanes
// Verilog-2001 only (no SV). Assumes fp16_add_wrapper has throughput=1.
module hp_add_vec #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1
)(
  input  wire                              clk,
  input  wire                              rstn,
  input  wire                              valid_i,     // fire add this cycle
  input  wire [H_TILE*P_TILE*DW-1:0]       a_i,         // (hp)
  input  wire [H_TILE*P_TILE*DW-1:0]       b_i,         // (hp)
  output wire [H_TILE*P_TILE*DW-1:0]       y_o,         // (hp)
  output wire                              valid_o
);
  genvar hp;
  wire v0;

  generate
    for (hp = 0; hp < H_TILE*P_TILE; hp = hp + 1) begin : G
      wire [DW-1:0] a_w = a_i[DW*(hp+1)-1 -: DW];
      wire [DW-1:0] b_w = b_i[DW*(hp+1)-1 -: DW];
      wire [DW-1:0] y_w;
      wire          v_w;

      fp16_add_wrapper u_add (
        .clk       (clk),
        .valid_in  (valid_i),
        .a         (a_w),
        .b         (b_w),
        .result    (y_w),
        .valid_out (v_w)
      );

      assign y_o[DW*(hp+1)-1 -: DW] = y_w;
      if (hp == 0) begin : GV
        assign v0 = v_w;   // 모든 래인이 동일 레이턴시 가정
      end
    end
  endgenerate

  assign valid_o = v0;

endmodule
