module fp16_add #(parameter DW=16, LAT=11)(
    input clk, input rstn, input valid_i,
    input [DW-1:0] a_i, b_i,
    output [DW-1:0] sum_o, output valid_o
);
    // Replace with your FP16 subtract IP wrapper.
endmodule