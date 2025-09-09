module div_fp16 #(parameter DW=16, LAT=14)(
    input clk, input rstn, input valid_i,
    input [DW-1:0] a_i, b_i,
    output [DW-1:0] y_o, output valid_o
);
    // Replace with your FP16 div IP wrapper.
endmodule