module fp16_add #(parameter DW=16, LAT=11)(
    input clk, input rstn, input valid_i,
    input [DW-1:0] a_i, b_i,
    output [DW-1:0] sum_o, output valid_o
);
    // Replace with your FP16 subtract IP wrapper.
    floating_point_0 u_fp16_add (
        .aclk(clk),
        .s_axis_a_tvalid(valid_i),
        .s_axis_a_tdata(a_i),
        .s_axis_b_tvalid(valid_i),
        .s_axis_b_tdata(b_i),
        .m_axis_result_tvalid(valid_o),
        .m_axis_result_tdata(sum_o)
    );
endmodule