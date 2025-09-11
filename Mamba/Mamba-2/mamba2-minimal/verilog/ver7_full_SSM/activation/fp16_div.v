module div_fp16 #(parameter DW=16, LAT=14)(
    input clk, input rstn, input valid_i,
    input [DW-1:0] a_i, b_i,
    output [DW-1:0] y_o, output valid_o
);
    // Replace with your FP16 div IP wrapper.
    floating_point_2 u_fp16_div (
        .aclk(clk),
        .s_axis_a_tvalid(valid_i),
        .s_axis_a_tdata(a_i),
        .s_axis_b_tvalid(valid_i),
        .s_axis_b_tdata(b_i),
        .m_axis_result_tvalid(valid_o),
        .m_axis_result_tdata(y_o)
    );
endmodule