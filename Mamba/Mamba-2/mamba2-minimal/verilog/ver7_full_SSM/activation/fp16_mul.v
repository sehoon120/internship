module fp16_mul #(parameter DW=16, LAT=6)(
    input clk, input rstn, input valid_i,
    input [DW-1:0] a_i, b_i,
    output [DW-1:0] p_o, output valid_o
);
    // Replace with your FP16 mul IP wrapper.
    // For now, direct assign to help integration; keep interfaces identical.
    // ...
    floating_point_1 u_fp16_mult (
        .aclk(clk),
        .s_axis_a_tvalid(valid_i),
        .s_axis_a_tdata(a_i),
        .s_axis_b_tvalid(valid_i),
        .s_axis_b_tdata(b_i),
        .m_axis_result_tvalid(valid_o),
        .m_axis_result_tdata(p_o)
    );
endmodule