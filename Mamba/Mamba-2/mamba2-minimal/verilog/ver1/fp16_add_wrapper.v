module fp16_add_wrapper (
    input  wire clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        valid_in,
    output wire [15:0] result,
    output wire        valid_out
);
    floating_point_1 u_fp16_add (
        .aclk(clk),
        .s_axis_a_tvalid(valid_in),
        .s_axis_a_tdata(a),
        .s_axis_b_tvalid(valid_in),
        .s_axis_b_tdata(b),
        .m_axis_result_tvalid(valid_out),
        .m_axis_result_tdata(result)
    );
endmodule
