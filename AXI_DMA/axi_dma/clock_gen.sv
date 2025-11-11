module clock_gen (
    input  logic clk_in1_p,
    input  logic clk_in1_n,
    input  logic reset,
    output logic locked,
    output logic clock
);

    // Instance of the clock wizard module
    clk_wiz_0 CLOCKING (
        .clk_in1_p(clk_in1_p),
        .clk_in1_n(clk_in1_n),
        .clk_out1 (clock),      // 100 MHz
        .reset    (reset),
        .locked   (locked)
    );

endmodule
