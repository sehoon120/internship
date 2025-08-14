// N = 128 FP16 adder tree using fp16_add_wrapper
// Assumes latency-insensitive, pipelined valid_in → valid_out structure
// 128일시 N = 128로 교체, 주석 해제 등

module fp16_adder_tree_128 #(parameter DW=16, N=128)(    // N에 따라 교체
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [N*DW-1:0] in_flat,  // 128 inputs, each DW-bit
    output wire [DW-1:0]     sum,      // final accumulated sum
    output wire              valid_out
);

    // Unpack input
    wire [DW-1:0] in_level0 [0:127];
    // Declare wires for each tree level
    wire [DW-1:0] level1 [0:63];
    wire [DW-1:0] level2 [0:31];
    wire [DW-1:0] level3 [0:15];
    wire [DW-1:0] level4 [0:7];
    wire [DW-1:0] level5 [0:3];
    wire [DW-1:0] level6 [0:1];
    wire [DW-1:0] level7;
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : UNPACK
            assign in_level0[i] = in_flat[(i+1)*DW-1 -: DW]; // 128일시 교체
            // assign level2[i] = in_flat[(i+1)*DW-1 -: DW];
        end
    endgenerate

    

    wire valid_l1, valid_l2, valid_l3, valid_l4, valid_l5, valid_l6, valid_l7;

    assign valid_l1 = valid_in;  // 128일시 교체
    // assign valid_l3 = valid_in;

    // 128일시 주석 해제
    // Level 1 (128 → 64)
    generate
        for (i = 0; i < 64; i = i + 1) begin : L1
            fp16_add_wrapper add (
                .clk(clk),
                .a(in_level0[2*i]),
                .b(in_level0[2*i+1]),
                .valid_in(valid_l1),
                .result(level1[i]),
                .valid_out(valid_l2)
            );
        end
    endgenerate

    // Level 2 (64 → 32)
    generate
        for (i = 0; i < 32; i = i + 1) begin : L2
            fp16_add_wrapper add (
                .clk(clk),
                .a(level1[2*i]),
                .b(level1[2*i+1]),
                .valid_in(valid_l2),
                .result(level2[i]),
                .valid_out(valid_l3)
            );
        end
    endgenerate

    // Level 3 (32 → 16)
    generate
        for (i = 0; i < 16; i = i + 1) begin : L3
            fp16_add_wrapper add (
                .clk(clk),
                .a(level2[2*i]),
                .b(level2[2*i+1]),
                .valid_in(valid_l3),
                .result(level3[i]),
                .valid_out(valid_l4)
            );
        end
    endgenerate

    // Level 4 (16 → 8)
    generate
        for (i = 0; i < 8; i = i + 1) begin : L4
            fp16_add_wrapper add (
                .clk(clk),
                .a(level3[2*i]),
                .b(level3[2*i+1]),
                .valid_in(valid_l4),
                .result(level4[i]),
                .valid_out(valid_l5)
            );
        end
    endgenerate

    // Level 5 (8 → 4)
    generate
        for (i = 0; i < 4; i = i + 1) begin : L5
            fp16_add_wrapper add (
                .clk(clk),
                .a(level4[2*i]),
                .b(level4[2*i+1]),
                .valid_in(valid_l5),
                .result(level5[i]),
                .valid_out(valid_l6)
            );
        end
    endgenerate

    // Level 6 (4 → 2)
    generate
        for (i = 0; i < 2; i = i + 1) begin : L6
            fp16_add_wrapper add (
                .clk(clk),
                .a(level5[2*i]),
                .b(level5[2*i+1]),
                .valid_in(valid_l6),
                .result(level6[i]),
                .valid_out(valid_l7)
            );
        end
    endgenerate

    // Level 7 (2 → 1)
    fp16_add_wrapper final_add (
        .clk(clk),
        .a(level6[0]),
        .b(level6[1]),
        .valid_in(valid_l7),
        .result(level7),
        .valid_out(valid_out)
    );

    assign sum = level7;

endmodule
