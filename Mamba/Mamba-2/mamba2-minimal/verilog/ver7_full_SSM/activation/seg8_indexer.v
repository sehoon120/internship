module seg8_indexer #(
    parameter integer DW=16
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] f_i,      // 0<=f<1
    output reg  [2:0]    seg_o,    // 0..7
    output reg           valid_o
);
    // 간단 비교로 1/8, 2/8, ... 경계 찾기
    localparam [DW-1:0] H_1_8 = 16'h2E00; // ≈0.125 (approx)
    localparam [DW-1:0] H_2_8 = 16'h3200; // ≈0.25
    localparam [DW-1:0] H_3_8 = 16'h3400; // ≈0.375
    localparam [DW-1:0] H_4_8 = 16'h3800; // 0.5
    localparam [DW-1:0] H_5_8 = 16'h3980; // ≈0.625
    localparam [DW-1:0] H_6_8 = 16'h3A80; // ≈0.75
    localparam [DW-1:0] H_7_8 = 16'h3B40; // ≈0.875

    function [0:0] le(input [DW-1:0] a, input [DW-1:0] b);
        // FP16 a<=b (부호/지수/가수 비교 간단 구현; 정확 비교기는 별도 모듈 권장)
        begin
            // 매우 단순화된 비교(양수 가정). f∈[0,1)라 부호=0.
            le = (a <= b);
        end
    endfunction

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin seg_o<=3'd0; valid_o<=1'b0; end
        else begin
            valid_o <= valid_i;
            // 0..1/8..2/8.. 구간 선택
            if      (le(f_i, H_1_8)) seg_o <= 3'd0;
            else if (le(f_i, H_2_8)) seg_o <= 3'd1;
            else if (le(f_i, H_3_8)) seg_o <= 3'd2;
            else if (le(f_i, H_4_8)) seg_o <= 3'd3;
            else if (le(f_i, H_5_8)) seg_o <= 3'd4;
            else if (le(f_i, H_6_8)) seg_o <= 3'd5;
            else if (le(f_i, H_7_8)) seg_o <= 3'd6;
            else                     seg_o <= 3'd7;
        end
    end
endmodule
