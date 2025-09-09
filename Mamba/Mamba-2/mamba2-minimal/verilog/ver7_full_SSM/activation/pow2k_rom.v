module pow2k_rom #(
    parameter integer DW=16,
    parameter integer K_MIN=-16,
    parameter integer K_MAX= 16
)(
    input  wire          clk, input wire rstn,
    input  wire          valid_i,
    input  wire signed [7:0] k_i,
    output reg  [DW-1:0] two_pow_k_o,
    output reg           valid_o
);
    // 간단: half-precision에서 지수만 세팅하면 2^k 생성 가능(정규/비정규 경계 처리)
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin two_pow_k_o<=0; valid_o<=1'b0; end
        else begin
            valid_o <= valid_i;
            // saturate 범위
            if (k_i > 15) begin
                two_pow_k_o <= 16'h7BFF; // max half
            end else if (k_i >= -14) begin
                // sign=0, exp = k+15, frac=0
                two_pow_k_o <= {1'b0, (k_i+15)[4:0], 10'b0};
            end else if (k_i == -15) begin
                two_pow_k_o <= 16'h0200; // denorm
            end else if (k_i == -16) begin
                two_pow_k_o <= 16'h0100; // smaller denorm
            end else begin
                two_pow_k_o <= 16'h0000; // underflow
            end
        end
    end
endmodule
