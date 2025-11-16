module pow2k_rom #(
    parameter integer DW=16,
    parameter integer K_MIN=-16,
    parameter integer K_MAX= 16
)(
    input  wire               clk,
    input  wire               rstn,
    input  wire               valid_i,
    input  wire signed [7:0]  k_i,
    output reg  [DW-1:0]      two_pow_k_o,
    output reg                valid_o
);
    // 중간값: k + bias(15). 연산 결과를 wire에 담고 슬라이스.
    wire signed [8:0] k_plus_bias = $signed(k_i) + 9'sd15;  // [-256..255] 안전
    // 참고: exp_field는 k ∈ [-14..15]에서만 사용됨.

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            two_pow_k_o <= {DW{1'b0}};
            valid_o     <= 1'b0;
        end else begin
            valid_o <= valid_i;

            if (k_i > 8'sd15) begin
                // 2^k가 half 범위를 넘으면 최대 finite로 saturate
                two_pow_k_o <= 16'h7BFF;      // +65504 (max half)
            end else if (k_i >= -8'sd14) begin
                // 정상수 영역: sign=0, exp = k + 15, frac=0
                // k_plus_bias[4:0] == exponent field
                two_pow_k_o <= {1'b0, k_plus_bias[4:0], 10'b0}; // 2^k (k∈[-14..15])
            end else if (k_i == -8'sd15) begin
                // 서브노멀에서 2^-15 = 2^-24 * 2^9 → frac=512
                two_pow_k_o <= 16'h0200;
            end else if (k_i == -8'sd16) begin
                // 2^-16 = 2^-24 * 2^8 → frac=256
                two_pow_k_o <= 16'h0100;
            end else begin
                // 그 이하 언더플로우는 0
                two_pow_k_o <= 16'h0000;
            end
        end
    end
endmodule
