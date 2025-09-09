module base2_k_picker #(
    parameter integer DW=16,
    parameter integer K_MIN=-16,
    parameter integer K_MAX= 16
)(
    input  wire          clk,
    input  wire          rstn,
    input  wire          valid_i,
    input  wire [DW-1:0] t_i,          // FP16
    output reg  signed [7:0] k_o,      // floor(t)
    output reg  [DW-1:0]     k_fp16_o, // FP16-repr of k
    output wire [DW-1:0]     f_o,      // t - k
    output wire              valid_o
);
    // 경계 { …, -1, 0, 1, 2, … } 를 FP16로 보관
    // 간단히 k를 saturated floor로 선택
    // 구현: 비교기 체인 + 우선순위 인코더(범위 좁아 작음)

    // 예시 간단 FSM: valid_i 한 싸이클 뒤에 결과 내놓기 (조정 가능)
    reg v_d;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            k_o <= 0; k_fp16_o <= {DW{1'b0}}; v_d <= 1'b0;
        end else begin
            v_d <= valid_i;
            // 아래는 개념: 실제론 FP16 비교로 floor를 찾음
            // (간결화를 위해 0, ±1~±16만 예시)
            // 실제 구현에선 반복 generate로 경계 배열을 넣으세요.
            // 여기선 t_i가 [-16,16] 안이라고 가정하고, 정수 근처 비교로 대충 예시:
            // *** 실제로는 ROM에 FP16 정수상수 넣고 범위 비교하세요. ***

            // PLACEHOLDER: 0 기준만 예시 (반드시 교체)
            if (t_i[DW-1]) begin // t<0
                k_o <= -1; k_fp16_o <= 16'hBC00; // -1.0 half
            end else begin
                k_o <= 0;  k_fp16_o <= 16'h0000; // 0.0 half
            end
        end
    end
    assign valid_o = v_d;

    // f = t - k
    fp16_add u_sub_k (
        .clk(clk), .rstn(rstn),
        .valid_i(valid_o),
        .a_i(t_i),
        .b_i({~k_fp16_o[DW-1], k_fp16_o[DW-2:0]}),
        .sum_o(f_o),
        .valid_o( /* unused; downstream aligns separately */ )
    );
endmodule
