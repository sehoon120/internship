`timescale 1ns / 1ps

module testbench_fp16;

    parameter B = 1, H = 4, P = 4, N = 4;
    parameter DW = 16;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [B*H*P*DW-1:0] y_flat;

    reg  [B*H*DW-1:0]     dt_flat;
    reg  [B*H*DW-1:0]     dA_flat;
    reg  [B*N*DW-1:0]     Bmat_flat;
    reg  [B*N*DW-1:0]     C_flat;
    reg  [H*DW-1:0]       D_flat;
    reg  [B*H*P*DW-1:0]   x_flat;
    reg  [B*H*P*N*DW-1:0] h_prev_flat;

    // DUT
    ssm_block_fp16_top #(
        .B(B), .H(H), .P(P), .N(N), .DW(DW)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .dt_flat(dt_flat), .dA_flat(dA_flat),
        .Bmat_flat(Bmat_flat), .C_flat(C_flat), .D_flat(D_flat),
        .x_flat(x_flat), .h_prev_flat(h_prev_flat),
        .y_flat(y_flat), .done(done)
    );

    // Clock generation
    always #5 clk = ~clk;
    integer i, j, k, n;

    initial begin
        $display("==== FP16 SSM Block Simulation ====");
        clk = 0;
        rst = 1;
        start = 0;

        #10; rst = 0;

        // 초기화
        for (i = 0; i < B; i = i + 1) begin
            for (j = 0; j < H; j = j + 1) begin
                dt_flat[DW*(i*H + j) +: DW] = 16'h4000;
                dA_flat[DW*(i*H + j) +: DW] = 16'h4000;
                for (k = 0; k < P; k = k + 1) begin
                    x_flat[DW*(i*H*P + j*P + k) +: DW] = 16'h4000;
                    for (n = 0; n < N; n = n + 1) begin
                        h_prev_flat[DW*(i*H*P*N + j*P*N + k*N + n) +: DW] = 16'h3C00;  // 0
                    end
                end
            end
        end

        for (i = 0; i < B; i = i + 1)
            for (j = 0; j < N; j = j + 1) begin
                Bmat_flat[DW*(i*N + j) +: DW] = 16'h4000;  // 2
                C_flat[DW*(i*N + j) +: DW]    = 16'h4200;  // 3
            end

        for (i = 0; i < H; i = i + 1)
            D_flat[DW*i +: DW] = 16'h3C00;

        #10; start = 1;
        #10; start = 0;

        wait (done == 1);
        #10;

        // 출력
        for (i = 0; i < B; i = i + 1)
            for (j = 0; j < H; j = j + 1)
                for (k = 0; k < P; k = k + 1) begin
                    $display("y[%0d][%0d][%0d] = 0x%h",
                        i, j, k, y_flat[DW*(i*H*P + j*P + k) +: DW]);
                end

        $finish;
    end

endmodule
