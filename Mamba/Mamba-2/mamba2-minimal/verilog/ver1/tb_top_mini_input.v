`timescale 1ns / 1ps

module testbench_fp16;

    parameter B = 1, H = 4, P = 4, N = 4;
    // parameter B = 1, H = 24, P = 64, N = 128;
    parameter DW = 16;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [B*H*P*DW-1:0] y_flat;

    reg [B*H*DW-1:0]     dt_flat;
    reg [B*H*DW-1:0]     dA_flat;
    reg [B*N*DW-1:0]     Bmat_flat;
    reg [B*N*DW-1:0]     C_flat;
    reg [H*DW-1:0]       D_flat;
    reg [B*H*P*DW-1:0]   x_flat;
    reg [B*H*P*N*DW-1:0] h_prev_flat;

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

    // Memory arrays for loading .hex files
    reg [DW-1:0] dt_mem     [0:B*H-1];
    reg [DW-1:0] dA_mem     [0:B*H-1];
    reg [DW-1:0] Bmat_mem   [0:B*N-1];
    reg [DW-1:0] C_mem      [0:B*N-1];
    reg [DW-1:0] D_mem      [0:H-1];
    reg [DW-1:0] x_mem      [0:B*H*P-1];
    reg [DW-1:0] h_prev_mem [0:B*H*P*N-1];

    // Clock generation
    always #5 clk = ~clk;

    integer i;
    integer fout;

    initial begin
        $display("==== FP16 SSM Block Simulation ====");
        clk = 0; rst = 1; start = 0;

        // Load from .hex
        $readmemh("C:/Internship/intermediate_datas/0_dt_copy.hex",     dt_mem);
        $readmemh("C:/Internship/intermediate_datas/0_dA_copy.hex",     dA_mem);
        $readmemh("C:/Internship/intermediate_datas/0_B_copy.hex",   Bmat_mem);
        $readmemh("C:/Internship/intermediate_datas/0_C_copy.hex",      C_mem);
        $readmemh("C:/Internship/intermediate_datas/0_D_copy.hex",      D_mem);
        $readmemh("C:/Internship/intermediate_datas/0_x_copy.hex",      x_mem);
        $readmemh("C:/Internship/intermediate_datas/0_ssm_state_copy.hex", h_prev_mem);

        // Flatten to *_flat
        for (i = 0; i < B*H; i = i + 1) begin
            dt_flat[DW*i +: DW] = dt_mem[i];
            dA_flat[DW*i +: DW] = dA_mem[i];
        end

        for (i = 0; i < B*N; i = i + 1) begin
            Bmat_flat[DW*i +: DW] = Bmat_mem[i];
            C_flat[DW*i +: DW]    = C_mem[i];
        end

        for (i = 0; i < H; i = i + 1)
            D_flat[DW*i +: DW] = D_mem[i];

        for (i = 0; i < B*H*P; i = i + 1)
            x_flat[DW*i +: DW] = x_mem[i];

        for (i = 0; i < B*H*P*N; i = i + 1)
            h_prev_flat[DW*i +: DW] = h_prev_mem[i];

        #10 rst = 0;
        #10 start = 1;
        #10 start = 0;

        wait (done == 1);
        #10;

        // Save result to y.hex
        
        fout = $fopen("C:/Internship/intermediate_datas/0_y_out_copy.hex", "w");
        for (i = 0; i < B*H*P; i = i + 1) begin
            $fdisplay(fout, "%04h", y_flat[DW*i +: DW]);
        end
        $fclose(fout);

        $display("✅ Simulation done. Output saved to y.hex");
        $finish;
    end

endmodule
