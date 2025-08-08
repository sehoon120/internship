`timescale 1ns / 1ps

module testbench_fp16_wrapper;
    parameter B = 1, H = 24, P = 64, N = 128;
    // parameter B = 1, H = 24, P = 4, N = 32;
    // parameter H_tile = 12, P_tile = 16;
    parameter H_tile = 24, P_tile = 64;
    parameter DW = 16;

    reg clk;
    reg rst;
    reg start;
    wire done;

    reg  [B*H*DW-1:0]        dt_flat;
    reg  [B*H*DW-1:0]        dA_flat;
    reg  [B*N*DW-1:0]        Bmat_flat;
    reg  [B*N*DW-1:0]        C_flat;
    reg  [H*DW-1:0]          D_flat;
    reg  [B*H*P*DW-1:0]      x_flat;
    reg  [B*H*P*N*DW-1:0]    h_prev_flat;
    wire [B*H*P*DW-1:0]      y_flat_out;

    reg [DW-1:0] dt_mem     [0:B*H-1];
    reg [DW-1:0] dA_mem     [0:B*H-1];
    reg [DW-1:0] Bmat_mem   [0:B*N-1];
    reg [DW-1:0] C_mem      [0:B*N-1];
    reg [DW-1:0] D_mem      [0:H-1];
    reg [DW-1:0] x_mem      [0:B*H*P-1];
    reg [DW-1:0] h_prev_mem [0:B*H*P*N-1];

    // reg [DW-1:0] y_flat_out_mem    [0:B*H*P-1];

    integer i;

    packing #(
        .B(B), .H(H), .P(P), .N(N),
        .H_tile(H_tile), .P_tile(P_tile), .DW(DW)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .dt_flat_in(dt_flat), .dA_flat_in(dA_flat),
        .Bmat_flat_in(Bmat_flat), .C_flat_in(C_flat), .D_flat_in(D_flat),
        .x_flat_in(x_flat), .h_prev_flat_in(h_prev_flat),
        .y_flat_out(y_flat_out)
    );

    always #5 clk = ~clk;

    integer fout;
    
    initial begin
        $display("==== FP16 SSM Block Full Wrapper Testbench ====");
        clk = 0; rst = 1; start = 0;

        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",     dt_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",     dA_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",      Bmat_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",      C_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",      D_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",      x_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex", h_prev_mem);

         // Flatten
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

        wait(done);
        #10;

        $display("✅ Wrapper done. Writing result...");
        
        fout = $fopen("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out.hex", "w");
        for (i = 0; i < B*H*P; i = i + 1)
            $fdisplay(fout, "%04h", y_flat_out[DW*i +: DW]);
        $fclose(fout);

        $display("✅ Output saved. Simulation completed.");
        #10 $finish;
    end

endmodule
