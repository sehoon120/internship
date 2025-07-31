`timescale 1ns / 1ps
// B_mat 제대로 할당 안되는 문제가 있음
// 수정중
module testbench_fp16_tile;

    parameter B = 1, H = 24, P = 64, N = 128;
    // parameter B = 1, H = 16, P = 4, N = 16;
    parameter H_tile = 4, P_tile = 4;
    parameter DW = 16;

    reg clk;
    reg rst;
    reg start;

    wire done;
    wire [B*H_tile*P_tile*DW-1:0] y_tile_flat;

    reg [B*H_tile*DW-1:0] dt_flat;
    reg [B*H_tile*DW-1:0] dA_flat;
    reg [B*N*DW-1:0] Bmat_flat;
    reg [B*N*DW-1:0] C_flat;
    reg [H_tile*DW-1:0]   D_flat;
    reg [B*H_tile*P_tile*DW-1:0] x_flat;
    reg [B*H_tile*P_tile*N*DW-1:0] h_prev_flat;

    ssm_block_fp16_top #(
        .B(B), .H(H_tile), .P(P_tile), .N(N), .DW(DW)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .dt_flat(dt_flat), .dA_flat(dA_flat),
        .Bmat_flat(Bmat_flat), .C_flat(C_flat), .D_flat(D_flat),
        .x_flat(x_flat), .h_prev_flat(h_prev_flat),
        .y_flat(y_tile_flat), .done(done)
    );

    reg [DW-1:0] dt_mem     [0:B*H-1];
    reg [DW-1:0] dA_mem     [0:B*H-1];
    reg [DW-1:0] Bmat_mem   [0:B*N-1];
    reg [DW-1:0] C_mem      [0:B*N-1];
    reg [DW-1:0] D_mem      [0:H-1];
    reg [DW-1:0] x_mem      [0:B*H*P-1];
    reg [DW-1:0] h_prev_mem [0:B*H*P*N-1];
    reg [DW-1:0] y_accum    [0:B*H*P-1];

    reg [DW-1:0] add_a, add_b;
    reg add_valid_in;
    wire [DW-1:0] add_result;
    wire add_valid_out;

    fp16_add_wrapper u_add (
        .clk(clk),
        .a(add_a),
        .b(add_b),
        .valid_in(add_valid_in),
        .result(add_result),
        .valid_out(add_valid_out)
    );

    always #5 clk = ~clk;

    integer i, h_idx, p_idx, h_off, p_off, n;
    integer fout;
    integer tile_count = 0;
    integer h_rel, p_rel, hp_rel, h_abs, p_abs, index_out;
    integer acc_idx;
    reg waiting_for_add;
    reg [DW-1:0] tile_data [0:H_tile*P-1];

    initial begin
        $display("==== FP16 SSM Block Tiled Simulation ====");
        clk = 0; rst = 1; start = 0; add_valid_in = 0; waiting_for_add = 0;

        $readmemh("C:/Internship/intermediate_datas/0_dt.hex",     dt_mem);
        $readmemh("C:/Internship/intermediate_datas/0_dA.hex",     dA_mem);
        $readmemh("C:/Internship/intermediate_datas/0_B.hex",   Bmat_mem);
        $readmemh("C:/Internship/intermediate_datas/0_C.hex",      C_mem);
        $readmemh("C:/Internship/intermediate_datas/0_D.hex",      D_mem);
        $readmemh("C:/Internship/intermediate_datas/0_x.hex",      x_mem);
        $readmemh("C:/Internship/intermediate_datas/0_ssm_state.hex", h_prev_mem);

        for (i = 0; i < B*H*P; i = i + 1) y_accum[i] = 0;

        #10 rst = 0;

        for (h_idx = 0; h_idx < H; h_idx = h_idx + H_tile) begin
            for (p_idx = 0; p_idx < P; p_idx = p_idx + P_tile) begin

                h_off = h_idx;
                p_off = p_idx;

                // dt, dA, D 설정
                for (i = 0; i < H_tile; i = i + 1) begin
                    dt_flat[DW*i +: DW] = dt_mem[h_off + i];
                    dA_flat[DW*i +: DW] = dA_mem[h_off + i];
                    D_flat[DW*i +: DW]  = D_mem[h_off + i];
                end

                // B, C 설정
                for (i = 0; i < N; i = i + 1) begin
                    Bmat_flat[DW*i +: DW] = Bmat_mem[i];
                    C_flat[DW*i +: DW]    = C_mem[i];
                end

                // x 설정
                for (i = 0; i < H_tile*P_tile; i = i + 1) begin
                    h_rel = i / P_tile;
                    p_rel = i % P_tile;
                    h_abs = h_off + h_rel;
                    p_abs = p_off + p_rel;
                    x_flat[DW*i +: DW] = x_mem[(h_abs * P) + p_abs];
                end

                // h_prev 설정
                for (i = 0; i < H_tile*P_tile*N; i = i + 1) begin
                    hp_rel = i / N;
                    n = i % N;
                    h_rel = hp_rel / P_tile;
                    p_rel = hp_rel % P_tile;
                    h_abs = h_off + h_rel;
                    p_abs = p_off + p_rel;
                    h_prev_flat[DW*i +: DW] = h_prev_mem[((h_abs * P + p_abs) * N) + n];
                end

                #10 start = 1;
                #10 start = 0;
                wait (done == 1);
                #10;

                for (i = 0; i < H_tile*P; i = i + 1)
                    tile_data[i] = y_tile_flat[DW*i +: DW];

                acc_idx = 0;
                waiting_for_add = 0;

                while (acc_idx < H_tile * P_tile) begin
                    @(posedge clk);

                    h_rel = acc_idx / P_tile;
                    p_rel = acc_idx % P_tile;
                    h_abs = h_off + h_rel;
                    p_abs = p_off + p_rel;
                    index_out = h_abs * P + p_abs;

                    if (!waiting_for_add) begin
                        add_a <= y_accum[index_out];
                        add_b <= tile_data[acc_idx];
                        add_valid_in <= 1;
                        waiting_for_add <= 1;
                    end else begin
                        add_valid_in <= 0;
                        if (add_valid_out) begin
                            y_accum[index_out] <= add_result;
                            acc_idx = acc_idx + 1;
                            waiting_for_add <= 0;
                        end
                    end
                end

                tile_count = tile_count + 1;
                $display("acc_idx=%0d -> index_out=%0d -> add %04x + %04x = %04x", acc_idx, index_out, add_a, add_b, add_result);
                $display("Processed tile %0d/%0d (h=%0d~%0d, n=%0d~%0d)\n", tile_count, (H/H_tile)*(P/P_tile), h_idx, h_idx+H_tile-1, p_idx, p_idx+P_tile-1);
            end
        end

        #130

        fout = $fopen("C:/Internship/intermediate_datas/0_y_out.hex", "w");
        for (i = 0; i < B*H*P; i = i + 1) $fdisplay(fout, "%04h", y_accum[i]);
        $fclose(fout);

        $display("All tiles processed with FP16 addition. Output saved to y_out.hex");
        #10 $finish;
    end

endmodule
