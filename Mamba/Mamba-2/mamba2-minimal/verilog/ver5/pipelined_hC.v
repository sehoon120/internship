// 
// ================================================================================
// 전반 piplining
// ================================================================================

module pipelined_hC #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter N = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    parameter A_LAT = 11,
    parameter PAR = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*DW-1:0]     dt_flat,
    input  wire [B*N*DW-1:0]     Bmat_flat,
    input  wire [B*N*DW-1:0]     C_flat,
    input  wire [B*H*P*DW-1:0]   x_flat,
    // input  wire [B*H*P*N*DW-1:0] dAh_flat,
    input  wire [B*H*DW-1:0]      dA_flat,
    input  wire [B*H*P*N*DW-1:0]  h_prev_flat,
    // output wire [B*H*P*N*DW-1:0]  h_mul_flat,

    output wire [B*H*P*N*DW-1:0] hC_flat,
    // output reg  done_dx, done_dxB, don_dAh_dxB,
    output reg  done
);

    wire [DW-1:0] dt     [0:B*H-1];
    wire [DW-1:0] dA     [0:B*H-1];
    wire [DW-1:0] Bmat   [0:B*N-1];
    wire [DW-1:0] C      [0:B*N-1];
    wire [DW-1:0] x      [0:B*H*P-1];
    wire [DW-1:0] h_prev [0:B*H*P*N-1];
    // wire [DW-1:0] dAh    [0:B*H*P*N-1];
    
    reg  [DW-1:0] dAh    [0:B*H*P*N-1];
    reg  [DW-1:0] hC     [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1) begin
            assign dt[g] = dt_flat[(g+1)*DW-1 -: DW];
            assign dA[g] = dA_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*N; g = g + 1) begin
            assign Bmat[g] = Bmat_flat[(g+1)*DW-1 -: DW];
            assign C[g] = C_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign x[g] = x_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            // assign dAh[g] = dAh_flat[(g+1)*DW-1 -: DW];
            assign h_prev[g] = h_prev_flat[(g+1)*DW-1 -: DW];
            assign hC_flat[(g+1)*DW-1 -: DW] = hC[g];
        end
    endgenerate

    reg [1:0] state;
    localparam IDLE = 2'd0, CALC = 2'd1, STAGE_FLUSH = 2'd2, DONE = 2'd3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p, n;
    localparam SHIFT_DEPTH = (3*M_LAT + A_LAT + 1);
    reg [9:0] b_shift [0:PAR-1][0:SHIFT_DEPTH - 1];
    reg [9:0] h_shift [0:PAR-1][0:SHIFT_DEPTH - 1];
    reg [9:0] p_shift [0:PAR-1][0:SHIFT_DEPTH - 1];
    reg [9:0] n_shift [0:PAR-1][0:SHIFT_DEPTH - 1];


    reg  [DW-1:0] in1_stage1 [0:PAR-1], in2_stage1 [0:PAR-1];
    wire [DW-1:0] out_stage1 [0:PAR-1];
    wire          valid_stage1 [0:PAR-1];

    reg  [DW-1:0] in1_dAh [0:PAR-1], in2_dAh [0:PAR-1];
    wire [DW-1:0] out_dAh [0:PAR-1];
    wire          valid_dAh [0:PAR-1];

    reg  [DW-1:0] in2_stage2 [0:PAR-1];
    wire [DW-1:0] out_stage2 [0:PAR-1];
    wire          valid_stage2 [0:PAR-1], valid_add [0:PAR-1];

    genvar vi;
    generate
        for (vi = 0; vi < PAR; vi = vi + 1) begin
            assign valid_add[vi] = valid_stage2[vi] && valid_dAh[vi];
        end
    endgenerate

    reg  [DW-1:0] in2_stage3 [0:PAR-1];
    wire [DW-1:0] out_stage3 [0:PAR-1];
    wire          valid_stage3 [0:PAR-1];

    reg  [DW-1:0] in2_stage4 [0:PAR-1];
    wire [DW-1:0] out_stage4 [0:PAR-1];
    wire          valid_stage4 [0:PAR-1];

    reg valid_in;
    integer i, j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            // done_dx, done_dxB, don_dAh_dxB <= 0;
            done  <= 0;
            b <= 0; h <= 0; p <= 0; n <= 0;
            flush_cnt <= 0;
            valid_in <= 0;
        end else begin
            case (state)
                IDLE: begin
                    // done_dx, done_dxB, don_dAh_dxB <= 0;
                    done <= 0;
                    flush_cnt <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0; n <= 0;
                        state <= CALC;
                        valid_in <= 0;
                    end
                end
                CALC: begin
                    valid_in <= 1;
                    
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin    // input 채우기
                            in1_stage1[i] <= dt[b*H + h];
                            in2_stage1[i] <= x[b*H*P + h*P + p];

                            in1_dAh[i]    <= dA[b*H + h];
                            in2_dAh[i]    <= h_prev[b*H*P*N + h*P*N + p*N + n+i];

                            in2_stage2[i] <= Bmat[b_shift[i][M_LAT-1]*N + n_shift[i][M_LAT-1]];
                            in2_stage3[i] <= dAh[b_shift[i][2*M_LAT-1]*H*P*N + h_shift[i][2*M_LAT-1]*P*N + p_shift[i][2*M_LAT-1]*N + n_shift[i][2*M_LAT-1]];
                            in2_stage4[i] <= C[b_shift[i][2*M_LAT+A_LAT-1]*N + n_shift[i][2*M_LAT+A_LAT-1]];
                        end
                        b_shift[i][0] <= b;
                        h_shift[i][0] <= h;
                        p_shift[i][0] <= p;
                        n_shift[i][0] <= n + i;
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end
                    for (i = 0; i < PAR; i = i + 1) begin   // result 연결
                        if (valid_dAh[i]) begin
                            dAh[b_shift[i][M_LAT]*H*P*N + h_shift[i][M_LAT]*P*N + p_shift[i][M_LAT]*N + n_shift[i][M_LAT]] <= out_dAh[i];
                        end
                        if (valid_stage4[i]) begin
                            hC[b_shift[i][SHIFT_DEPTH - 1]*H*P*N + h_shift[i][SHIFT_DEPTH - 1]*P*N + p_shift[i][SHIFT_DEPTH - 1]*N + n_shift[i][SHIFT_DEPTH - 1]] <= out_stage4[i];
                        end
                    end
                    if (n + PAR >= N) begin // idx 증가
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                    state <= STAGE_FLUSH;
                                end else b <= b + 1;
                            end else h <= h + 1;
                        end else p <= p + 1;
                    end else n <= n + PAR;
                end
                STAGE_FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin    // input 채우기
                            in2_stage2[i] <= Bmat[b_shift[i][M_LAT-1]*N + n_shift[i][M_LAT-1]];
                            in2_stage3[i] <= dAh[b_shift[i][2*M_LAT-1]*H*P*N + h_shift[i][2*M_LAT-1]*P*N + p_shift[i][2*M_LAT-1]*N + n_shift[i][2*M_LAT-1]];
                            in2_stage4[i] <= C[b_shift[i][2*M_LAT+A_LAT-1]*N + n_shift[i][2*M_LAT+A_LAT-1]];
                        end
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end
                    
                    for (i = 0; i < PAR; i = i + 1) begin   // result 연결
                        if (valid_dAh[i]) begin
                            dAh[b_shift[i][M_LAT]*H*P*N + h_shift[i][M_LAT]*P*N + p_shift[i][M_LAT]*N + n_shift[i][M_LAT]] <= out_dAh[i];
                        end
                        if (valid_stage4[i]) begin
                            hC[b_shift[i][SHIFT_DEPTH - 1]*H*P*N + h_shift[i][SHIFT_DEPTH - 1]*P*N + p_shift[i][SHIFT_DEPTH - 1]*N + n_shift[i][SHIFT_DEPTH - 1]] <= out_stage4[i];
                        end
                    end

                    if (n + PAR >= N) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                    // state <= STAGE_FLUSH;
                                end else b <= b + 1;
                            end else h <= h + 1;
                        end else p <= p + 1;
                    end else n <= n + PAR;

                    if (flush_cnt == SHIFT_DEPTH - 1) begin
                        state <= DONE;
                    end
                end
                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    generate
        for (g = 0; g < PAR; g = g + 1) begin : PIPELINE
            fp16_mult_wrapper mul1 (
                .clk(clk),
                .a(in1_stage1[g]),
                .b(in2_stage1[g]),
                .valid_in(valid_in),
                .result(out_stage1[g]),
                .valid_out(valid_stage1[g])
            );

            fp16_mult_wrapper muldAh (
                .clk(clk),
                .a(in1_dAh[g]),
                .b(in2_dAh[g]),
                .valid_in(valid_in),
                .result(out_dAh[g]),
                .valid_out(valid_dAh[g])
            );

            fp16_mult_wrapper mul2 (
                .clk(clk),
                .a(out_stage1[g]),
                .b(in2_stage2[g]),
                .valid_in(valid_stage1[g]),
                .result(out_stage2[g]),
                .valid_out(valid_stage2[g])
            );

            fp16_add_wrapper add1 (
                .clk(clk),
                .a(out_stage2[g]),
                .b(in2_stage3[g]),
                .valid_in(valid_stage2[g]),  // valid_add
                .result(out_stage3[g]),
                .valid_out(valid_stage3[g])
            );

            fp16_mult_wrapper mul3 (
                .clk(clk),
                .a(out_stage3[g]),
                .b(in2_stage4[g]),
                .valid_in(valid_stage3[g]),
                .result(out_stage4[g]),
                .valid_out(valid_stage4[g])
            );
        end
    endgenerate

endmodule
