// dBx[b][h][p][n] = dt[b][h] * Bmat[b][n] * x[b][h][p];
// dBx_calc_parallel.v
// ================================================================================
// 개선?��?��:
// - FLUSH ?���? ?��?��
// - 8-way 병렬 multiplier ?��?��
// - Index shift pipeline 구조 ?���?
// 
// dB는 더 빨리 계산되기에 이 부분 단축 가능
//
// ================================================================================

module dBx_calc_fp16 #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter N = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    parameter PAR = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*DW-1:0]    dt_flat,
    input  wire [B*N*DW-1:0]    Bmat_flat,
    input  wire [B*H*P*DW-1:0]  x_flat,

    output wire [B*H*P*N*DW-1:0] dBx_flat,
    output reg  done
);

    wire [DW-1:0] dt   [0:B*H-1];
    wire [DW-1:0] Bmat [0:B*N-1];
    wire [DW-1:0] x    [0:B*H*P-1];
    reg  [DW-1:0] dBx  [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1) begin
            assign dt[g] = dt_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*N; g = g + 1) begin
            assign Bmat[g] = Bmat_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign x[g] = x_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign dBx_flat[(g+1)*DW-1 -: DW] = dBx[g];
        end
    endgenerate

    reg [1:0] state;
    localparam IDLE = 2'd0, CALC = 2'd1, STAGE_FLUSH = 2'd2, DONE = 2'd3;
    reg [3:0] flush_cnt;

    reg [9:0] b, h, p, n;
    reg [9:0] b_shift [0:PAR-1][0:M_LAT*2];
    reg [9:0] h_shift [0:PAR-1][0:M_LAT*2];
    reg [9:0] p_shift [0:PAR-1][0:M_LAT*2];
    reg [9:0] n_shift [0:PAR-1][0:M_LAT*2];


    reg  [DW-1:0] in1_stage1 [0:PAR-1], in2_stage1 [0:PAR-1];
    wire [DW-1:0] out_stage1 [0:PAR-1];
    wire          valid_stage1 [0:PAR-1];

    reg  [DW-1:0] in2_stage2 [0:PAR-1];
    wire [DW-1:0] out_stage2 [0:PAR-1];
    wire          valid_stage2 [0:PAR-1];

    reg valid_in;
    integer i, j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done  <= 0;
            b <= 0; h <= 0; p <= 0; n <= 0;
            flush_cnt <= 0;
            valid_in <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0; n <= 0;
                        state <= CALC;
                        valid_in <= 0;
                    end
                end
                CALC: begin
                    valid_in <= 1;
                    
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin
                            in1_stage1[i] <= dt[b*H + h];
                            in2_stage1[i] <= Bmat[b*N + (n + i)];
                            in2_stage2[i] <= x[b_shift[i][M_LAT-1]*H*P + h_shift[i][M_LAT-1]*P + p_shift[i][M_LAT-1]];
                        end
                        b_shift[i][0] <= b;
                        h_shift[i][0] <= h;
                        p_shift[i][0] <= p;
                        n_shift[i][0] <= n + i;
                        for (j = 1; j < M_LAT+M_LAT+1; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (valid_stage2[i]) begin
                            dBx[b_shift[i][M_LAT*2]*H*P*N + h_shift[i][M_LAT*2]*P*N + p_shift[i][M_LAT*2]*N + n_shift[i][M_LAT*2]] <= out_stage2[i];
                        end
                    end
                    if (n + PAR >= N) begin
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
                        if (n + i < N) begin
                            in2_stage2[i] <= x[b_shift[i][M_LAT-1]*H*P + h_shift[i][M_LAT-1]*P + p];
                        end
                        for (j = 1; j < M_LAT+M_LAT+1; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end
                    
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (valid_stage2[i]) begin
                            dBx[b_shift[i][M_LAT*2]*H*P*N + h_shift[i][M_LAT*2]*P*N + p_shift[i][M_LAT*2]*N + n_shift[i][M_LAT*2]] <= out_stage2[i];
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
                    if (flush_cnt == M_LAT+M_LAT) begin
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

            fp16_mult_wrapper mul2 (
                .clk(clk),
                .a(out_stage1[g]),
                .b(in2_stage2[g]),
                .valid_in(valid_stage1[g]),
                .result(out_stage2[g]),
                .valid_out(valid_stage2[g])
            );
        end
    endgenerate

endmodule
