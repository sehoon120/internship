// h*C

module hC #(
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

    input  wire [B*H*P*N*DW-1:0] h_flat,
    input  wire [B*N*DW-1:0]     C_flat,

    output wire [B*H*P*N*DW-1:0]   hC_flat,
    output reg  done
);

    localparam SHIFT_DEPTH = (M_LAT + 1);

    // Unpacked wires
    wire [DW-1:0] h [0:B*H*P*N-1];
    wire [DW-1:0] C [0:B*N-1];
    reg  [DW-1:0] hC [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign h[g] = h_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*N; g = g + 1) begin
            assign C[g] = C_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign hC_flat[(g+1)*DW-1 -: DW] = hC[g];
        end
    endgenerate

    // FSM state
    reg [2:0] state;
    localparam IDLE = 0, CALC = 1, FLUSH = 2, DONE = 3;
    reg [4:0] flush_cnt;

    reg [9:0] n, b, h_idx, p;
    reg [9:0] b_shift[0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] h_shift[0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] p_shift[0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] n_shift[0:PAR-1][0:SHIFT_DEPTH-1];

    reg  [DW-1:0] in1_mul[0:PAR-1], in2_mul[0:PAR-1];
    wire [DW-1:0] out_mul[0:PAR-1];
    wire          valid_mul[0:PAR-1];

    reg valid_in;
    integer i, j, l;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            n <= 0; b <= 0; h_idx <= 0; p <= 0;
            flush_cnt <= 0;
            valid_in <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    flush_cnt <= 0;
                    valid_in <= 0;
                    n <= 0; b <= 0; h_idx <= 0; p <= 0;
                    // for (i = 0; i < B*H*P; i = i + 1) acc[i] <= 0; n <= 0; b <= 0; h_idx <= 0; p <= 0;
                    if (start) state <= CALC;
                end
                CALC: begin
                    valid_in <= 1;

                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin
                            in1_mul[i] <= h[b*H*P*N + h_idx*P*N + p*N + n + i];
                            in2_mul[i] <= C[b*N + n + i];    
                        end
                        b_shift[i][0] <= b;
                        h_shift[i][0] <= h_idx;
                        p_shift[i][0] <= p;
                        n_shift[i][0] <= n + i;
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end

                    // 결과 저장
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (valid_mul[i]) begin
                            hC[b_shift[i][SHIFT_DEPTH-1]*H*P*N + h_shift[i][SHIFT_DEPTH-1]*P*N + p_shift[i][SHIFT_DEPTH-1]*N + n_shift[i][SHIFT_DEPTH-1]] <= out_mul[i];
                        end
                    end

                    // index 증가
                    if (n + PAR >= N) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h_idx == H-1) begin
                                h_idx <= 0;
                                if (b == B-1) begin
                                    state <= FLUSH;
                                end else b <= b + 1;
                            end else h_idx <= h_idx + 1;
                        end else p <= p + 1;
                    end else n <= n + PAR;
                end
                FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (i = 0; i < PAR; i = i + 1) begin
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end

                    for (i = 0; i < PAR; i = i + 1) begin
                        if (valid_mul[i]) begin
                            hC[b_shift[i][SHIFT_DEPTH-1]*H*P*N + h_shift[i][SHIFT_DEPTH-1]*P*N + p_shift[i][SHIFT_DEPTH-1]*N + n_shift[i][SHIFT_DEPTH-1]] <= out_mul[i];
                        end
                    end

                    // if (n + PAR >= N) begin
                    //     n <= 0;
                    //     if (p == P-1) begin
                    //         p <= 0;
                    //         if (h_idx == H-1) begin
                    //             h_idx <= 0;
                    //             if (b == B-1) begin
                    //                 // state <= FLUSH;
                    //             end else b <= b + 1;
                    //         end else h_idx <= h_idx + 1;
                    //     end else p <= p + 1;
                    // end else n <= n + PAR;
                    if (flush_cnt == SHIFT_DEPTH-1) begin
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

    // IP wrappers
    generate
        for (g = 0; g < PAR; g = g + 1) begin : PIPELINE
            fp16_mult_wrapper mul1 (
                .clk(clk),
                .a(in1_mul[g]),
                .b(in2_mul[g]),
                .valid_in(valid_in),
                .result(out_mul[g]),
                .valid_out(valid_mul[g])
            );
        end
    endgenerate

endmodule