module dAh_dBx #(parameter B=1, H=4, P=4, N=4, DW=16, A_LAT=11, PAR = 16) (
    input  wire clk, rst, start1, start2,
    input  wire [B*H*P*N*DW-1:0]  h_mul_flat,
    input  wire [B*H*P*N*DW-1:0]  dBx_flat,
    output wire [B*H*P*N*DW-1:0]  h_next_flat,
    output reg  done
);
    wire [DW-1:0] dBx     [0:B*H*P*N-1];
    wire [DW-1:0] h_mul   [0:B*H*P*N-1];
    reg  [DW-1:0] h_next  [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign dBx[g] = dBx_flat[(g+1)*DW-1 -: DW];
            assign h_mul[g] = h_mul_flat[(g+1)*DW-1 -: DW];
            assign h_next_flat[(g+1)*DW-1 -: DW] = h_next[g];
        end
    endgenerate

    wire start;
    assign start = start1 & start2;
    
    reg [1:0] state;
    localparam IDLE = 2'd0,
               CALC = 2'd1,
               FLUSH = 2'd2,
               DONE = 2'd3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p, n;
    localparam SHIFT_DEPTH = (A_LAT + 1);
    reg [9:0] b_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] h_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] p_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] n_shift [0:PAR-1][0:SHIFT_DEPTH-1];

    wire [DW-1:0] add_out [0:PAR-1];

    reg  [DW-1:0] add_in1 [0:PAR-1], add_in2 [0:PAR-1];
    reg           valid_in;
    wire          add_valid [0:PAR-1];

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
                    valid_in <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0; n <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    valid_in <= 1;
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin
                            add_in1[i] <= h_mul[b*H*P*N + h*P*N + p*N + n + i];
                            add_in2[i] <= dBx[b*H*P*N + h*P*N + p*N + n + i];    
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

                    for (i = 0; i < PAR; i = i + 1) begin
                        if (add_valid[i]) begin
                            h_next[b_shift[i][A_LAT]*H*P*N + h_shift[i][A_LAT]*P*N + p_shift[i][A_LAT]*N + n_shift[i][A_LAT]] <= add_out[i];
                        end
                    end

                    if (n + PAR >= N) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                    state <= FLUSH;
                                end else b <= b + 1;
                            end else h <= h + 1;
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
                        if (add_valid[i]) begin
                            h_next[b_shift[i][A_LAT]*H*P*N + h_shift[i][A_LAT]*P*N + p_shift[i][A_LAT]*N + n_shift[i][A_LAT]] <= add_out[i];
                        end
                    end
                    if (n + PAR >= N) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                end else b <= b + 1;
                            end else h <= h + 1;
                        end else p <= p + 1;
                    end else n <= n + PAR;
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

    generate
        for (g = 0; g < PAR; g = g + 1) begin : PIPELINE
            fp16_add_wrapper mul1 (
                .clk(clk),
                .a(add_in1[g]),
                .b(add_in2[g]),
                .valid_in(valid_in),
                .result(add_out[g]),
                .valid_out(add_valid[g])
            );
        end
    endgenerate
endmodule
