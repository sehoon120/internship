// ssm_update.v
// h_next[b][h][p][n] = h_prev[b][h][p][n] × dA[b][h] + dBx[b][h][p][n];

module ssm_update_fp16 #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter N = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    parameter A_LAT = 11
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*DW-1:0]      dA_flat,
    input  wire [B*H*P*N*DW-1:0]  h_prev_flat,
    input  wire [B*H*P*N*DW-1:0]  dBx_flat,

    output wire [B*H*P*N*DW-1:0]  h_next_flat,
    output reg  done
);

    // 내부 배열
    wire [DW-1:0] dA     [0:B*H-1];
    wire [DW-1:0] h_prev [0:B*H*P*N-1];
    wire [DW-1:0] dBx    [0:B*H*P*N-1];
    reg  [DW-1:0] h_next [0:B*H*P*N-1];

    // flatten/unflatten
    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1)
            assign dA[g] = dA_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign h_prev[g] = h_prev_flat[(g+1)*DW-1 -: DW];
            assign dBx[g]    = dBx_flat[(g+1)*DW-1 -: DW];
            assign h_next_flat[(g+1)*DW-1 -: DW] = h_next[g];
        end
    endgenerate

    // FSM 상태
    reg [1:0] state;
    localparam IDLE = 2'd0,
               CALC = 2'd1,
               FLUSH = 2'd2,
               DONE = 2'd3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p, n;
    localparam SHIFT_DEPTH = (A_LAT + M_LAT + 1);
    reg [9:0] b_shift [0:SHIFT_DEPTH-1];
    reg [9:0] h_shift [0:SHIFT_DEPTH-1];
    reg [9:0] p_shift [0:SHIFT_DEPTH-1];
    reg [9:0] n_shift [0:SHIFT_DEPTH-1];

    wire [DW-1:0] mul_out, add_out;
    wire          mul_valid, add_valid;

    reg  [DW-1:0] mul_in1, mul_in2;
    // reg  [DW-1:0] add_in1; 
    reg  [DW-1:0] add_in2;
    reg           valid_in;

    integer j;

    // FSM + Pipeline 제어
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
                    // 곱셈 입력
                    mul_in1 <= h_prev[b*H*P*N + h*P*N + p*N + n];
                    mul_in2 <= dA[b*H + h];
                    valid_in <= 1;

                    // shift
                    b_shift[0] <= b; h_shift[0] <= h; p_shift[0] <= p; n_shift[0] <= n;
                    for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                        b_shift[j] <= b_shift[j-1];
                        h_shift[j] <= h_shift[j-1];
                        p_shift[j] <= p_shift[j-1];
                        n_shift[j] <= n_shift[j-1];
                    end

                    // 덧셈 입력
                    // add_in1 <= mul_out;
                    add_in2 <= dBx[b_shift[M_LAT-1]*H*P*N + h_shift[M_LAT-1]*P*N + p_shift[M_LAT-1]*N + n_shift[M_LAT-1]];

                    // 결과 저장
                    if (add_valid) begin
                        h_next[b_shift[SHIFT_DEPTH-1]*H*P*N + h_shift[SHIFT_DEPTH-1]*P*N + p_shift[SHIFT_DEPTH-1]*N + n_shift[SHIFT_DEPTH-1]] <= add_out;
                    end

                    // index 증가
                    if (n == N-1) begin
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
                    end else n <= n + 1;
                end

                FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                        b_shift[j] <= b_shift[j-1];
                        h_shift[j] <= h_shift[j-1];
                        p_shift[j] <= p_shift[j-1];
                        n_shift[j] <= n_shift[j-1];
                    end
                    add_in2 <= dBx[b_shift[M_LAT-1]*H*P*N + h_shift[M_LAT-1]*P*N + p_shift[M_LAT-1]*N + n_shift[M_LAT-1]];
                    if (add_valid) begin
                        h_next[b_shift[SHIFT_DEPTH-1]*H*P*N + h_shift[SHIFT_DEPTH-1]*P*N + p_shift[SHIFT_DEPTH-1]*N + n_shift[SHIFT_DEPTH-1]] <= add_out;
                    end
                    if (n == N-1) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                end else b <= b + 1;
                            end else h <= h + 1;
                        end else p <= p + 1;
                    end else n <= n + 1;
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

    // IP 연결
    fp16_mult_wrapper u_mult (
        .clk(clk),
        .a(mul_in1),
        .b(mul_in2),
        .valid_in(valid_in),
        .result(mul_out),
        .valid_out(mul_valid)
    );

    fp16_add_wrapper u_add (
        .clk(clk),
        .a(mul_out),
        .b(add_in2),
        .valid_in(mul_valid),
        .result(add_out),
        .valid_out(add_valid)
    );

endmodule

