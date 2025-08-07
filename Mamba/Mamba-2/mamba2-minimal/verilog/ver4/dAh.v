// h_mul = h_prev × dA
module dAh #(parameter B=1, H=4, P=4, N=4, DW=16, M_LAT=6, PAR = 16) (
    input  wire clk, rst, start, done_hC,
    input  wire [B*H*DW-1:0]      dA_flat,
    input  wire [B*H*P*N*DW-1:0]  h_prev_flat,
    output wire [B*H*P*N*DW-1:0]  h_mul_flat,
    output reg  done
);
    wire [DW-1:0] dA     [0:B*H-1];
    wire [DW-1:0] h_prev [0:B*H*P*N-1];
    reg  [DW-1:0] h_mul  [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1)
            assign dA[g] = dA_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign h_prev[g] = h_prev_flat[(g+1)*DW-1 -: DW];
            assign h_mul_flat[(g+1)*DW-1 -: DW] = h_mul[g];
        end
    endgenerate

    reg [1:0] state;
    localparam IDLE = 2'd0,
               CALC = 2'd1,
               FLUSH = 2'd2,
               DONE = 2'd3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p, n;
    localparam SHIFT_DEPTH = (M_LAT + 1);
    reg [9:0] b_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] h_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] p_shift [0:PAR-1][0:SHIFT_DEPTH-1];
    reg [9:0] n_shift [0:PAR-1][0:SHIFT_DEPTH-1];

    wire [DW-1:0] mul_out [0:PAR-1];

    reg  [DW-1:0] mul_in1 [0:PAR-1], mul_in2 [0:PAR-1];
    reg           valid_in;
    wire          mul_valid [0:PAR-1];

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
                    flush_cnt <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0; n <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    valid_in <= 1;
                    // 곱셈 입력
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (n + i < N) begin
                            mul_in1[i] <= h_prev[b*H*P*N + h*P*N + p*N + n + i];
                            mul_in2[i] <= dA[b*H + h];    
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

                    // 결과 저장
                    for (i = 0; i < PAR; i = i + 1) begin
                        if (mul_valid[i]) begin
                            h_mul[b_shift[i][M_LAT]*H*P*N + h_shift[i][M_LAT]*P*N + p_shift[i][M_LAT]*N + n_shift[i][M_LAT]] <= mul_out[i];
                        end
                    end

                    // index 증가
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
                        if (mul_valid[i]) begin
                            h_mul[b_shift[i][M_LAT]*H*P*N + h_shift[i][M_LAT]*P*N + p_shift[i][M_LAT]*N + n_shift[i][M_LAT]] <= mul_out[i];
                        end
                    end
                    // if (n + PAR >= N) begin
                    //     n <= 0;
                    //     if (p == P-1) begin
                    //         p <= 0;
                    //         if (h == H-1) begin
                    //             h <= 0;
                    //             if (b == B-1) begin
                    //                 // state <= FLUSH;
                    //             end else b <= b + 1;
                    //         end else h <= h + 1;
                    //     end else p <= p + 1;
                    // end else n <= n + PAR;
                    if (flush_cnt == SHIFT_DEPTH-1) begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    done <= 1;
                    if (done_hC == 1) state <= IDLE;
                end
            endcase
        end
    end

    generate
        for (g = 0; g < PAR; g = g + 1) begin : PIPELINE
            fp16_mult_wrapper mul1 (
                .clk(clk),
                .a(mul_in1[g]),
                .b(mul_in2[g]),
                .valid_in(valid_in),
                .result(mul_out[g]),
                .valid_out(mul_valid[g])
            );
        end
    endgenerate
    // 내부 구조는 기존 `ssm_update_fp16`에서 add_in2 제거하고, 
    // mul_out을 바로 `h_mul_flat`에 저장하도록 구성합니다.

    // ↪ 이전 모듈에서 `add_in2` 제거
    // ↪ `h_next[...] <= mul_out;` 위치를 M_LAT + pipeline latency 뒤로 조정
    // ↪ 최종 `h_mul_flat[(g+1)*DW-1 -: DW] = h_mul[g];` 처리
endmodule
