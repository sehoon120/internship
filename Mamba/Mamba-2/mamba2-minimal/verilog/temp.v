// dBx[b][h][p][n] = dt[b][h] * Bmat[b][n] * x[b][h][p];
// dBx_calc.v
// ================================================================================
// ?��?��?��?��:
// 개선?�� 버전: 8-way 병렬 multiplier 구조 ?��?��
// ================================================================================


module dBx_calc_fp16 #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter N = 4,
    parameter DW = 16,
    parameter M_LAT = 6,    // FP16 multiplier latency
    parameter PAR = 8       // 병렬 곱셈�? 개수
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*DW-1:0]    dt_flat,
    input  wire [B*N*DW-1:0]    Bmat_flat,
    input  wire [B*H*P*DW-1:0]  x_flat,

    output reg  [B*H*P*N*DW-1:0] dBx_flat,
    output reg  done
);

    // ?���? 배열
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
            always @(*) dBx_flat[(g+1)*DW-1 -: DW] = dBx[g];
        end
    endgenerate

    // FSM
    localparam IDLE = 0, CALC = 1, FLUSH = 2, DONE = 3;
    reg [1:0] state;
    reg [$clog2(B):0] b;
    reg [$clog2(H):0] h;
    reg [$clog2(P):0] p;
    reg [$clog2(N):0] n;
    reg [3:0] flush_cnt;

    reg [DW-1:0] dt_val, x_val;
    wire [DW-1:0] dtx_out [0:PAR-1];
    wire [DW-1:0] dBx_out [0:PAR-1];
    wire          valid_stage [0:PAR-1];
    wire          valid_o_1 [0:PAR-1];
    reg valid_in;

    integer i,j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
            flush_cnt <= 0;
            b <= 0; h <= 0; p <= 0; n <= 0;
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
                            if (valid_stage[i]) begin
                                dBx[b*H*P*N + h*P*N + p*N + (n + i)] <= dBx_out[i];
                            end
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
                    end else begin
                        n <= n + PAR;
                    end
                end
                FLUSH:begin
                    flush_cnt <= flush_cnt + 1
                    for (j = 0; j < PAR; j = j + 1) begin
                        if (n + j < N) begin
                            if (valid_stage[j]) begin
                                dBx[b*H*P*N + h*P*N + p*N + (n + j)] <= dBx_out[i];
                            end
                        end
                    end
                    if (n + PAR >= N) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                    // state <= DONE;
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

    // 병렬 곱셈�?
    generate
        for (g = 0; g < PAR; g = g + 1) begin : MUL_ARRAY
            wire [DW-1:0] dt_in, x_in, B_in;
            assign dt_in = dt[b*H + h];
            assign x_in  = x[b*H*P + h*P + p];
            assign B_in  = (n + g < N) ? Bmat[b*N + (n + g)] : 16'd0;

            wire [DW-1:0] dtx;
            fp16_mult_wrapper mul1 (
                .clk(clk), .a(dt_in), .b(x_in), .valid_in(valid_in),
                .result(dtx), .valid_out(valid_o_1[g])
            );
            fp16_mult_wrapper mul2 (
                .clk(clk), .a(dtx), .b(B_in), .valid_in(valid_o_1[g]),
                .result(dBx_out[g]), .valid_out(valid_stage[g])
            );
        end
    endgenerate

endmodule
