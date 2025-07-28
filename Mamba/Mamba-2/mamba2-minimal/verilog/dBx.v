// dBx[b][h][p][n] = dt[b][h] * Bmat[b][n] * x[b][h][p];
// dBx_calc.v

module dBx_calc_fp16 #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter N = 4,
    parameter DW = 16,
    parameter M_LAT = 6
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

    // === 내부 배열 선언 (unpacked) ===
    wire [DW-1:0] dt   [0:B*H-1];
    wire [DW-1:0] Bmat [0:B*N-1];
    wire [DW-1:0] x    [0:B*H*P-1];
    reg  [DW-1:0] dBx  [0:B*H*P*N-1];

    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1)
            assign dt[g] = dt_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*N; g = g + 1)
            assign Bmat[g] = Bmat_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*H*P; g = g + 1)
            assign x[g] = x_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*H*P*N; g = g + 1)
            always @(*) dBx_flat[(g+1)*DW-1 -: DW] = dBx[g];
    endgenerate

    // === FSM ===
    reg [1:0] state;
    localparam IDLE = 2'd0, CALC = 2'd1, DONE = 2'd2;

    reg [9:0] b, h, p, n;
    reg [9:0] b_shift[0:M_LAT-1], h_shift[0:M_LAT-1], p_shift[0:M_LAT-1], n_shift[0:M_LAT-1];

    reg  [DW-1:0] in1_stage1, in2_stage1;
    wire [DW-1:0] out_stage1;
    wire          valid_stage1;

    reg  [DW-1:0] in1_stage2, in2_stage2;
    wire [DW-1:0] out_stage2;
    wire          valid_stage2;

    reg valid_in;

    integer j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done  <= 0;
            b <= 0; h <= 0; p <= 0; n <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0; n <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    // Stage 1
                    in1_stage1 <= dt[b*H + h];
                    in2_stage1 <= Bmat[b*N + n];
                    valid_in   <= 1;

                    // shift index
                    b_shift[0] <= b; h_shift[0] <= h; p_shift[0] <= p; n_shift[0] <= n;
                    for (j = 1; j < M_LAT; j = j + 1) begin
                        b_shift[j] <= b_shift[j-1];
                        h_shift[j] <= h_shift[j-1];
                        p_shift[j] <= p_shift[j-1];
                        n_shift[j] <= n_shift[j-1];
                    end

                    // Stage 2
                    in1_stage2 <= out_stage1;
                    in2_stage2 <= x[b_shift[M_LAT-1]*H*P + h_shift[M_LAT-1]*P + p_shift[M_LAT-1]];

                    // 결과 저장
                    if (valid_stage2) begin
                        dBx[b_shift[M_LAT-1]*H*P*N + h_shift[M_LAT-1]*P*N + p_shift[M_LAT-1]*N + n_shift[M_LAT-1]] <= out_stage2;
                    end

                    // Index++
                    if (n == N-1) begin
                        n <= 0;
                        if (p == P-1) begin
                            p <= 0;
                            if (h == H-1) begin
                                h <= 0;
                                if (b == B-1) begin
                                    state <= DONE;
                                end else b <= b + 1;
                            end else h <= h + 1;
                        end else p <= p + 1;
                    end else n <= n + 1;
                end

                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // === IP 연결 ===
    fp16_mult_wrapper u_fp16_mul_1 (
        .clk(clk),
        .a(in1_stage1),
        .b(in2_stage1),
        .valid_in(valid_in),
        .result(out_stage1),
        .valid_out(valid_stage1)
    );

    fp16_mult_wrapper u_fp16_mul_2 (
        .clk(clk),
        .a(in1_stage2),
        .b(in2_stage2),
        .valid_in(valid_stage1),
        .result(out_stage2),
        .valid_out(valid_stage2)
    );

endmodule
