// y[b][h][p] = sum over n of (h[b][h][p][n] × C[b][n]);
// Reordered to: n outer loop to simplify accumulation
// output_calc_fp16_n_outer.v
// True outer loop over `n`, so acc[...] accumulates fully before reuse

module output_calc_fp16 #(
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

    input  wire [B*H*P*N*DW-1:0] h_flat,
    input  wire [B*N*DW-1:0]     C_flat,

    output wire [B*H*P*DW-1:0]   y_flat,
    output reg  done
);

    localparam SHIFT_DEPTH = (A_LAT + M_LAT + 1);

    // Unpacked wires
    wire [DW-1:0] h [0:B*H*P*N-1];
    wire [DW-1:0] C [0:B*N-1];
    reg  [DW-1:0] y [0:B*H*P-1];

    genvar g;
    generate
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign h[g] = h_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*N; g = g + 1) begin
            assign C[g] = C_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign y_flat[(g+1)*DW-1 -: DW] = y[g];
        end
    endgenerate

    // FSM state
    reg [2:0] state;
    localparam IDLE = 0, CALC = 1, FLUSH = 2, DONE = 3;
    reg [4:0] flush_cnt;

    reg [9:0] n, b, h_idx, p;
    reg [9:0] b_s[0:SHIFT_DEPTH-1], h_s[0:SHIFT_DEPTH-1], p_s[0:SHIFT_DEPTH-1];

    reg [DW-1:0] acc [0:B*H*P-1];

    reg  [DW-1:0] in1_mul, in2_mul;
    wire [DW-1:0] out_mul;
    wire          valid_mul;

    reg  [DW-1:0] in1_add;
    wire [DW-1:0] out_add;
    wire          valid_add;

    reg valid_in;
    integer i, l;

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
                    valid_in <= 0;
                    for (i = 0; i < B*H*P; i = i + 1) acc[i] <= 0; n <= 0; b <= 0; h_idx <= 0; p <= 0;
                    if (start) state <= CALC;
                end
                CALC: begin
                    in1_mul <= h[b*H*P*N + h_idx*P*N + p*N + n];
                    in2_mul <= C[b*N + n];
                    valid_in <= 1;

                    b_s[0] <= b; h_s[0] <= h_idx; p_s[0] <= p;
                    for (l = 1; l < SHIFT_DEPTH; l = l + 1) begin
                        b_s[l] <= b_s[l-1];
                        h_s[l] <= h_s[l-1];
                        p_s[l] <= p_s[l-1];
                    end

                    in1_add <= acc[b_s[M_LAT-1]*H*P + h_s[M_LAT-1]*P + p_s[M_LAT-1]];

                    if (valid_add)
                        acc[b_s[SHIFT_DEPTH-1]*H*P + h_s[SHIFT_DEPTH-1]*P + p_s[SHIFT_DEPTH-1]] <= out_add;
                    if (valid_add && n == N-1)
                        y[b_s[SHIFT_DEPTH-1]*H*P + h_s[SHIFT_DEPTH-1]*P + p_s[SHIFT_DEPTH-1]] <= out_add;
                    
                    // Advance b,h,p loop
                    if (p == P-1) begin
                        p <= 0;
                        if (h_idx == H-1) begin
                            h_idx <= 0;
                            if (b == B-1) begin
                                b <= 0;
                                if (n == N-1)
                                    state <= FLUSH;
                                else n <= n + 1;
                            end else b <= b + 1;
                        end else h_idx <= h_idx + 1;
                    end else p <= p + 1;
                end
                FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (l = 1; l < SHIFT_DEPTH; l = l + 1) begin
                        b_s[l] <= b_s[l-1];
                        h_s[l] <= h_s[l-1];
                        p_s[l] <= p_s[l-1];
                    end
                    in1_add <= acc[b_s[M_LAT-1]*H*P + h_s[M_LAT-1]*P + p_s[M_LAT-1]];
                    if (valid_add) begin
                        acc[b_s[SHIFT_DEPTH-1]*H*P + h_s[SHIFT_DEPTH-1]*P + p_s[SHIFT_DEPTH-1]] <= out_add;
                    end
                    if (valid_add && n == N-1)
                        y[b_s[SHIFT_DEPTH-1]*H*P + h_s[SHIFT_DEPTH-1]*P + p_s[SHIFT_DEPTH-1]] <= out_add;
                    if (p == P-1) begin
                        p <= 0;
                        if (h_idx == H-1) begin
                            h_idx <= 0;
                            if (b == B-1) begin
                                b <= 0;
                                if (n == N-1) begin
                                end else n <= n + 1;
                            end else b <= b + 1;
                        end else h_idx <= h_idx + 1;
                    end else p <= p + 1;
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
    fp16_mult_wrapper u_mul (
        .clk(clk), .a(in1_mul), .b(in2_mul), .valid_in(valid_in),
        .result(out_mul), .valid_out(valid_mul)
    );

    fp16_add_wrapper u_add (
        .clk(clk), .a(in1_add), .b(out_mul), .valid_in(valid_mul),
        .result(out_add), .valid_out(valid_add)
    );

endmodule
