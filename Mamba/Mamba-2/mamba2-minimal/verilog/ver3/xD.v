// xD[b*H*P + h*H*P + p] = D[h] × x[b*H*P + h*H*P + p];
// xD.v
// xD = x * D

module xD #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    // parameter A_LAT = 11,
    parameter PAR_H = 12
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire acc_sig,

    // input  wire [B*H*P*DW-1:0] y_in_flat,
    input  wire [H*DW-1:0]     D_flat,
    input  wire [B*H*P*DW-1:0] x_flat,
    output wire [B*H*P*DW-1:0] xD_flat,
    output reg  done
);

    localparam SHIFT_DEPTH = (M_LAT + 1);

    // ?���? 배열 ?��?��
    // wire [DW-1:0] y_in [0:B*H*P-1];
    wire [DW-1:0] x     [0:B*H*P-1];
    wire [DW-1:0] D     [0:H-1];
    reg  [DW-1:0] xD    [0:B*H*P-1];

    // Unpack flat input/output
    genvar g;
    generate
        for (g = 0; g < B*H*P; g = g + 1) begin
            // assign y_in[g] = y_in_flat[(g+1)*DW-1 -: DW];
            assign x[g]    = x_flat[(g+1)*DW-1 -: DW];
            assign xD_flat[(g+1)*DW-1 -: DW] = xD[g];
        end
        for (g = 0; g < H; g = g + 1) begin
            assign D[g] = D_flat[(g+1)*DW-1 -: DW];
        end
    endgenerate

    // FSM
    reg [1:0] state;
    localparam IDLE = 0, CALC = 1, FLUSH = 2, DONE = 3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p;
    reg [9:0] b_shift[0:PAR_H-1][0:SHIFT_DEPTH-1];
    reg [9:0] h_shift[0:PAR_H-1][0:SHIFT_DEPTH-1];
    reg [9:0] p_shift[0:PAR_H-1][0:SHIFT_DEPTH-1];

    // IP ?��?��/출력
    reg  [DW-1:0] in1_mul[0:PAR_H-1], in2_mul[0:PAR_H-1];
    wire [DW-1:0] out_mul[0:PAR_H-1];
    wire          valid_mul[0:PAR_H-1];

    // reg  [DW-1:0] in1_add;
    // reg  [DW-1:0] in2_add;
    // wire [DW-1:0] out_add;
    // wire          valid_add;

    reg valid_in;
    integer i, j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            b <= 0; h <= 0; p <= 0;
            done <= 0;
            flush_cnt <= 0;
            valid_in <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    valid_in <= 0;
                    flush_cnt <= 0;
                    if (start) begin
                        b <= 0; h <= 0; p <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    valid_in <= 1;

                    // 곱셈 ?��?��
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (h + i < H) begin
                            // Stage 1: D × x
                            in1_mul[i] <= D[h+i];
                            in2_mul[i] <= x[b*H*P + (h+i)*P + p];
                        end
                        b_shift[i][0] <= b;
                        h_shift[i][0] <= h + i;
                        p_shift[i][0] <= p;
                        // n_shift[i][0] <= n;
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            // n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end

                    // 결과 ???��
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (valid_mul[i]) begin
                            xD[b_shift[i][SHIFT_DEPTH-1]*H*P + h_shift[i][SHIFT_DEPTH-1]*P + p_shift[i][SHIFT_DEPTH-1]] <= out_mul[i];
                        end
                    end

                    // ?��?��?�� 증�?
                    if (p == P-1) begin
                        p <= 0;
                        if (h + PAR_H >= H) begin
                            h <= 0;
                            if (b == B-1) begin
                                state <= FLUSH;
                            end else b <= b + 1;
                        end else h <= h + PAR_H;
                    end else p <= p + 1;
                end

                FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                        end
                    end
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (valid_mul[i]) begin
                            xD[b_shift[i][SHIFT_DEPTH-1]*H*P + h_shift[i][SHIFT_DEPTH-1]*P + p_shift[i][SHIFT_DEPTH-1]] <= out_mul[i];
                        end
                    end
                    // if (p == P-1) begin
                    //     p <= 0;
                    //     if (h + PAR_H >= H) begin
                    //         h <= 0;
                    //         if (b == B-1) begin
                    //             // state <= FLUSH;
                    //         end else b <= b + 1;
                    //     end else h <= h + PAR_H;
                    // end else p <= p + 1;
                    if (flush_cnt == SHIFT_DEPTH-1) begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    done <= 1;
                    if (acc_sig == 1) state <= IDLE;  // acc stage ?��?��?�� 종료
                end
            endcase
        end
    end

    // FP16 IP
    generate
        for (g = 0; g < PAR_H; g = g + 1) begin : PIPELINE
        fp16_mult_wrapper u_mul (
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
