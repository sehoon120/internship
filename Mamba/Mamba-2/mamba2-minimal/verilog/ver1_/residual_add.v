// y_out[b*H*P + h*H*P + p] = y_in[b*H*P + h*H*P + p] + D[h] × x[b*H*P + h*H*P + p];
// residual_add.v

module residual_add_fp16 #(
    parameter B = 1,
    parameter H = 4,
    parameter P = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    parameter A_LAT = 11
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*P*DW-1:0] y_in_flat,
    input  wire [H*DW-1:0]     D_flat,
    input  wire [B*H*P*DW-1:0] x_flat,

    output wire [B*H*P*DW-1:0] y_out_flat,
    output reg  done
);

    localparam SHIFT_DEPTH = (A_LAT + M_LAT + 1);

    // 내부 배열 선언
    wire [DW-1:0] y_in [0:B*H*P-1];
    wire [DW-1:0] x     [0:B*H*P-1];
    wire [DW-1:0] D     [0:H-1];
    reg  [DW-1:0] y_out [0:B*H*P-1];

    // Unpack flat input/output
    genvar g;
    generate
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign y_in[g] = y_in_flat[(g+1)*DW-1 -: DW];
            assign x[g]    = x_flat[(g+1)*DW-1 -: DW];
            assign y_out_flat[(g+1)*DW-1 -: DW] = y_out[g];
        end
        for (g = 0; g < H; g = g + 1)
            assign D[g] = D_flat[(g+1)*DW-1 -: DW];
    endgenerate

    // FSM
    reg [1:0] state;
    localparam IDLE = 0, CALC = 1, FLUSH = 2, DONE = 3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p;
    reg [9:0] b_shift[0:SHIFT_DEPTH-1];
    reg [9:0] h_shift[0:SHIFT_DEPTH-1];
    reg [9:0] p_shift[0:SHIFT_DEPTH-1];

    // IP 입력/출력
    reg  [DW-1:0] in1_mul, in2_mul;
    wire [DW-1:0] out_mul;
    wire          valid_mul;

    // reg  [DW-1:0] in1_add;
    reg  [DW-1:0] in2_add;
    wire [DW-1:0] out_add;
    wire          valid_add;

    reg valid_in;
    integer i;

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
                    if (start) begin
                        b <= 0; h <= 0; p <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    // Stage 1: D × x
                    in1_mul <= D[h];
                    in2_mul <= x[b*H*P + h*P + p];
                    valid_in <= 1;

                    // Shift register
                    b_shift[0] <= b;
                    h_shift[0] <= h;
                    p_shift[0] <= p;
                    for (i = 1; i < SHIFT_DEPTH; i = i + 1) begin
                        b_shift[i] <= b_shift[i-1];
                        h_shift[i] <= h_shift[i-1];
                        p_shift[i] <= p_shift[i-1];
                    end

                    // Stage 2: + y_in
                    // in1_add <= out_mul;
                    in2_add <= y_in[b_shift[M_LAT-1]*H*P + h_shift[M_LAT-1]*P + p_shift[M_LAT-1]];

                    // Stage 3: 저장
                    if (valid_add)
                        y_out[b_shift[SHIFT_DEPTH-1]*H*P + h_shift[SHIFT_DEPTH-1]*P + p_shift[SHIFT_DEPTH-1]] <= out_add;

                    // 인덱스 증가
                    if (p == P-1) begin
                        p <= 0;
                        if (h == H-1) begin
                            h <= 0;
                            if (b == B-1) begin
                                state <= FLUSH;
                            end else b <= b + 1;
                        end else h <= h + 1;
                    end else p <= p + 1;
                end

                FLUSH: begin
                    flush_cnt <= flush_cnt + 1;
                    for (i = 1; i < SHIFT_DEPTH; i = i + 1) begin
                        b_shift[i] <= b_shift[i-1];
                        h_shift[i] <= h_shift[i-1];
                        p_shift[i] <= p_shift[i-1];
                    end
                    in2_add <= y_in[b_shift[M_LAT-1]*H*P + h_shift[M_LAT-1]*P + p_shift[M_LAT-1]];
                    if (valid_add)
                        y_out[b_shift[SHIFT_DEPTH-1]*H*P + h_shift[SHIFT_DEPTH-1]*P + p_shift[SHIFT_DEPTH-1]] <= out_add;
                    if (p == P-1) begin
                        p <= 0;
                        if (h == H-1) begin
                            h <= 0;
                            if (b == B-1) begin
                            end else b <= b + 1;
                        end else h <= h + 1;
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

    // FP16 IP
    fp16_mult_wrapper u_mul (
        .clk(clk),
        .a(in1_mul),
        .b(in2_mul),
        .valid_in(valid_in),
        .result(out_mul),
        .valid_out(valid_mul)
    );

    fp16_add_wrapper u_add (
        .clk(clk),
        .a(out_mul),
        .b(in2_add),
        .valid_in(valid_mul),
        .result(out_add),
        .valid_out(valid_add)
    );

endmodule
