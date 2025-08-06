// dx[b][h][p] = dt[b][h] * x[b][h][p];
// dx_calc_parallel.v
// ================================================================================
// dx ë¨¼ì? ê°œì„ ?•˜?„ë¡? ?ˆœ?„œ ë³?ê²?
// ?—°?‚° ?ˆ˜ ì¤„ì¼ ?ˆ˜ ?ˆ?Œ
// ================================================================================

module dx #(
    parameter B=1, H=4, P=4, 
    // parameter N=4, 
    parameter DW=16, M_LAT=6, PAR_H = 12
)(
    input  wire clk, rst, start,
    input  wire [B*H*DW-1:0]    dt_flat,
    input  wire [B*H*P*DW-1:0]  x_flat,
    output wire [B*H*P*DW-1:0]  dx_flat,
    output reg  done
);
    wire [DW-1:0] dt     [0:B*H-1];
    wire [DW-1:0] x      [0:B*H*P-1];
    reg  [DW-1:0] dx     [0:B*H*P-1];

    genvar g;
    generate
        for (g = 0; g < B*H; g = g + 1)
            assign dt[g] = dt_flat[(g+1)*DW-1 -: DW];
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign x[g] = x_flat[(g+1)*DW-1 -: DW];
            assign dx_flat[(g+1)*DW-1 -: DW] = dx[g];
        end
    endgenerate

    reg [1:0] state;
    localparam IDLE = 2'd0,
               CALC = 2'd1,
               FLUSH = 2'd2,
               DONE = 2'd3;
    reg [4:0] flush_cnt;

    reg [9:0] b, h, p;
    localparam SHIFT_DEPTH = (M_LAT + 1);
    reg [9:0] b_shift [0:PAR_H-1][0:SHIFT_DEPTH-1];
    reg [9:0] h_shift [0:PAR_H-1][0:SHIFT_DEPTH-1];
    reg [9:0] p_shift [0:PAR_H-1][0:SHIFT_DEPTH-1];
    // reg [9:0] n_shift [0:PAR_H-1][0:SHIFT_DEPTH-1];

    wire [DW-1:0] mul_out [0:PAR_H-1];

    reg  [DW-1:0] mul_in1 [0:PAR_H-1], mul_in2 [0:PAR_H-1];
    reg           valid_in;
    wire          mul_valid [0:PAR_H-1];

    integer i, j;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done  <= 0;
            b <= 0; h <= 0; p <= 0; 
            // n <= 0;
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
                        //  n <= 0;
                        state <= CALC;
                    end
                end

                CALC: begin
                    valid_in <= 1;
                    // ê³±ì…ˆ ?…? ¥
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (h + i < H) begin
                            mul_in1[i] <= x[b*H*P + (h+i)*P + p];
                            mul_in2[i] <= dt[b*H + h + i];    
                        end
                        b_shift[i][0] <= b;
                        h_shift[i][0] <= h + i;
                        p_shift[i][0] <= p;
                        // n_shift[i][0] <= n + i;
                        for (j = 1; j < SHIFT_DEPTH; j = j + 1) begin
                            b_shift[i][j] <= b_shift[i][j-1];
                            h_shift[i][j] <= h_shift[i][j-1];
                            p_shift[i][j] <= p_shift[i][j-1];
                            // n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end

                    // ê²°ê³¼ ???¥
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (mul_valid[i]) begin
                            dx[b_shift[i][M_LAT]*H*P + h_shift[i][M_LAT]*P + p_shift[i][M_LAT]] <= mul_out[i];
                        end
                    end

                    // index ì¦ê?
                    if (p == P-1) begin
                        p <= 0;
                        if (h + PAR_H == H) begin
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
                            // n_shift[i][j] <= n_shift[i][j-1];
                        end
                    end
                    for (i = 0; i < PAR_H; i = i + 1) begin
                        if (mul_valid[i]) begin
                            dx[b_shift[i][M_LAT]*H*P + h_shift[i][M_LAT]*P + p_shift[i][M_LAT]] <= mul_out[i];
                        end
                    end
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
        for (g = 0; g < PAR_H; g = g + 1) begin : PIPELINE
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
endmodule
