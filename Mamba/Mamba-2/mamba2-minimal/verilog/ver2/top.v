// ============================================================
// dA: torch.Size([1, 24])
// dt: torch.Size([1, 24])
// x: torch.Size([1, 24, 64])
// B: torch.Size([1, 128])
// C: torch.Size([1, 128])
// D: torch.Size([24])
// h[i].ssm_state: torch.Size([1, 24, 64, 128])

// B = 1
// H = 24
// P = 64
// N = 128
// parameter B = 1, H = 24, P = 64, N = 128
// ============================================================

module ssm_block_fp16_top #(
    parameter B = 1, H = 4, P = 4, N = 4,
    parameter DW = 16,
    parameter M_LAT = 6,
    parameter A_LAT = 11
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    input  wire [B*H*DW-1:0]     dt_flat,
    input  wire [B*H*DW-1:0]     dA_flat,
    input  wire [B*N*DW-1:0]     Bmat_flat,
    input  wire [B*N*DW-1:0]     C_flat,
    input  wire [H*DW-1:0]       D_flat,
    input  wire [B*H*P*DW-1:0]   x_flat,
    input  wire [B*H*P*N*DW-1:0] h_prev_flat,

    output wire [B*H*P*DW-1:0]   y_flat,
    output reg  done
);

    // FSM stage �?�?
    reg [2:0] stage;
    localparam STAGE_IDLE   = 0,
               STAGE_DBX    = 1,
               STAGE_UPDATE = 2,
               STAGE_HC     = 3,
               STAGE_YCALC  = 4,
               STAGE_RESADD = 5,
               STAGE_DONE   = 6;

    // 중간 버퍼
    wire done_dBx, done_mul, done_add, done_out, done_res, done_hC, done_acc, done_xD;
    wire [B*H*P*N*DW-1:0] dBx_flat;
    wire [B*H*P*N*DW-1:0] h_next_flat;
    wire [B*H*P*DW-1:0]   y_tmp_flat;
    wire [B*H*P*DW-1:0]   y_sum_flat;
    wire [B*H*P*N*DW-1:0] h_mul_flat;
    wire [B*H*P*N*DW-1:0] hC_flat;
    wire [B*H*P*DW-1:0]   xD_flat;

    reg stage_dBx_prev, stage_dAh_dBx, stage_hC, stage_acc, stage_y;
    wire start_dBx, start_ssm, start_hc, start_output, start_residual;

    assign start_dBx      = (stage == STAGE_DBX)    && !stage_dBx_prev;
    assign start_ssm      = (stage == STAGE_UPDATE) && !stage_dAh_dBx;
    assign start_hc      =  (stage == STAGE_HC)     && !stage_hC;
    assign start_output   = (stage == STAGE_YCALC)  && !stage_acc;
    assign start_residual = (stage == STAGE_RESADD) && !stage_y;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stage <= STAGE_IDLE;
            done <= 0;
        end else begin
            case (stage)
                STAGE_IDLE: begin
                    done <= 0;
                    if (start)
                        stage <= STAGE_DBX;
                end
                STAGE_DBX:    if (done_dBx)    stage <= STAGE_UPDATE;
                STAGE_UPDATE: if (done_add)    stage <= STAGE_HC;
                STAGE_HC:     if (done_hC)     stage <= STAGE_YCALC;
                STAGE_YCALC:  if (done_acc)    stage <= STAGE_RESADD;
                STAGE_RESADD: if (done_res)    stage <= STAGE_DONE;
                STAGE_DONE: begin
                    done <= 1;
                    stage <= STAGE_IDLE;
                end
            endcase
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            stage_dBx_prev      <= 0;
            stage_dAh_dBx      <= 0;
            stage_hC    <= 0;
            stage_acc   <= 0;
            stage_y <= 0;
        end else begin
            stage_dBx_prev      <= (stage == STAGE_DBX);
            stage_dAh_dBx      <= (stage == STAGE_UPDATE);
            stage_hC    <= (stage == STAGE_HC);
            stage_acc   <= (stage == STAGE_YCALC);
            stage_y <= (stage == STAGE_RESADD);
        end
    end

    // dBx 계산
    dBx_calc_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT)) u_dBx (
        .clk(clk), .rst(rst), .start(start_dBx),
        .dt_flat(dt_flat), .Bmat_flat(Bmat_flat), .x_flat(x_flat),
        .dBx_flat(dBx_flat), .done(done_dBx)
    );

    dAh #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT)) u_dAh (
        .clk(clk), .rst(rst), .start(start_dBx), .dBx_sig(done_dBx),
        .dA_flat(dA_flat), .h_prev_flat(h_prev_flat),
        .h_mul_flat(h_mul_flat), .done(done_mul)
    );

    dAh_dBx #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .A_LAT(A_LAT)) u_dAh_dBx (
        .clk(clk), .rst(rst), .start1(done_dBx), .start2(done_mul),
        .h_mul_flat(h_mul_flat), .dBx_flat(dBx_flat),
        .h_next_flat(h_next_flat), .done(done_add)
    );

    // h_next = h_prev × dA + dBx
    // ssm_update_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_upd (
    //     .clk(clk), .rst(rst), .start(start_ssm),
    //     .dA_flat(dA_flat), .h_prev_flat(h_prev_flat), .dBx_flat(dBx_flat),
    //     .h_next_flat(h_next_flat), .done(done_update)
    // );

    hC #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT)) u_hC (
        .clk(clk), .rst(rst), .start(done_add),
        .C_flat(C_flat), .h_flat(h_next_flat),
        .hC_flat(hC_flat), .done(done_hC)
    );

    accumulator #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .ADD_LAT(A_LAT)) u_acc (
        .clk(clk), .rst(rst), .start(done_hC),
        .hC_flat(hC_flat),
        .hC_sum_flat(y_sum_flat), .done(done_acc)
    );

    // // y_tmp = einsum(h_next × C)
    // output_calc_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_out (
    //     .clk(clk), .rst(rst), .start(done_add),
    //     .h_flat(h_next_flat), .C_flat(C_flat),
    //     .y_flat(y_tmp_flat), .done(done_out)
    // );

    // xD = x*D
    xD #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT)) u_dxD (
        .clk(clk), .rst(rst), .start(done_add), .acc_sig(done_acc),
        .D_flat(D_flat), .x_flat(x_flat),
        .xD_flat(xD_flat), .done(done_xD)
    );

    y_res #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .A_LAT(A_LAT)) u_y_res (
        .clk(clk), .rst(rst), .start1(done_acc), .start2(done_xD),
        .y_in_flat(y_sum_flat), .xD_flat(xD_flat),
        .y_out_flat(y_flat), .done(done_res)
    );

    // y_out = y_tmp + D × x
    // residual_add_fp16 #(.B(B), .H(H), .P(P), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_res (
    //     .clk(clk), .rst(rst), .start(done_acc),
    //     .y_in_flat(y_sum_flat), .x_flat(x_flat), .D_flat(D_flat),
    //     .y_out_flat(y_flat), .done(done_res)
    // );

endmodule
