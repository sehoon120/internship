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

    // FSM stage 관리
    reg [2:0] stage;
    localparam STAGE_IDLE   = 0,
               STAGE_DBX    = 1,
               STAGE_UPDATE = 2,
               STAGE_YCALC  = 3,
               STAGE_RESADD = 4,
               STAGE_DONE   = 5;

    // 중간 버퍼
    wire done_dBx, done_update, done_out, done_res;
    wire [B*H*P*N*DW-1:0] dBx_flat;
    wire [B*H*P*N*DW-1:0] h_next_flat;
    wire [B*H*P*DW-1:0]   y_tmp_flat;

    reg stage_dBx_prev, stage_ssm_prev, stage_output_prev, stage_residual_prev;
    wire start_dBx, start_ssm, start_output, start_residual;

    // 펄스 생성: 해당 단계로 처음 진입한 순간에만 1
    assign start_dBx      = (stage == STAGE_DBX)    && !stage_dBx_prev;
    assign start_ssm      = (stage == STAGE_UPDATE) && !stage_ssm_prev;
    assign start_output   = (stage == STAGE_YCALC)  && !stage_output_prev;
    assign start_residual = (stage == STAGE_RESADD) && !stage_residual_prev;

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
                STAGE_UPDATE: if (done_update) stage <= STAGE_YCALC;
                STAGE_YCALC:  if (done_out)    stage <= STAGE_RESADD;
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
            stage_ssm_prev      <= 0;
            stage_output_prev   <= 0;
            stage_residual_prev <= 0;
        end else begin
            stage_dBx_prev      <= (stage == STAGE_DBX);
            stage_ssm_prev      <= (stage == STAGE_UPDATE);
            stage_output_prev   <= (stage == STAGE_YCALC);
            stage_residual_prev <= (stage == STAGE_RESADD);
        end
    end

    // dBx 계산
    dBx_calc_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT)) u_dBx (
        .clk(clk), .rst(rst), .start(start_dBx),
        .dt_flat(dt_flat), .Bmat_flat(Bmat_flat), .x_flat(x_flat),
        .dBx_flat(dBx_flat), .done(done_dBx)
    );

    // h_next = h_prev × dA + dBx
    ssm_update_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_upd (
        .clk(clk), .rst(rst), .start(start_ssm),
        .dA_flat(dA_flat), .h_prev_flat(h_prev_flat), .dBx_flat(dBx_flat),
        .h_next_flat(h_next_flat), .done(done_update)
    );

    // y_tmp = einsum(h_next × C)
    output_calc_fp16 #(.B(B), .H(H), .P(P), .N(N), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_out (
        .clk(clk), .rst(rst), .start(start_output),
        .h_flat(h_next_flat), .C_flat(C_flat),
        .y_flat(y_tmp_flat), .done(done_out)
    );

    // y_out = y_tmp + D × x
    residual_add_fp16 #(.B(B), .H(H), .P(P), .DW(DW), .M_LAT(M_LAT), .A_LAT(A_LAT)) u_res (
        .clk(clk), .rst(rst), .start(start_residual),
        .y_in_flat(y_tmp_flat), .x_flat(x_flat), .D_flat(D_flat),
        .y_out_flat(y_flat), .done(done_res)
    );

endmodule
