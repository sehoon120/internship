`timescale 1ns / 1ps

module packing #(
    parameter B = 1,
    parameter H = 24,
    parameter P = 64,
    parameter N = 128,
    parameter H_tile = 12,
    parameter P_tile = 16,
    parameter DW = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire start,  // HIGH for 1 cycle
    output reg  done,

    input  wire [B*H*DW-1:0]        dt_flat_in,
    input  wire [B*H*DW-1:0]        dA_flat_in,
    input  wire [B*N*DW-1:0]        Bmat_flat_in,
    input  wire [B*N*DW-1:0]        C_flat_in,
    input  wire [H*DW-1:0]          D_flat_in,
    input  wire [B*H*P*DW-1:0]      x_flat_in,
    input  wire [B*H*P*N*DW-1:0]    h_prev_flat_in,
    output reg  [B*H*P*DW-1:0]      y_flat_out
);

    localparam NUM_TILE_H = H / H_tile;
    localparam NUM_TILE_P = P / P_tile;

    reg [3:0] h_idx, p_idx;
    reg start_top;
    wire done_top;

    // sliced tile inputs/outputs
    reg  [B*H_tile*DW-1:0]     dt_tile;
    reg  [B*H_tile*DW-1:0]     dA_tile;
    reg  [H_tile*DW-1:0]       D_tile;
    reg  [B*H_tile*P_tile*DW-1:0] x_tile;
    reg  [B*H_tile*P_tile*N*DW-1:0] h_prev_tile;
    wire [B*H_tile*P_tile*DW-1:0] y_tile;

    ssm_block_fp16_top #(
        .B(B), .H(H_tile), .P(P_tile), .N(N), .DW(DW)
    ) u_tile (
        .clk(clk), .rst(rst), .start(start_top),
        .dt_flat(dt_tile), .dA_flat(dA_tile),
        .Bmat_flat(Bmat_flat_in),
        .C_flat(C_flat_in),
        .D_flat(D_tile),
        .x_flat(x_tile),
        .h_prev_flat(h_prev_tile),
        .y_flat(y_tile),
        .done(done_top)
    );

    integer i, h_rel, p_rel, h_abs, p_abs, n, hp_idx;
    reg [1:0] state;
    localparam IDLE=0, RUN=1, WRITE=2, DONE=3;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            h_idx <= 0; p_idx <= 0;
            start_top <= 0;
            done <= 0;
            y_flat_out <= 0;
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        h_idx <= 0;
                        p_idx <= 0;
                        state <= RUN;
                        start_top <= 1;
                    end
                end
                RUN: begin
                    start_top <= 0;
                    if (done_top) begin
                        state <= WRITE;
                    end
                end
                WRITE: begin
                    for (i = 0; i < H_tile * P_tile; i = i + 1) begin
                        h_rel = i / P_tile;
                        p_rel = i % P_tile;
                        h_abs = h_idx * H_tile + h_rel;
                        p_abs = p_idx * P_tile + p_rel;
                        y_flat_out[DW*(h_abs*P + p_abs) +: DW] <= y_tile[DW*i +: DW];
                    end
                    if (p_idx == NUM_TILE_P - 1) begin
                        p_idx <= 0;
                        if (h_idx == NUM_TILE_H - 1) begin
                            state <= DONE;
                        end else begin
                            h_idx <= h_idx + 1;
                            state <= RUN;
                            start_top <= 1;
                        end
                    end else begin
                        p_idx <= p_idx + 1;
                        state <= RUN;
                        start_top <= 1;
                    end
                end
                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // tile slicing logic
    always @(*) begin
        for (i = 0; i < H_tile; i = i + 1) begin
            dt_tile[DW*i +: DW] = dt_flat_in[DW*((h_idx*H_tile)+i) +: DW];
            dA_tile[DW*i +: DW] = dA_flat_in[DW*((h_idx*H_tile)+i) +: DW];
            D_tile[DW*i +: DW]  = D_flat_in[DW*((h_idx*H_tile)+i) +: DW];
        end
        for (i = 0; i < H_tile*P_tile; i = i + 1) begin
            h_rel = i / P_tile;
            p_rel = i % P_tile;
            h_abs = h_idx * H_tile + h_rel;
            p_abs = p_idx * P_tile + p_rel;
            x_tile[DW*i +: DW] = x_flat_in[DW*(h_abs*P + p_abs) +: DW];
        end
        for (i = 0; i < H_tile*P_tile*N; i = i + 1) begin
            hp_idx = i / N;
            n = i % N;
            h_rel = hp_idx / P_tile;
            p_rel = hp_idx % P_tile;
            h_abs = h_idx * H_tile + h_rel;
            p_abs = p_idx * P_tile + p_rel;
            h_prev_tile[DW*i +: DW] = h_prev_flat_in[DW*((h_abs*P + p_abs)*N + n) +: DW];
        end
    end

endmodule