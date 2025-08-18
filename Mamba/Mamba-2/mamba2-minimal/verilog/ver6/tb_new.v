`timescale 1ns / 1ps

module testbench_fp16_wrapper;
    parameter B = 1, H = 24, P = 64, N = 128;
    // parameter B = 1, H = 24, P = 4, N = 32;
    // parameter H_tile = 12, P_tile = 16;
    parameter H_tile = 1, P_tile = 1;
    parameter DW = 16;

    reg clk;
    reg rst;
    reg start;
    wire done;

    reg  [B*H*DW-1:0]        dt_flat;
    reg  [B*H*DW-1:0]        dA_flat;
    reg  [B*N*DW-1:0]        Bmat_flat;
    reg  [B*N*DW-1:0]        C_flat;
    reg  [H*DW-1:0]          D_flat;
    reg  [B*H*P*DW-1:0]      x_flat;
    reg  [B*H*P*N*DW-1:0]    h_prev_flat;
    wire [B*H*P*DW-1:0]      y_flat_out;

    reg [DW-1:0] dt_mem     [0:B*H-1];
    reg [DW-1:0] dA_mem     [0:B*H-1];
    reg [DW-1:0] Bmat_mem   [0:B*N-1];
    reg [DW-1:0] C_mem      [0:B*N-1];
    reg [DW-1:0] D_mem      [0:H-1];
    reg [DW-1:0] x_mem      [0:B*H*P-1];
    reg [DW-1:0] h_prev_mem [0:B*H*P*N-1];

    // reg [DW-1:0] y_flat_out_mem    [0:B*H*P-1];

    integer i;

    packing #(
        .B(B), .H(H), .P(P), .N(N),
        .H_tile(H_tile), .P_tile(P_tile), .DW(DW)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .dt_flat_in(dt_flat), .dA_flat_in(dA_flat),
        .Bmat_flat_in(Bmat_flat), .C_flat_in(C_flat), .D_flat_in(D_flat),
        .x_flat_in(x_flat), .h_prev_flat_in(h_prev_flat),
        .y_flat_out(y_flat_out)
    );

    always #5 clk = ~clk;

    integer fout;
    
    initial begin
        $display("==== FP16 SSM Block Full Wrapper Testbench ====");
        clk = 0; rst = 1; start = 0;

        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",     dt_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",     dA_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",      Bmat_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",      C_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",      D_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",      x_mem);
        $readmemh("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex", h_prev_mem);

         // Flatten
        for (i = 0; i < B*H; i = i + 1) begin
            dt_flat[DW*i +: DW] = dt_mem[i];
            dA_flat[DW*i +: DW] = dA_mem[i];
        end
        for (i = 0; i < B*N; i = i + 1) begin
            Bmat_flat[DW*i +: DW] = Bmat_mem[i];
            C_flat[DW*i +: DW]    = C_mem[i];
        end
        for (i = 0; i < H; i = i + 1)
            D_flat[DW*i +: DW] = D_mem[i];

        for (i = 0; i < B*H*P; i = i + 1)
            x_flat[DW*i +: DW] = x_mem[i];

        for (i = 0; i < B*H*P*N; i = i + 1)
            h_prev_flat[DW*i +: DW] = h_prev_mem[i];

        #10 rst = 0;
        #10 start = 1;
        #10 start = 0;

        wait(done);
        #10;

        $display("✅ Wrapper done. Writing result...");
        
        fout = $fopen("/home/intern-2501//internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out.hex", "w");
        for (i = 0; i < B*H*P; i = i + 1)
            $fdisplay(fout, "%04h", y_flat_out[DW*i +: DW]);
        $fclose(fout);

        $display("✅ Output saved. Simulation completed.");
        #10 $finish;
    end

endmodule


module packing #(
    parameter B = 1,
    parameter H = 24,
    parameter P = 64,
    parameter N = 128,
    parameter H_tile = 1,
    parameter P_tile = 1,
    parameter N_TILE = 16,
    parameter DW = 16,
    // 누적 모드: 0이면 마지막 n블록 결과만 출력, 1이면 타일 내부 누적(별도 FP16 adder 필요)
    parameter ACCUMULATE = 0
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
    localparam N_BLKS     = N / N_TILE;

    // tile indices
    reg [$clog2(NUM_TILE_H)-1:0] h_idx;
    reg [$clog2(NUM_TILE_P)-1:0] p_idx;
    reg [$clog2(N_BLKS)    -1:0] n_blk;

    reg start_top;
    wire done_top;

    // sliced tile inputs/outputs
    reg  [B*H_tile*DW-1:0]                     dt_tile;
    reg  [B*H_tile*DW-1:0]                     dA_tile;
    reg  [H_tile*DW-1:0]                       D_tile;
    reg  [B*H_tile*P_tile*DW-1:0]              x_tile;
    reg  [B*H_tile*P_tile*N_TILE*DW-1:0]       h_prev_tile;   // N_TILE만큼
    reg  [B*N_TILE*DW-1:0]                     Bmat_tile;     // N_TILE만큼
    reg  [B*N_TILE*DW-1:0]                     C_tile;        // N_TILE만큼
    wire [B*H_tile*P_tile*DW-1:0]              y_tile;

    // (옵션) 누적 버퍼
    reg  [B*H_tile*P_tile*DW-1:0]              y_accum;
    wire [B*H_tile*P_tile*DW-1:0]              y_accum_next;

    // 누적 모드가 아니면 패스스루
    generate
        if (ACCUMULATE == 0) begin : g_no_accum
            assign y_accum_next = y_tile; // 누적 없음
        end else begin : g_do_accum
            // 여기에 FP16 벡터-요소별 덧셈을 연결하세요.
            // 예: fp16_add elemwise (y_accum, y_tile) -> y_accum_next
            // 아래는 자릿수만 맞춘 placeholder(합성 불가). 실제 FP16 adder 인스턴스와 연결 필요.
            genvar k;
            for (k = 0; k < B*H_tile*P_tile; k = k + 1) begin : g_add_each
                // fp16_adder u_add(.clk(clk), .a(y_accum[DW*k +: DW]), .b(y_tile[DW*k +: DW]), .sum(y_accum_next[DW*k +: DW]));
                assign y_accum_next[DW*k +: DW] = y_tile[DW*k +: DW]; // placeholder
            end
        end
    endgenerate

    ssm_block_fp16_top #(
        .B(B), .H(H_tile), .P(P_tile), .N(N_TILE), .DW(DW)
    ) u_tile (
        .clk(clk), .rst(rst), .start(start_top),
        .dt_flat(dt_tile), .dA_flat(dA_tile),
        .Bmat_flat(Bmat_tile),
        .C_flat(C_tile),
        .D_flat(D_tile),
        .x_flat(x_tile),
        .h_prev_flat(h_prev_tile),
        .y_flat(y_tile),
        .done(done_top)
    );
    SSMBLOCK_TOP #(
        parameter integer DW        = 16,
        parameter integer N_TILE    = 16,
        parameter integer N_TOTAL   = 128,
        parameter integer LAT_DX_M  = 6,   // dx: dt*x (mul)
        parameter integer LAT_DBX_M = 6,   // dBx: dx*B (mul)
        parameter integer LAT_DAH_M = 6,   // dAh: dA*hprev (mul)
        parameter integer LAT_ADD_A = 11,  // h_next: dAh+dBx
        parameter integer LAT_HC_M  = 6    // hC: h_next*C (mul)
    )(
        .clk(clk), .rstn(rst), .start(start_top),
        tile_valid_i,
        .tile_ready_o(1'b1),   // 필요시 backpressure, 기본 1
        .dt_i(dt_tile), .dA_i(dA_tile), .x_i(x_tile), .D_i(D_tile),
        .B_tile_i(Bmat_tile), .C_tile_i(C_tile), .hprev_tile_i(h_prev_tile),
        .y_final_o(y_tile), .y_final_valid_o(y_final_valid_o)
    );

    integer i, h_rel, p_rel, h_abs, p_abs, n, hp_idx, n_local, n_abs;

    // 상태기계
    reg [2:0] state;
    localparam IDLE=0, KICK=1, RUN=2, ACCUM=3, NEXT_N=4, WRITE=5, NEXT_HP=6, DONE=7;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            h_idx      <= 0;
            p_idx      <= 0;
            n_blk      <= 0;
            start_top  <= 0;
            done       <= 0;
            y_flat_out <= 0;
            y_accum    <= 0;
            state      <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        h_idx     <= 0;
                        p_idx     <= 0;
                        n_blk     <= 0;
                        y_accum   <= {B*H_tile*P_tile*DW{1'b0}}; // 새 (h,p) 시작 시 초기화
                        state     <= KICK;
                        start_top <= 1;
                    end
                end

                KICK: begin
                    start_top <= 0;      // 1사이클 펄스
                    state     <= RUN;
                end

                RUN: begin
                    if (done_top) begin
                        state <= ACCUM;
                    end
                end

                ACCUM: begin
                    // 누적 모드: 이전 누적 + 이번 블록 결과
                    // 비누적 모드: 이번 블록 결과를 y_accum_next에 그대로 전달
                    y_accum <= y_accum_next;
                    state   <= NEXT_N;
                end

                NEXT_N: begin
                    if (n_blk == N_BLKS - 1) begin
                        // N 전부 처리, 이제 쓰기
                        state <= WRITE;
                    end else begin
                        // 다음 N 블록
                        n_blk     <= n_blk + 1;
                        start_top <= 1;
                        state     <= KICK;
                    end
                end

                WRITE: begin
                    // 최종 결과 기록: 누적 모드면 누적된 값, 아니면 마지막 블록(y_tile)이 y_accum에 들어 있음
                    for (i = 0; i < H_tile * P_tile; i = i + 1) begin
                        h_rel = i / P_tile;
                        p_rel = i % P_tile;
                        h_abs = h_idx * H_tile + h_rel;
                        p_abs = p_idx * P_tile + p_rel;
                        y_flat_out[DW*(h_abs*P + p_abs) +: DW] <= y_accum[DW*i +: DW];
                    end
                    state <= NEXT_HP;
                end

                NEXT_HP: begin
                    // 다음 (h,p)로 이동
                    if (p_idx == NUM_TILE_P - 1) begin
                        p_idx <= 0;
                        if (h_idx == NUM_TILE_H - 1) begin
                            state <= DONE;
                        end else begin
                            h_idx   <= h_idx + 1;
                            n_blk   <= 0;
                            y_accum <= {B*H_tile*P_tile*DW{1'b0}};
                            start_top <= 1;
                            state   <= KICK;
                        end
                    end else begin
                        p_idx   <= p_idx + 1;
                        n_blk   <= 0;
                        y_accum <= {B*H_tile*P_tile*DW{1'b0}};
                        start_top <= 1;
                        state   <= KICK;
                    end
                end

                DONE: begin
                    done  <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    // ---------------- tile slicing logic ----------------
    // H, P는 (h_idx, p_idx)로 슬라이스. N은 (n_blk)로 N_TILE 슬라이스.
    always @(*) begin
        // dt, dA, D: H 타일 슬라이스
        for (i = 0; i < H_tile; i = i + 1) begin
            dt_tile[DW*i +: DW] = dt_flat_in[DW*((h_idx*H_tile)+i) +: DW];
            dA_tile[DW*i +: DW] = dA_flat_in[DW*((h_idx*H_tile)+i) +: DW];
            D_tile[DW*i +: DW]  = D_flat_in [DW*((h_idx*H_tile)+i) +: DW];
        end

        // x: (H_tile, P_tile) 슬라이스
        for (i = 0; i < H_tile*P_tile; i = i + 1) begin
            h_rel = i / P_tile;
            p_rel = i % P_tile;
            h_abs = h_idx * H_tile + h_rel;
            p_abs = p_idx * P_tile + p_rel;
            x_tile[DW*i +: DW] = x_flat_in[DW*(h_abs*P + p_abs) +: DW];
        end

        // B, C: N_TILE 슬라이스 (배치 B 차원은 1로 가정)
        for (n_local = 0; n_local < N_TILE; n_local = n_local + 1) begin
            n_abs = n_blk * N_TILE + n_local;
            Bmat_tile[DW*n_local +: DW] = Bmat_flat_in[DW*n_abs +: DW];
            C_tile   [DW*n_local +: DW] = C_flat_in   [DW*n_abs +: DW];
        end

        // h_prev: (H_tile * P_tile, N_TILE) 슬라이스
        for (i = 0; i < H_tile*P_tile*N_TILE; i = i + 1) begin
            hp_idx = i / N_TILE;          // 0 .. H_tile*P_tile-1
            n_local = i % N_TILE;         // 0 .. N_TILE-1
            n_abs = n_blk * N_TILE + n_local;

            h_rel = hp_idx / P_tile;
            p_rel = hp_idx % P_tile;
            h_abs = h_idx * H_tile + h_rel;
            p_abs = p_idx * P_tile + p_rel;

            h_prev_tile[DW*i +: DW] = h_prev_flat_in[DW*((h_abs*P + p_abs)*N + n_abs) +: DW];
        end
    end

endmodule
