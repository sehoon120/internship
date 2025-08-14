`timescale 1ns / 1ps

// packing + tb 기능을 합친 순수 Verilog 테스트벤치
module tb_packing_merged;
    parameter B = 1, H = 24, P = 64, N = 128;
    parameter H_tile = 12, P_tile = 64;   // 타일 크기
    parameter DW = 16;

    // 파생 파라미터
    localparam NUM_TILE_H = H / H_tile;
    localparam NUM_TILE_P = P / P_tile;

    // 유효성 체크 (시뮬 시간 출력용)
    initial begin
        if (H % H_tile != 0) begin
            $display("ERROR: H(%0d) %% H_tile(%0d) != 0", H, H_tile);
            $finish;
        end
        if (P % P_tile != 0) begin
            $display("ERROR: P(%0d) %% P_tile(%0d) != 0", P, P_tile);
            $finish;
        end
    end

    // 클럭/리셋/스타트
    reg  clk;
    reg  rst;
    reg  start;    // 전체 실행 트리거
    reg  done;     // 모든 타일 완료 (TB에서 생성)

    // 전체 플랫 버스 (입력은 고정, 출력은 타일마다 채워넣음)
    reg  [B*H*DW-1:0]        dt_flat;
    reg  [B*H*DW-1:0]        dA_flat;
    reg  [B*N*DW-1:0]        Bmat_flat;
    reg  [B*N*DW-1:0]        C_flat;
    reg  [H*DW-1:0]          D_flat;
    reg  [B*H*P*DW-1:0]      x_flat;
    reg  [B*H*P*N*DW-1:0]    h_prev_flat;
    reg  [B*H*P*DW-1:0]      y_flat_out;   // 최종 어셈블 버퍼

    // 파일 로딩용 메모리
    reg [DW-1:0] dt_mem     [0:B*H-1];
    reg [DW-1:0] dA_mem     [0:B*H-1];
    reg [DW-1:0] Bmat_mem   [0:B*N-1];
    reg [DW-1:0] C_mem      [0:B*N-1];
    reg [DW-1:0] D_mem      [0:H-1];
    reg [DW-1:0] x_mem      [0:B*H*P-1];
    reg [DW-1:0] h_prev_mem [0:B*H*P*N-1];

    // 타일 I/O (packing이 하던 분배/수집을 TB에서 수행)
    reg                   start_top;
    wire                  done_top;

    reg  [B*H_tile*DW-1:0]           dt_tile;
    reg  [B*H_tile*DW-1:0]           dA_tile;
    reg  [H_tile*DW-1:0]             D_tile;
    reg  [B*H_tile*P_tile*DW-1:0]    x_tile;
    reg  [B*H_tile*P_tile*N*DW-1:0]  h_prev_tile;
    wire [B*H_tile*P_tile*DW-1:0]    y_tile;

    // DUT: 패킹 없이 타일만 먹임 (포트명은 실제 모듈에 맞춰 사용)
    ssm_block_fp16_top #(
        .B(B), .H(H_tile), .P(P_tile), .N(N), .DW(DW)
    ) dut (
        .clk(clk), .rst(rst), .start(start_top),
        .dt_flat(dt_tile), .dA_flat(dA_tile),
        .Bmat_flat(Bmat_flat),  // 타일 경계 무관 → 공유
        .C_flat(C_flat),        // 타일 경계 무관 → 공유
        .D_flat(D_tile),
        .x_flat(x_tile),
        .h_prev_flat(h_prev_tile),
        .y_flat(y_tile),
        .done(done_top)
    );

    // 100MHz
    always #5 clk = ~clk;

    integer i, fout;

    // 초기화/로드/플래튼
    initial begin
        $display("==== FP16 SSM Block (TILED, NO PACKING) Testbench [pure Verilog] ====");
        clk = 0; rst = 1; start = 0; done = 0; start_top = 0;
        y_flat_out = {B*H*P*DW{1'b0}};

        // 경로는 환경 맞게 수정
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",        dt_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",        dA_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",         Bmat_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",         C_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",         D_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",         x_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex", h_prev_mem);

        // 전체 플랫으로 복사(슬라이싱은 아래 조합논리에서)
        for (i = 0; i < B*H; i = i + 1) begin
            dt_flat [DW*i +: DW] = dt_mem[i];
            dA_flat [DW*i +: DW] = dA_mem[i];
        end
        for (i = 0; i < B*N; i = i + 1) begin
            Bmat_flat[DW*i +: DW] = Bmat_mem[i];
            C_flat   [DW*i +: DW] = C_mem[i];
        end
        for (i = 0; i < H; i = i + 1)
            D_flat[DW*i +: DW] = D_mem[i];

        for (i = 0; i < B*H*P; i = i + 1)
            x_flat[DW*i +: DW] = x_mem[i];

        for (i = 0; i < B*H*P*N; i = i + 1)
            h_prev_flat[DW*i +: DW] = h_prev_mem[i];

        // reset & global start
        #20 rst = 0;
        #20 start = 1;
        #10 start = 0;
    end

    // ========= 타일 슬라이싱(조합) =========
    reg [31:0] h_idx, p_idx;
    integer t, h_rel, p_rel, h_abs, p_abs, n, hp_idx;

    always @(*) begin
        // dt/dA/D : h만 따라감
        for (t = 0; t < H_tile; t = t + 1) begin
            dt_tile[DW*t +: DW] = dt_flat[DW*((h_idx*H_tile) + t) +: DW];
            dA_tile[DW*t +: DW] = dA_flat[DW*((h_idx*H_tile) + t) +: DW];
            D_tile [DW*t +: DW] = D_flat [DW*((h_idx*H_tile) + t) +: DW];
        end
        // x : (h,p) 따라감
        for (t = 0; t < H_tile*P_tile; t = t + 1) begin
            h_rel = t / P_tile;
            p_rel = t % P_tile;
            h_abs = h_idx * H_tile + h_rel;
            p_abs = p_idx * P_tile + p_rel;
            x_tile[DW*t +: DW] = x_flat[DW*(h_abs*P + p_abs) +: DW];
        end
        // h_prev : (h,p,n) 따라감
        for (t = 0; t < H_tile*P_tile*N; t = t + 1) begin
            hp_idx = t / N;       // 0 .. (H_tile*P_tile-1)
            n      = t % N;       // 0 .. (N-1)
            h_rel  = hp_idx / P_tile;
            p_rel  = hp_idx % P_tile;
            h_abs  = h_idx * H_tile + h_rel;
            p_abs  = p_idx * P_tile + p_rel;
            h_prev_tile[DW*t +: DW] = h_prev_flat[DW*(((h_abs*P) + p_abs)*N + n) +: DW];
        end
    end

    // ========= FSM: 타일 순회 & 어셈블 (순수 Verilog: parameter 상태) =========
    parameter S_IDLE  = 3'd0;
    parameter S_PULSE = 3'd1;
    parameter S_WAIT  = 3'd2;
    parameter S_WRITE = 3'd3;
    parameter S_NEXT  = 3'd4;
    parameter S_DONE  = 3'd5;

    reg [2:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            h_idx <= 0; p_idx <= 0;
            start_top <= 0;
            done <= 0;
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        h_idx <= 0; p_idx <= 0;
                        start_top <= 1;          // 1사이클 펄스
                        state <= S_PULSE;
                    end
                end
                S_PULSE: begin
                    start_top <= 0;
                    state <= S_WAIT;
                end
                S_WAIT: begin
                    if (done_top) begin
                        state <= S_WRITE;
                    end
                end
                S_WRITE: begin
                    // y_tile → y_flat_out 어셈블 (정확한 (h_abs,p_abs)에 기록)
                    for (t = 0; t < H_tile*P_tile; t = t + 1) begin
                        h_rel = t / P_tile;
                        p_rel = t % P_tile;
                        h_abs = h_idx * H_tile + h_rel;
                        p_abs = p_idx * P_tile + p_rel;
                        y_flat_out[DW*(h_abs*P + p_abs) +: DW] <= y_tile[DW*t +: DW];
                    end
                    state <= S_NEXT;
                end
                S_NEXT: begin
                    if (p_idx == NUM_TILE_P - 1) begin
                        p_idx <= 0;
                        if (h_idx == NUM_TILE_H - 1) begin
                            state <= S_DONE;
                        end else begin
                            h_idx <= h_idx + 1;
                            start_top <= 1;
                            state <= S_PULSE;
                        end
                    end else begin
                        p_idx <= p_idx + 1;
                        start_top <= 1;
                        state <= S_PULSE;
                    end
                end
                S_DONE: begin
                    done <= 1;        // 모든 타일 완료
                    state <= S_IDLE;  // 필요시 반복 실행 가능
                end
            endcase
        end
    end

    // ========= 결과 덤프 =========
    initial begin : DUMP_BLOCK
        wait(done);
        #10;
        $display("✅ All tiles done. Writing result...");
        fout = $fopen("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out.hex", "w");
        for (i = 0; i < B*H*P; i = i + 1)
            $fdisplay(fout, "%04h", y_flat_out[DW*i +: DW]);
        $fclose(fout);
        $display("✅ Output saved. Simulation completed.");
        #10 $finish;
    end
endmodule
