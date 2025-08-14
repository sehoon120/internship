`timescale 1ns / 1ps

// tile-in (1*1*1*N_TILE*DW) 순차 스트리밍 TB (B=H=P=1 고정)
module tb_ssmblock_tile_stream;
    // -----------------------------
    // Params
    // -----------------------------
    localparam integer DW      = 16;
    localparam integer N       = 128;
    localparam integer N_TILE  = 16;
    localparam integer TILES   = N / N_TILE; // == 8

    // -----------------------------
    // DUT I/O
    // -----------------------------
    reg                      clk;
    reg                      rstn;

    reg                      tile_valid_i;
    wire                     tile_ready_o;

    reg  [DW-1:0]            dt_i;
    reg  [DW-1:0]            dA_i;
    reg  [DW-1:0]            x_i;
    reg  [DW-1:0]            D_i;

    reg  [N_TILE*DW-1:0]     B_tile_i;
    reg  [N_TILE*DW-1:0]     C_tile_i;
    reg  [N_TILE*DW-1:0]     hprev_tile_i;

    wire [DW-1:0]            y_final_o;
    wire                     y_final_valid_o;

    // -----------------------------
    // Memories (필요한 만큼만 선언)
    // -----------------------------
    reg [DW-1:0] dt_mem   [0:0];        // dt[0]
    reg [DW-1:0] dA_mem   [0:0];        // dA[0]
    reg [DW-1:0] D_mem    [0:0];        // D[0]
    reg [DW-1:0] x_mem    [0:0];        // x[0]
    reg [DW-1:0] B_mem    [0:N-1];      // B[0..127]
    reg [DW-1:0] C_mem    [0:N-1];      // C[0..127]
    reg [DW-1:0] h_mem    [0:N-1];      // h_prev[0..127] (H=P=1 가정)

    integer i, j, base, fout;

    // -----------------------------
    // DUT
    // -----------------------------
    SSMBLOCK_TOP #(
        .DW(DW), .N_TILE(N_TILE), .N_TOTAL(N),
        .LAT_DX_M(6), .LAT_DBX_M(6), .LAT_DAH_M(6),
        .LAT_ADD_A(11), .LAT_HC_M(6)
    ) dut (
        .clk(clk),
        .rstn(rstn),

        .tile_valid_i(tile_valid_i),
        .tile_ready_o(tile_ready_o),

        .dt_i(dt_i),
        .dA_i(dA_i),
        .x_i(x_i),
        .D_i(D_i),

        .B_tile_i(B_tile_i),
        .C_tile_i(C_tile_i),
        .hprev_tile_i(hprev_tile_i),

        .y_final_o(y_final_o),
        .y_final_valid_o(y_final_valid_o)
    );

    // -----------------------------
    // 100 MHz clock
    // -----------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -----------------------------
    // Reset & init
    // -----------------------------
    initial begin
        rstn = 1'b0;
        tile_valid_i = 1'b0;
        B_tile_i = 0; C_tile_i = 0; hprev_tile_i = 0;
        dt_i = 0; dA_i = 0; x_i = 0; D_i = 0;

        // 경로는 환경에 맞게 수정하세요.
        // 파일이 더 큰 사이즈여도, 여기 메모리 크기만큼만 앞에서부터 읽힙니다.
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dt.hex",        dt_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_dA.hex",        dA_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_D.hex",         D_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_x.hex",         x_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_B.hex",         B_mem);
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_C.hex",         C_mem);
        // h_prev는 원래 B*H*P*N 크기였지만, 여기선 H=P=1만 사용 → 앞 N개만 로드하면 충분
        $readmemh("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_ssm_state.hex", h_mem);

        // 스칼라들 세팅(단 1개씩 사용)
        dt_i = dt_mem[0];
        dA_i = dA_mem[0];
        D_i  = D_mem[0];
        x_i  = x_mem[0];

        // release reset
        #100;
        rstn = 1'b1;

        // 한 템포
        @(posedge clk);
        @(posedge clk);

        // -------------------------
        // 타일 스트리밍 시작 (II=1)
        // 매 클록 base=i*N_TILE에서 16개씩 버스에 적재하고 valid=1
        // -------------------------
        for (i = 0; i < TILES; i = i + 1) begin
            base = i * N_TILE;

            // 타일 버스 패킹
            for (j = 0; j < N_TILE; j = j + 1) begin
                B_tile_i     [DW*j +: DW] = B_mem[base + j];
                C_tile_i     [DW*j +: DW] = C_mem[base + j];
                hprev_tile_i [DW*j +: DW] = h_mem [base + j];
            end

            // 핸드셰이크 (현재 DUT는 tile_ready_o=1이지만, 일반화)
            wait (tile_ready_o == 1'b1);
            tile_valid_i = 1'b1;
            @(posedge clk);
            tile_valid_i = 1'b0; // 1사이클 펄스(II=1). 지속 1로 두고 매 클록 교체해도 무방.
        end

        // 모든 타일 전송 완료 → 결과 대기
        wait (y_final_valid_o == 1'b1);
        $display("✅ y_final = %04h at time %0t", y_final_o, $time);

        // (옵션) 결과 파일로 저장
        fout = $fopen("/home/intern-2501/internship/Mamba/Mamba-2/mamba2-minimal/verilog/intermediate_datas/0_y_out_scalar.hex", "w");
        $fdisplay(fout, "%04h", y_final_o);
        $fclose(fout);

        #50;
        $finish;
    end
endmodule
