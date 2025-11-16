`timescale 1 ns / 1 ps

module accel_axis_wrapper #(
    // SSMBLOCK_TOP와 동일한 파라미터
    parameter integer DW          = 16,
    parameter integer H_TILE      = 1,
    parameter integer P_TILE      = 1,
    parameter integer N_TILE      = 128,
    parameter integer P_TOTAL     = 64,
    parameter integer N_TOTAL     = 128,

    // AXIS 데이터 폭 (DMA 쪽과 맞춰야 함)
    parameter integer AXIS_IN_DATA_WIDTH  = 256,   // MM2S 방향
    parameter integer AXIS_OUT_DATA_WIDTH = 256    // S2MM 방향
)(
    input  logic clk,
    input  logic rstn,

    // DMA -> 가속기 (MM2S, AXIS Slave 입장)
    axis.slave  axis_mm2s,

    // 가속기 -> DMA (S2MM, AXIS Master 입장)
    axis.master axis_s2mm
);

    // =========================================================
    // 1. 로컬 파라미터: SSMBLOCK_TOP 입출력 폭 계산
    // =========================================================
    localparam int DT_W        = H_TILE * DW;
    localparam int DT_BIAS_W   = H_TILE * DW;
    localparam int A_W         = H_TILE * DW;
    localparam int X_W         = H_TILE * P_TILE * DW;
    localparam int D_W         = H_TILE * DW;
    localparam int B_TILE_W    = N_TILE * DW;
    localparam int C_TILE_W    = N_TILE * DW;
    localparam int HPREV_W     = H_TILE * P_TILE * N_TILE * DW;

    localparam int TOTAL_IN_W  = DT_W + DT_BIAS_W + A_W + X_W + D_W
                               + B_TILE_W + C_TILE_W + HPREV_W;

    localparam int Y_W         = H_TILE * P_TILE * DW;

    // AXIS 폭이 기대와 일치하는지 간단 체크 (시뮬레이터에서 에러로 확인)
    initial begin
        if (AXIS_IN_DATA_WIDTH != TOTAL_IN_W) begin
            $error("AXIS_IN_DATA_WIDTH (%0d) != TOTAL_IN_W (%0d). \
                    래퍼의 입력 패킹 규칙에 맞춰 DATA_WIDTH를 조정해야 합니다.",
                    AXIS_IN_DATA_WIDTH, TOTAL_IN_W);
        end
        if (AXIS_OUT_DATA_WIDTH != Y_W) begin
            $error("AXIS_OUT_DATA_WIDTH (%0d) != Y_W (%0d). \
                    y_final_o 폭과 AXIS 출력 폭이 일치해야 합니다.",
                    AXIS_OUT_DATA_WIDTH, Y_W);
        end
    end

    // =========================================================
    // 2. SSMBLOCK_TOP에 연결될 실제 신호들
    // =========================================================
    logic                             tile_valid_i;
    logic                             tile_ready_o;  // SSMBLOCK_TOP에서 생성
    logic [DT_W      -1:0]            dt_i;
    logic [DT_BIAS_W-1:0]            dt_bias_i;
    logic [A_W       -1:0]            A_i;
    logic [X_W       -1:0]            x_i;
    logic [D_W       -1:0]            D_i;
    logic [B_TILE_W  -1:0]            B_tile_i;
    logic [C_TILE_W  -1:0]            C_tile_i;
    logic [HPREV_W   -1:0]            hprev_tile_i;

    logic [Y_W       -1:0]            y_final_o;
    logic                             y_final_valid_o;

    // =========================================================
    // 3. AXIS MM2S → SSMBLOCK_TOP 입력으로 언팩
    //
    //    가정: axis_mm2s.tdata = {
    //              hprev_tile_i,
    //              C_tile_i,
    //              B_tile_i,
    //              D_i,
    //              x_i,
    //              A_i,
    //              dt_bias_i,
    //              dt_i
    //          }
    // =========================================================

    // 타일 핸드쉐이크: AXIS (valid/ready) <-> SSMBLOCK_TOP (tile_valid/tile_ready)
    assign tile_valid_i      = axis_mm2s.tvalid;
    assign axis_mm2s.tready  = tile_ready_o;  // backpressure를 SSMBLOCK_TOP에서 처리

    // tdata 언팩 (MSB부터 순서대로 슬라이스)
    // 인덱싱을 위해 임시 변수 사용
    logic [AXIS_IN_DATA_WIDTH-1:0] mm2s_tdata_q;

    assign mm2s_tdata_q = axis_mm2s.tdata;

    // 비트 위치 계산용 인덱스
    localparam int OFF_DT       = 0;
    localparam int OFF_DT_BIAS  = OFF_DT      + DT_W;
    localparam int OFF_A        = OFF_DT_BIAS + DT_BIAS_W;
    localparam int OFF_X        = OFF_A       + A_W;
    localparam int OFF_D        = OFF_X       + X_W;
    localparam int OFF_B_TILE   = OFF_D       + D_W;
    localparam int OFF_C_TILE   = OFF_B_TILE  + B_TILE_W;
    localparam int OFF_HPREV    = OFF_C_TILE  + C_TILE_W;
    // OFF_HPREV + HPREV_W == TOTAL_IN_W == AXIS_IN_DATA_WIDTH

    // 여기서는 LSB부터 증가시키는 방식으로 패킹했다고 가정
    // (필요하면 이 부분만 바꿔서 MSB-first로 맞춰도 됨)
    assign dt_i          = mm2s_tdata_q[OFF_DT      +: DT_W];
    assign dt_bias_i     = mm2s_tdata_q[OFF_DT_BIAS +: DT_BIAS_W];
    assign A_i           = mm2s_tdata_q[OFF_A       +: A_W];
    assign x_i           = mm2s_tdata_q[OFF_X       +: X_W];
    assign D_i           = mm2s_tdata_q[OFF_D       +: D_W];
    assign B_tile_i      = mm2s_tdata_q[OFF_B_TILE  +: B_TILE_W];
    assign C_tile_i      = mm2s_tdata_q[OFF_C_TILE  +: C_TILE_W];
    assign hprev_tile_i  = mm2s_tdata_q[OFF_HPREV   +: HPREV_W];

    // axis_mm2s.tlast는 필요하다면
    // - 타일 경계 / 시퀀스 끝 표시로 사용 가능
    // - 지금은 SSMBLOCK_TOP에 별도 포트가 없으므로 래퍼 내부 상태에서 활용하거나 무시해도 됨.

    // =========================================================
    // 4. SSMBLOCK_TOP 인스턴스
    // =========================================================
    SSMBLOCK_TOP #(
        .DW         (DW),
        .H_TILE     (H_TILE),
        .P_TILE     (P_TILE),
        .N_TILE     (N_TILE),
        .P_TOTAL    (P_TOTAL),
        .N_TOTAL    (N_TOTAL),
        .LAT_DX_M   (6),
        .LAT_DBX_M  (6),
        .LAT_DAH_M  (6),
        .LAT_ADD_A  (11),
        .LAT_ACCU   (11*7),
        .LAT_HC_M   (6),
        .LAT_MUL    (6),
        .LAT_ADD    (11),
        .LAT_DIV    (15),
        .LAT_EXP    (6 + 6*3 + 11*3 + 1+1+1),
        .LAT_SP     ( (6 + 6*3 + 11*3 + 1+1+1) + 6 + 11 + 15 + 1 )
    ) u_ssmblock_top (
        .clk             (clk),
        .rstn            (rstn),

        .tile_valid_i    (tile_valid_i),
        .tile_ready_o    (tile_ready_o),

        .dt_i            (dt_i),
        .dt_bias_i       (dt_bias_i),
        .A_i             (A_i),
        .x_i             (x_i),
        .D_i             (D_i),

        .B_tile_i        (B_tile_i),
        .C_tile_i        (C_tile_i),
        .hprev_tile_i    (hprev_tile_i),

        .y_final_o       (y_final_o),
        .y_final_valid_o (y_final_valid_o)
    );

    // =========================================================
    // 5. SSMBLOCK_TOP 출력 → AXIS S2MM 변환
    //
    //    - y_final_o 폭 = Y_W
    //    - 한 타일 결과를 1 beat로 내보낸다고 가정
    //    - y_final_valid_o = 1 되는 사이클에 결과를 캡쳐해서
    //      AXIS 쪽에서 tready가 올 때까지 유지하는 1-딥 버퍼 구현
    // =========================================================

    logic              out_valid_q;
    logic [Y_W-1:0]    out_data_q;

    // 간단한 1-딥 skid buffer
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            out_valid_q <= 1'b0;
            out_data_q  <= '0;
        end else begin
            // SSMBLOCK_TOP에서 새 결과가 나왔고 버퍼가 비어있으면 캡쳐
            if (y_final_valid_o && !out_valid_q) begin
                out_valid_q <= 1'b1;
                out_data_q  <= y_final_o;
            end else if (out_valid_q && axis_s2mm.tready) begin
                // DMA가 한 beat를 소비하면 비우기
                out_valid_q <= 1'b0;
            end
        end
    end

    assign axis_s2mm.tdata  = out_data_q;
    assign axis_s2mm.tvalid = out_valid_q;

    // 결과가 한 타일당 한 beat라고 가정하면 tlast=1로 내보내는 것이 자연스럽다.
    assign axis_s2mm.tlast  = out_valid_q;

endmodule
