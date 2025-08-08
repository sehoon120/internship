// packing_axi_top.v
// AXI4 Master 기반 타일 로딩/저장 스켈레톤 (Synth 가능 형태)
// - 필요 최소화: 단일 outstanding, 단일 버스트, 단순 FSM
// - 실제 프로젝트에 맞게 주소/버스트 길이/정렬/대기 삽입 등 보완 필요

module packing_axi_top #(
    parameter integer B       = 1,
    parameter integer H       = 24,
    parameter integer P       = 64,
    parameter integer N       = 128,
    parameter integer H_tile  = 12,
    parameter integer P_tile  = 16,
    parameter integer DW      = 16,   // fp16
    parameter integer AXI_ADDR_W = 64,
    parameter integer AXI_DATA_W = 128 // 128/64 권장 (FPGA DDR IF 폭 맞추기)
)(
    input  wire                     clk,
    input  wire                     rst,

    // control
    input  wire                     start,
    output reg                      done,

    // base addresses (byte address) - SW가 레지스터로 써주거나, 정적으로 연결
    input  wire [AXI_ADDR_W-1:0]    base_dt,
    input  wire [AXI_ADDR_W-1:0]    base_dA,
    input  wire [AXI_ADDR_W-1:0]    base_Bmat,
    input  wire [AXI_ADDR_W-1:0]    base_C,
    input  wire [AXI_ADDR_W-1:0]    base_D,
    input  wire [AXI_ADDR_W-1:0]    base_x,
    input  wire [AXI_ADDR_W-1:0]    base_hprev,
    input  wire [AXI_ADDR_W-1:0]    base_y,

    // AXI4 Master (write address)
    output reg  [AXI_ADDR_W-1:0]    M_AXI_AWADDR,
    output reg  [7:0]               M_AXI_AWLEN,   // 0..255 beats
    output reg  [2:0]               M_AXI_AWSIZE,  // log2(AXI_DATA_W/8)
    output reg  [1:0]               M_AXI_AWBURST, // INCR=2'b01
    output reg                      M_AXI_AWVALID,
    input  wire                     M_AXI_AWREADY,
    // write data
    output reg  [AXI_DATA_W-1:0]    M_AXI_WDATA,
    output reg  [AXI_DATA_W/8-1:0]  M_AXI_WSTRB,
    output reg                      M_AXI_WLAST,
    output reg                      M_AXI_WVALID,
    input  wire                     M_AXI_WREADY,
    // write response
    input  wire [1:0]               M_AXI_BRESP,
    input  wire                     M_AXI_BVALID,
    output reg                      M_AXI_BREADY,

    // AXI4 Master (read address)
    output reg  [AXI_ADDR_W-1:0]    M_AXI_ARADDR,
    output reg  [7:0]               M_AXI_ARLEN,
    output reg  [2:0]               M_AXI_ARSIZE,
    output reg  [1:0]               M_AXI_ARBURST,
    output reg                      M_AXI_ARVALID,
    input  wire                     M_AXI_ARREADY,
    // read data
    input  wire [AXI_DATA_W-1:0]    M_AXI_RDATA,
    input  wire [1:0]               M_AXI_RRESP,
    input  wire                     M_AXI_RLAST,
    input  wire                     M_AXI_RVALID,
    output reg                      M_AXI_RREADY
);

    // =========================
    // 유틸 상수
    // =========================
    localparam integer WORDS_PER_BEAT = AXI_DATA_W / DW; // AXI 한 비트당 몇 개의 fp16?
    localparam [1:0] BURST_INCR = 2'b01;

    // Vivado용 SIZE=log2(bytes/beat)
    function [2:0] f_axi_size(input integer data_w);
        integer bytes;
        begin
            bytes = data_w/8;
            case (bytes)
                1: f_axi_size = 3'd0;
                2: f_axi_size = 3'd1;
                4: f_axi_size = 3'd2;
                8: f_axi_size = 3'd3;
                16: f_axi_size = 3'd4;
                32: f_axi_size = 3'd5;
                64: f_axi_size = 3'd6;
                128: f_axi_size = 3'd7;
                default: f_axi_size = 3'd4; // fallback
            endcase
        end
    endfunction

    // =========================
    // 타일 인덱서
    // =========================
    localparam integer NUM_TILE_H = H / H_tile;
    localparam integer NUM_TILE_P = P / P_tile;

    reg [$clog2(NUM_TILE_H)-1:0] h_idx;
    reg [$clog2(NUM_TILE_P)-1:0] p_idx;

    // =========================
    // 온칩 타일 버퍼 (BRAM/URAM로 유도)
    // =========================
    // *Synth 되도록 단순 1D 메모리로 선언 (generate로 2D도 가능)
    // 데이터량 추정:
    //  - x_tile: H_tile*P_tile * 16b        (~192*16=3072b)
    //  - hprev_tile: H_tile*P_tile*N * 16b (~192*128*16 = 393,216b = 48KB)  -> BRAM ok
    //
    // 필요시 (* ram_style = "ultra" *) / (* ram_style = "block" *) 붙여서 URAM/BRAM 강제
    //
    reg [DW-1:0] dt_tile   [0:B*H_tile-1];
    reg [DW-1:0] dA_tile   [0:B*H_tile-1];
    reg [DW-1:0] D_tile    [0:H_tile-1];
    reg [DW-1:0] x_tile    [0:B*H_tile*P_tile-1];
    reg [DW-1:0] hprev_tile[0:B*H_tile*P_tile*N-1];
    reg [DW-1:0] y_tile    [0:B*H_tile*P_tile-1]; // 결과 버퍼

    // ssm_block 연결용 플랫 와이어
    wire [B*H_tile*DW-1:0]      w_dt_flat;
    wire [B*H_tile*DW-1:0]      w_dA_flat;
    wire [H_tile*DW-1:0]        w_D_flat;
    wire [B*N*DW-1:0]           w_Bmat_flat;
    wire [B*N*DW-1:0]           w_C_flat;
    wire [B*H_tile*P_tile*DW-1:0] w_x_flat;
    wire [B*H_tile*P_tile*N*DW-1:0] w_hprev_flat;
    wire [B*H_tile*P_tile*DW-1:0]   w_y_flat;

    // 간단한 pack (LSB에 낮은 인덱스)
    genvar gi;
    generate
        for (gi=0; gi < B*H_tile; gi=gi+1) begin : G_PACK_DT
            assign w_dt_flat[DW*gi +: DW] = dt_tile[gi];
            assign w_dA_flat[DW*gi +: DW] = dA_tile[gi];
        end
        for (gi=0; gi < H_tile; gi=gi+1) begin : G_PACK_D
            assign w_D_flat[DW*gi +: DW] = D_tile[gi];
        end
        for (gi=0; gi < B*H_tile*P_tile; gi=gi+1) begin : G_PACK_XY
            assign w_x_flat[DW*gi +: DW] = x_tile[gi];
            // 결과 언팩은 아래 WRITE 단계에서 y_tile[]로 받아서 버스트 씀
        end
        for (gi=0; gi < B*H_tile*P_tile*N; gi=gi+1) begin : G_PACK_HPREV
            assign w_hprev_flat[DW*gi +: DW] = hprev_tile[gi];
        end
    endgenerate

    // Bmat/C는 타일 독립 → 한 번만 읽어 일정 레지스터/BRAM에 상주시켜도 OK
    // 여기서는 간단히 "별도 세션에서 읽어놓았다"고 가정하고 내부 버퍼로 둠
    reg [DW-1:0] Bmat_buf[0:B*N-1];
    reg [DW-1:0] C_buf   [0:B*N-1];
    generate
        for (gi=0; gi < B*N; gi=gi+1) begin : G_PACK_BC
            assign w_Bmat_flat[DW*gi +: DW] = Bmat_buf[gi];
            assign w_C_flat   [DW*gi +: DW] = C_buf[gi];
        end
    endgenerate

    // =========================
    // SSM 타일 엔진 인스턴스
    // =========================
    reg start_tile;
    wire done_tile;

    ssm_block_fp16_top #(
        .B(B), .H(H_tile), .P(P_tile), .N(N), .DW(DW)
    ) u_tile (
        .clk(clk), .rst(rst), .start(start_tile),
        .dt_flat(w_dt_flat),
        .dA_flat(w_dA_flat),
        .Bmat_flat(w_Bmat_flat),
        .C_flat(w_C_flat),
        .D_flat(w_D_flat),
        .x_flat(w_x_flat),
        .h_prev_flat(w_hprev_flat),
        .y_flat(w_y_flat),
        .done(done_tile)
    );

    // y_flat → y_tile 배열로 언팩 (쓰기 편하게)
    integer uy;
    always @(*) begin
        for (uy=0; uy < B*H_tile*P_tile; uy=uy+1) begin
            y_tile[uy] = w_y_flat[DW*uy +: DW];
        end
    end

    // =========================
    // AXI 간단 FSM
    // =========================
    typedef enum reg [3:0] {
        S_IDLE=0,
        S_PREP_STATIC,     // Bmat/C/D 1회 로드(옵션)
        S_READ_DT_DA,      // dt/dA 타일 읽기
        S_READ_X,          // x 타일 읽기
        S_READ_HPREV,      // hprev 타일 읽기
        S_RUN_TILE,        // ssm_block 실행
        S_WRITE_Y,         // y 타일 쓰기
        S_NEXT_TILE,       // 인덱스 증가
        S_DONE
    } state_t;
    state_t state, nstate;

    // AR/W ch 공통 유틸
    localparam [2:0] AXI_SIZE = f_axi_size(AXI_DATA_W);

    // 간단 카운터
    reg [15:0] beat_cnt; // 한 번의 버스트 내 beat 카운트
    reg [31:0] word_cnt; // 해당 섹션에서 수집/배출한 fp16 words

    // 주소 생성용 베이스 + 오프셋 (너 프로젝트 레이아웃대로 계산식 바꾸면 됨)
    reg [AXI_ADDR_W-1:0] cur_addr;
    reg [31:0]           cur_words;  // 이번에 옮길 총 word 수 (fp16 단위)
    reg [31:0]           words_target;

    // 간단한 read/write 파이프 제어
    wire rd_done = (M_AXI_RVALID && M_AXI_RLAST && M_AXI_RREADY);
    wire wr_done = (M_AXI_WVALID && M_AXI_WREADY && M_AXI_WLAST);

    // RREADY/WVALID 기본 정책
    always @(*) begin
        M_AXI_ARBURST = BURST_INCR;
        M_AXI_AWBURST = BURST_INCR;
        M_AXI_ARSIZE  = AXI_SIZE;
        M_AXI_AWSIZE  = AXI_SIZE;
        M_AXI_WSTRB   = {AXI_DATA_W/8{1'b1}};
    end

    // FSM 전이
    always @(posedge clk or posedge rst) begin
        if (rst) state <= S_IDLE;
        else     state <= nstate;
    end

    // FSM 논리
    always @(*) begin
        nstate = state;
        case (state)
            S_IDLE:        nstate = start ? S_PREP_STATIC : S_IDLE;
            S_PREP_STATIC: nstate = /* Bmat/C/D 읽기 끝났으면 */ S_READ_DT_DA;
            S_READ_DT_DA:  nstate = /* dt/dA 읽기 완료 */ S_READ_X;
            S_READ_X:      nstate = /* x 읽기 완료 */ S_READ_HPREV;
            S_READ_HPREV:  nstate = /* hprev 읽기 완료 */ S_RUN_TILE;
            S_RUN_TILE:    nstate = done_tile ? S_WRITE_Y : S_RUN_TILE;
            S_WRITE_Y:     nstate = /* y 쓰기 완료 */ S_NEXT_TILE;
            S_NEXT_TILE:   nstate = ( (h_idx==NUM_TILE_H-1) && (p_idx==NUM_TILE_P-1) )
                                   ? S_DONE : S_READ_DT_DA;
            S_DONE:        nstate = S_IDLE;
            default:       nstate = S_IDLE;
        endcase
    end

    // 인덱스/완료 플래그
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done    <= 1'b0;
            h_idx   <= 'd0;
            p_idx   <= 'd0;
            start_tile <= 1'b0;
        end else begin
            done    <= 1'b0;
            start_tile <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        h_idx <= 0; p_idx <= 0;
                    end
                end
                S_RUN_TILE: begin
                    if (state!=nstate && nstate==S_RUN_TILE) begin
                        // stay
                    end
                end
                S_NEXT_TILE: begin
                    if (p_idx == NUM_TILE_P-1) begin
                        p_idx <= 0;
                        h_idx <= h_idx + 1;
                    end else begin
                        p_idx <= p_idx + 1;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                end
            endcase
            // 타일 실행 트리거
            if (state==S_RUN_TILE && nstate==S_RUN_TILE && start_tile==1'b0) begin
                start_tile <= 1'b1;
            end
        end
    end

    // =========================
    // 아주 단순한 Read/Write 핸드셰이크 예시
    // (실전에서는 burst 길이/정렬/경계 넘김 방지 로직 추가 필수)
    // =========================

    // 기본값
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            M_AXI_ARVALID <= 1'b0; M_AXI_RREADY <= 1'b0;
            M_AXI_AWVALID <= 1'b0; M_AXI_WVALID <= 1'b0;
            M_AXI_WLAST   <= 1'b0; M_AXI_BREADY <= 1'b0;
            M_AXI_ARLEN   <= 8'd0; M_AXI_AWLEN   <= 8'd0;
            beat_cnt      <= 16'd0; word_cnt     <= 32'd0;
            cur_addr      <= '0;    cur_words    <= 32'd0;
        end else begin
            // 기본 비활성
            if (state!=S_READ_DT_DA && state!=S_READ_X && state!=S_READ_HPREV) begin
                M_AXI_ARVALID <= 1'b0;
                M_AXI_RREADY  <= 1'b0;
            end
            if (state!=S_WRITE_Y) begin
                M_AXI_AWVALID <= 1'b0;
                M_AXI_WVALID  <= 1'b0;
                M_AXI_BREADY  <= 1'b0;
                M_AXI_WLAST   <= 1'b0;
            end

            case (state)
                // ====== 예: dt/dA 읽기 세션 ======
                S_READ_DT_DA: begin
                    // 예시: dt(H_tile words) + dA(H_tile words) → 두 번에 나눠 읽거나, 하나씩 처리
                    // 여기선 dt만 보여줌 (dA도 동일 패턴으로 반복)
                    if (!M_AXI_ARVALID && !M_AXI_RREADY) begin
                        cur_addr   <= base_dt + (/*b=0*/0)*H*2 + (h_idx*H_tile)*2; // FP16=2 bytes
                        cur_words  <= B*H_tile; // dt 타일 words
                        words_target <= B*H_tile;
                        M_AXI_ARADDR  <= cur_addr;
                        M_AXI_ARLEN   <= ( ( (B*H_tile + WORDS_PER_BEAT-1)/WORDS_PER_BEAT ) - 1);
                        M_AXI_ARVALID <= 1'b1;
                        beat_cnt      <= 0;
                        word_cnt      <= 0;
                    end else if (M_AXI_ARVALID && M_AXI_ARREADY) begin
                        M_AXI_ARVALID <= 1'b0;
                        M_AXI_RREADY  <= 1'b1;
                    end else if (M_AXI_RVALID && M_AXI_RREADY) begin
                        // 수신 비트폭 → WORDS_PER_BEAT 만큼 unpack 후 dt_tile[]에 저장
                        integer w;
                        for (w=0; w<WORDS_PER_BEAT; w=w+1) begin
                            if (word_cnt < words_target)
                                dt_tile[word_cnt] <= M_AXI_RDATA[DW*w +: DW];
                            word_cnt <= word_cnt + 1;
                        end
                        if (M_AXI_RLAST) begin
                            M_AXI_RREADY <= 1'b0;
                            // TODO: dA 읽기 반복 (또 다른 AR 시퀀스)
                            // 여기서는 단순화: dt/dA 모두 읽었다고 가정하고 다음 상태로
                        end
                    end
                end

                // ====== x 타일 읽기 ======
                S_READ_X: begin
                    if (!M_AXI_ARVALID && !M_AXI_RREADY) begin
                        // x[b][h_abs][p_abs] 타일 words = B*H_tile*P_tile
                        cur_addr   <= base_x + ((h_idx*P) + (p_idx*P_tile))*2; // 예시 주소식
                        words_target <= B*H_tile*P_tile;
                        M_AXI_ARADDR  <= cur_addr;
                        M_AXI_ARLEN   <= (((words_target + WORDS_PER_BEAT-1)/WORDS_PER_BEAT) - 1);
                        M_AXI_ARVALID <= 1'b1;
                        beat_cnt      <= 0;
                        word_cnt      <= 0;
                    end else if (M_AXI_ARVALID && M_AXI_ARREADY) begin
                        M_AXI_ARVALID <= 1'b0;
                        M_AXI_RREADY  <= 1'b1;
                    end else if (M_AXI_RVALID && M_AXI_RREADY) begin
                        integer w;
                        for (w=0; w<WORDS_PER_BEAT; w=w+1) begin
                            if (word_cnt < words_target)
                                x_tile[word_cnt] <= M_AXI_RDATA[DW*w +: DW];
                            word_cnt <= word_cnt + 1;
                        end
                        if (M_AXI_RLAST) begin
                            M_AXI_RREADY <= 1'b0;
                        end
                    end
                end

                // ====== hprev 타일 읽기 ======
                S_READ_HPREV: begin
                    if (!M_AXI_ARVALID && !M_AXI_RREADY) begin
                        // words = B*H_tile*P_tile*N
                        cur_addr   <= base_hprev + (( (h_idx*P) + (p_idx*P_tile) ) * N)*2;
                        words_target <= B*H_tile*P_tile*N;
                        M_AXI_ARADDR  <= cur_addr;
                        M_AXI_ARLEN   <= (((words_target + WORDS_PER_BEAT-1)/WORDS_PER_BEAT) - 1);
                        M_AXI_ARVALID <= 1'b1;
                        beat_cnt      <= 0;
                        word_cnt      <= 0;
                    end else if (M_AXI_ARVALID && M_AXI_ARREADY) begin
                        M_AXI_ARVALID <= 1'b0;
                        M_AXI_RREADY  <= 1'b1;
                    end else if (M_AXI_RVALID && M_AXI_RREADY) begin
                        integer w;
                        for (w=0; w<WORDS_PER_BEAT; w=w+1) begin
                            if (word_cnt < words_target)
                                hprev_tile[word_cnt] <= M_AXI_RDATA[DW*w +: DW];
                            word_cnt <= word_cnt + 1;
                        end
                        if (M_AXI_RLAST) begin
                            M_AXI_RREADY <= 1'b0;
                        end
                    end
                end

                // ====== y 타일 쓰기 ======
                S_WRITE_Y: begin
                    if (!M_AXI_AWVALID && !M_AXI_WVALID) begin
                        cur_addr <= base_y + ((h_idx*P) + (p_idx*P_tile))*2;
                        words_target <= B*H_tile*P_tile;
                        M_AXI_AWADDR  <= cur_addr;
                        M_AXI_AWLEN   <= (((words_target + WORDS_PER_BEAT-1)/WORDS_PER_BEAT) - 1);
                        M_AXI_AWVALID <= 1'b1;
                        beat_cnt      <= 0;
                        word_cnt      <= 0;
                    end else if (M_AXI_AWVALID && M_AXI_AWREADY) begin
                        M_AXI_AWVALID <= 1'b0;
                        M_AXI_WVALID  <= 1'b1;
                    end else if (M_AXI_WVALID && M_AXI_WREADY) begin
                        // y_tile -> WDATA 패킹
                        integer w;
                        reg [AXI_DATA_W-1:0] wtmp;
                        wtmp = '0;
                        for (w=0; w<WORDS_PER_BEAT; w=w+1) begin
                            if (word_cnt < words_target)
                                wtmp[DW*w +: DW] = y_tile[word_cnt];
                            word_cnt <= word_cnt + 1;
                        end
                        M_AXI_WDATA <= wtmp;
                        M_AXI_WLAST <= (word_cnt >= words_target);
                        if (M_AXI_WLAST) begin
                            M_AXI_WVALID <= 1'b0;
                            M_AXI_BREADY <= 1'b1;
                        end
                    end else if (M_AXI_BVALID && M_AXI_BREADY) begin
                        M_AXI_BREADY <= 1'b0;
                    end
                end

                default: ; // 나머지 상태는 생략
            endcase
        end
    end

endmodule
