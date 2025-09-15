// ================================================================
// SSMBLOCK_TOP — tile-in → (dBx, dAh) → h_next → hC → sum128 → +xD → y_final
//  * softplus/exp는 단일 코어(softplus_or_exp16) 공유(time-multiplex)
//  * dA_exp가 준비될 때까지 dAh 경로만 타일 스트림을 지연시켜 dBx와 정렬
// ================================================================
module SSMBLOCK_TOP #(
    parameter integer DW        = 16,
    parameter integer H_TILE    = 1,
    parameter integer P_TILE    = 1,
    parameter integer N_TILE    = 128,
    parameter integer N_TOTAL   = 128,
    // Latency params (IP 설정에 맞춰 조정)
    parameter integer LAT_MUL  = 2,
    parameter integer LAT_ADD  = 2,
    parameter integer LAT_DIV  = 2,
    parameter integer LAT_DX_M  = 6,    // scalar mul (dx 등)
    parameter integer LAT_DBX_M = 6,    // dBx lane mul
    parameter integer LAT_DAH_M = 6,    // dAh lane mul
    parameter integer LAT_ADD_A = 11,   // h_next add
    parameter integer LAT_HC_M  = 6,    // hC lane mul
    parameter integer LAT_SOFT  = LAT_ADD + LAT_DIV + 1 + LAT_MUL + LAT_EXP + POST_SOFT_LAT,   // ★ softplus 파이프(softplus_or_exp16에서 실효 지연)
    parameter integer LAT_EXP   = 6 + LAT_MUL * 3 + LAT_ADD * 3    // ★ exp 파이프(softplus_or_exp16에서 실효 지연)
)(
    input  wire                   clk,
    input  wire                   rstn,

    // 타일 유효(연속 타일 스트리밍)
    input  wire                   tile_valid_i,
    output wire                   tile_ready_o,   // 기본 1 (필요시 backpressure)

    // Scalars (토큰 당 고정)
    input  wire [DW-1:0]          dt_i,       // before Softplus
    input  wire [DW-1:0]          dt_bias_i,
    input  wire [DW-1:0]          A_i,
    input  wire [DW-1:0]          x_i,
    input  wire [DW-1:0]          D_i,

    // Tile vectors (N_TILE lanes)
    input  wire [N_TILE*DW-1:0]   B_tile_i,
    input  wire [N_TILE*DW-1:0]   C_tile_i,
    input  wire [N_TILE*DW-1:0]   hprev_tile_i,

    input  wire                   mode_sp,    // 외부 모드(필요 시 사용), 내부에선 자동 시퀀싱

    // 최종 출력: y = sum_{n} hC[n] + x*D  (N_TOTAL 처리 끝날 때 1펄스)
    output wire [DW-1:0]          y_final_o,
    output wire                   y_final_valid_o
);
    // ------------------------------------------------------------
    localparam integer TILES_PER_GROUP = (N_TOTAL / N_TILE);
    localparam integer B_W = N_TILE*DW;
    localparam integer C_W = N_TILE*DW;

    // ============================================================
    // (1) delta = dt + dt_bias  (scalar)
    // ============================================================
    wire [DW-1:0] delta_o;
    wire          v_delta;

    // NOTE: delta 모듈 내부 add latency는 여기선 LAT_DX_M로 전달 (필요시 별도 파라미터로 분리 권장)
    delta #(.DW(DW), .MUL_LAT(LAT_DX_M)) u_delta (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i   (tile_valid_i),   // 토큰 시작과 함께 1펄스 이상
        .dt_i      (dt_i),
        .dt_bias_i (dt_bias_i),
        .delta_o   (delta_o),
        .valid_o   (v_delta)
    );

    // ============================================================
    // (2) softplus_or_exp16 (단일 코어 공유; 시퀀서로 2회 사용)
    //     - 1회차: softplus(delta)
    //     - 2회차: exp(dA)   (dA는 아래 mul_scalar에서 생성)
    // ============================================================
    // ★ u_soft 입력 멀티플렉서(간단한 pending 포함)
    wire [DW-1:0] dA_w;
    wire          v_dA;

    // soft 코어 입력 제어
    reg               pend_dA;
    reg  [DW-1:0]     pend_dA_x;

    wire want_delta_now = v_delta;     // softplus 요청
    wire want_dA_now    = v_dA;        // exp 요청

    // 발행 우선순위: (1) pending dA → (2) delta → (3) fresh dA
    reg        soft_valid_i;
    reg        soft_mode_i;            // 1: softplus, 0: exp
    reg [DW-1:0] soft_x_i;

    always @(*) begin
        soft_valid_i = 1'b0;
        soft_mode_i  = 1'b0;
        soft_x_i     = {DW{1'b0}};
        if (pend_dA) begin
            soft_valid_i = 1'b1;
            soft_mode_i  = 1'b0;          // exp
            soft_x_i     = pend_dA_x;
        end else if (want_delta_now) begin
            soft_valid_i = 1'b1;
            soft_mode_i  = 1'b1;          // softplus
            soft_x_i     = delta_o;
        end else if (want_dA_now) begin
            soft_valid_i = 1'b1;
            soft_mode_i  = 1'b0;          // exp
            soft_x_i     = dA_w;
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            pend_dA   <= 1'b0;
            pend_dA_x <= {DW{1'b0}};
        end else begin
            // 새 dA 요청이 있는데 이번 사이클에 발행되지 못하면 보류
            if (want_dA_now && !(soft_valid_i && !soft_mode_i && (soft_x_i==dA_w))) begin
                pend_dA   <= 1'b1;
                pend_dA_x <= dA_w;
            end else if (soft_valid_i && !soft_mode_i && pend_dA && (soft_x_i==pend_dA_x)) begin
                // 방금 pend_dA를 소모
                pend_dA <= 1'b0;
            end
        end
    end

    // 단일 soft/exp 코어
    wire [DW-1:0] y_soft, y_exp;
    wire          v_y_soft, v_y_exp;
    wire          mode_softplus_aligned;

    softplus_or_exp16 #(
        .DW(DW),
        .LAT_MUL(LAT_MUL),
        .LAT_ADD(LAT_ADD),
        .LAT_DIV(LAT_DIV),
        .LAT_EXP(LAT_EXP)
    ) u_soft (
        .clk             (clk),
        .rstn            (rstn),
        .valid_i         (soft_valid_i),
        .mode_softplus_i (soft_mode_i),
        .x_i             (soft_x_i),

        .y_o_S           (y_soft),
        .valid_o_S       (v_y_soft),
        .y_o_e           (y_exp),
        .valid_o_e       (v_y_exp),
        .mode_softplus_o (mode_softplus_aligned)
    );

    // ============================================================
    // (3) dA = delta_sp * A   (scalar)  — y_soft가 유효할 때 곱
    // ============================================================
    mul_scalar #(.DW(DW), .LAT_MUL(LAT_DX_M)) u_mul_dA (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_y_soft),   // softplus 결과가 준비된 순간
        .a_i     (y_soft),
        .b_i     (A_i),
        .y_o     (dA_w),
        .valid_o (v_dA)
    );

    // ============================================================
    // (4) dA_exp = exp(dA)   (scalar)
    //      - u_soft에서 mode=0 결과(y_exp)로 출력
    //      - 토큰 동안 유지(브로드캐스트)
    // ============================================================
    reg [DW-1:0] dA_exp_hold;
    reg          dA_exp_ready;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            dA_exp_hold  <= {DW{1'b0}};
            dA_exp_ready <= 1'b0;
        end else begin
            if (v_y_exp) begin
                dA_exp_hold  <= y_exp;
                dA_exp_ready <= 1'b1;  // 다음 토큰에서 적절히 클리어(필요 시)
            end
            // 토큰 경계에서 클리어하고 싶다면 y_final_valid_o 등을 이용해 dA_exp_ready <= 0; 처리
            // 여기선 스택 프로토콜상 순차 토큰을 가정하여 hold 갱신만 수행
        end
    end

    // ============================================================
    // (3') dx = delta_sp * x   (scalar) — 원 코드 유지
    // ============================================================
    wire [DW-1:0] dx_w;
    wire          v_dx;

    dx #(.DW(DW), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i),
        .dt_i    (dt_i),
        .x_i     (x_i),
        .dx_o    (dx_w),
        .valid_o (v_dx)
    );

    // ============================================================
    // (4') dBx = dx * B[n] (N_TILE 병렬)
    // ============================================================
    wire [N_TILE*DW-1:0] dBx_w_raw, dBx_w;
    wire                 v_dBx_raw,  v_dBx;

    // B 타일 정렬( dx latency 만큼 )
    integer bi;
    reg [B_W-1:0] B_tile_buffer [0:LAT_DX_M];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (bi = 0; bi <= LAT_DX_M; bi = bi + 1)
                B_tile_buffer[bi] <= {B_W{1'b0}};
        end else begin
            B_tile_buffer[0] <= B_tile_i;
            for (bi = 1; bi <= LAT_DX_M; bi = bi + 1)
                B_tile_buffer[bi] <= B_tile_buffer[bi-1];
        end
    end
    wire [B_W-1:0] B_tile_aligned = B_tile_buffer[LAT_DX_M];

    dBx #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx),
        .dx_i    (dx_w),
        .Bmat_i  (B_tile_aligned),
        .dBx_o   (dBx_w_raw),
        .valid_o (v_dBx_raw)
    );

    // ============================================================
    // (5) dAh = dA_exp * hprev[n] (N_TILE 병렬)
    //     ★ dA_exp가 준비될 때까지 hprev 타일 스트림을 지연
    // ============================================================
    // dA_exp 준비까지 필요한 지연(타일 valid 기준)
    //   delta(valid) 지연 LAT_DX_M + softplus LAT_SOFT + mul(A) LAT_DX_M + exp LAT_EXP
    localparam integer DLY_TILE_TO_DAH = (LAT_DX_M + LAT_SOFT + LAT_DX_M + LAT_MUL + LAT_EXP);
    // hprev 및 해당 valid 를 동일하게 지연
    wire [N_TILE*DW-1:0] hprev_aligned;
    wire                 v_tile_for_dAh;

    pipe_bus #(.W(N_TILE*DW), .D(DLY_TILE_TO_DAH)) u_dly_hprev (
        .clk(clk), .rstn(rstn),
        .din(hprev_tile_i), .vin(tile_valid_i),
        .dout(hprev_aligned), .vout(v_tile_for_dAh)
    );

    wire [N_TILE*DW-1:0] dAh_raw_w, dAh_w;
    wire                 v_dAh_raw,  v_dAh;

    dAh #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DAH_M)) u_dAh (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_tile_for_dAh),      // ★ 지연된 타일 valid
        .dA_i    (dA_exp_hold),         // ★ 브로드캐스트 스칼라
        .hprev_i (hprev_aligned),
        .dAh_o   (dAh_raw_w),
        .valid_o (v_dAh_raw)
    );

    // ============================================================
    // (5') dBx를 dAh에 정렬시키기 위한 추가 지연
    //   dAh 결과 시점:  DLY_TILE_TO_DAH + LAT_DAH_M
    //   dBx 결과 시점:  LAT_DX_M + LAT_DBX_M
    //   → dBx 추가 지연 = (DLY_TILE_TO_DAH + LAT_DAH_M) - (LAT_DX_M + LAT_DBX_M)
    // ============================================================
    localparam integer DLY_DBX_TO_ALIGN = ( (LAT_DX_M + LAT_SOFT + LAT_DX_M + LAT_EXP + LAT_DAH_M)
                                          - (LAT_DX_M + LAT_DBX_M) );
    generate
        if (DLY_DBX_TO_ALIGN <= 0) begin
            assign dBx_w = dBx_w_raw;
            assign v_dBx = v_dBx_raw;
        end else begin
            pipe_bus_bram #(.W(N_TILE*DW), .D(DLY_DBX_TO_ALIGN)) u_dly_dBx_bus (
                .clk(clk), .rstn(rstn),
                .din(dBx_w_raw), .vin(v_dBx_raw),
                .dout(dBx_w), .vout(v_dBx)
            );
        end
    endgenerate

    // dAh 경로는 추가 지연 없이 그대로 사용
    assign dAh_w = dAh_raw_w;
    assign v_dAh = v_dAh_raw;

    // ============================================================
    // (6) h_next = dBx + dAh (lane-wise) — 정렬된 valid 동시 인가
    // ============================================================
    wire [N_TILE*DW-1:0] hnext_w;
    wire                 v_hnext;

    h_next #(.DW(DW), .N_TILE(N_TILE), .ADD_LAT(LAT_ADD_A)) u_hnext (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dBx & v_dAh),
        .dBx_i   (dBx_w),
        .dAh_i   (dAh_w),
        .hnext_o (hnext_w),
        .valid_o (v_hnext)
    );

    // ============================================================
    // (7) hC = h_next * C[n] (lane-wise)
    //     ★ dBx에 추가 지연이 들어갔으니 C 정렬도 그만큼 더해준다
    // ============================================================
    wire [N_TILE*DW-1:0] hC_tile_o;
    wire                 v_hC;
    integer ci;
    reg [C_W-1:0] C_tile_buffer [0:LAT_DX_M+LAT_DBX_M+LAT_ADD_A
                                   + ((DLY_DBX_TO_ALIGN>0)?DLY_DBX_TO_ALIGN:0)];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (ci = 0; ci <= LAT_DX_M+LAT_DBX_M+LAT_ADD_A+((DLY_DBX_TO_ALIGN>0)?DLY_DBX_TO_ALIGN:0); ci = ci + 1)
                C_tile_buffer[ci] <= {C_W{1'b0}};
        end else begin
            C_tile_buffer[0] <= C_tile_i;
            for (ci = 1; ci <= LAT_DX_M+LAT_DBX_M+LAT_ADD_A+((DLY_DBX_TO_ALIGN>0)?DLY_DBX_TO_ALIGN:0); ci = ci + 1)
                C_tile_buffer[ci] <= C_tile_buffer[ci-1];
        end
    end
    wire [C_W-1:0] C_tile_aligned = C_tile_buffer[LAT_DX_M+LAT_DBX_M+LAT_ADD_A+((DLY_DBX_TO_ALIGN>0)?DLY_DBX_TO_ALIGN:0)];

    hC #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_HC_M)) u_hC (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_hnext),
        .hnext_i (hnext_w),
        .C_i     (C_tile_aligned),
        .hC_o    (hC_tile_o),
        .valid_o (v_hC)
    );

    // ============================================================
    // (8) hC TILES collect & (1) xD = x*D  (원 코드 유지, 약간 정리)
    // ============================================================
    reg  [N_TILE*DW-1:0] hC_buf [0:TILES_PER_GROUP-1];
    reg  [$clog2(TILES_PER_GROUP)-1:0] tile_ptr;
    reg                  grp_emit;
    wire                 accept_tile = v_hC;

    // xD 한 번 계산 (첫 타일에서 트리거)
    wire [DW-1:0] xD_w;
    wire          v_xD_w;

    xD u_mul_xD (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (accept_tile && (tile_ptr==0)),
        .x_i     (x_i),
        .D_i     (D_i),
        .xD_o    (xD_w),
        .valid_o (v_xD_w)
    );

    integer ti;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            tile_ptr <= '0;
            grp_emit <= 1'b0;
            for (ti=0; ti<TILES_PER_GROUP; ti=ti+1) hC_buf[ti] <= {N_TILE*DW{1'b0}};
        end else begin
            grp_emit <= 1'b0;
            if (accept_tile) begin
                hC_buf[tile_ptr] <= hC_tile_o;
                if (tile_ptr == TILES_PER_GROUP-1) begin
                    tile_ptr <= '0;
                    grp_emit <= 1'b1;
                end else begin
                    tile_ptr <= tile_ptr + 1'b1;
                end
            end
        end
    end

    // 128-lane 평탄화 (일반화)
    wire [N_TOTAL*DW-1:0] hC_128_bus;
    generate
        if (TILES_PER_GROUP == 1) begin
            assign hC_128_bus = hC_buf[0];
        end else begin : G_FLATTEN
            genvar gi;
            for (gi=0; gi<TILES_PER_GROUP; gi=gi+1) begin : G_CAT
                // 아래는 간단화를 위해 순차 연결. 필요시 정확한 순서로 재정렬.
            end
            assign hC_128_bus = {
                hC_buf[TILES_PER_GROUP-1], hC_buf[TILES_PER_GROUP-2],
                hC_buf[TILES_PER_GROUP-3], hC_buf[TILES_PER_GROUP-4],
                hC_buf[3], hC_buf[2], hC_buf[1], hC_buf[0]
            };
        end
    endgenerate

    // ============================================================
    // (9) 128합 트리 → y_tmp
    // ============================================================
    wire [DW-1:0] y_tmp_w;
    wire          y_tmp_v;

    fp16_adder_tree_128 u_sum128 (
        .clk       (clk),
        .rst       (rstn),
        .valid_in  (grp_emit),
        .in_flat   (hC_128_bus),
        .sum       (y_tmp_w),
        .valid_out (y_tmp_v)
    );

    // ============================================================
    // (10) y_final = y_tmp + xD (xD도 경로에 맞춰 별도 정렬 필요)
    //      * 기존 SHIFT_xD는 원 코드 유지 (필요시 재조정)
    // ============================================================
    integer xDi;
    localparam integer SHIFT_xD = LAT_DBX_M+LAT_ADD_A+LAT_HC_M+77; // TODO: 보정
    reg [DW-1:0] xD_tile_buffer [0:SHIFT_xD];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (xDi = 0; xDi <= SHIFT_xD; xDi = xDi + 1) xD_tile_buffer[xDi] <= {DW{1'b0}};
        end else begin
            xD_tile_buffer[0] <= xD_w;
            for (xDi = 1; xDi <= SHIFT_xD; xDi = xDi + 1)
                xD_tile_buffer[xDi] <= xD_tile_buffer[xDi-1];
        end
    end
    wire [DW-1:0] xD_tile_aligned = xD_tile_buffer[SHIFT_xD];

    wire [DW-1:0] y_final_w;
    wire          v_y_final_w;

    y_out u_add_yfinal (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (y_tmp_v),
        .ytmp_i  (y_tmp_w),
        .xD_i    (xD_tile_aligned),
        .y_o     (y_final_w),
        .valid_o (v_y_final_w)
    );

    assign y_final_o       = y_final_w;
    assign y_final_valid_o = v_y_final_w;

    // 타일 입력 항상 수락(상위가 backpressure 원하면 여기서 제어)
    assign tile_ready_o = 1'b1;

endmodule

// ------------------------------------------------------------
// Data+valid pipeline (레지스터 기반)
// ------------------------------------------------------------
module pipe_bus #(
    parameter integer W = 16,
    parameter integer D = 0
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire [W-1:0] din,
    input  wire         vin,
    output wire [W-1:0] dout,
    output wire         vout
);
    generate
        if (D == 0) begin : G_D0
            assign dout = din;
            assign vout = vin;
        end else begin : G_DN
            reg [W-1:0] q  [0:D-1];
            reg         qv [0:D-1];
            integer i;
            always @(posedge clk or negedge rstn) begin
                if (!rstn) begin
                    for (i=0;i<D;i=i+1) begin
                        q[i]  <= {W{1'b0}};
                        qv[i] <= 1'b0;
                    end
                end else begin
                    q [0] <= din;  qv[0] <= vin;
                    for (i=1;i<D;i=i+1) begin
                        q [i] <= q [i-1];
                        qv[i] <= qv[i-1];
                    end
                end
            end
            assign dout = q [D-1];
            assign vout = qv[D-1];
        end
    endgenerate
endmodule

// ------------------------------------------------------------ 
// Data+valid pipeline with BRAM (fixed D-cycle delay)
// ------------------------------------------------------------
module pipe_bus_bram #(
    parameter integer W = 256,
    parameter integer D = 6
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire [W-1:0] din,
    input  wire         vin,
    output wire [W-1:0] dout,
    output wire         vout
);
    generate if (D == 0) begin
        assign dout = din;
        assign vout = vin;
    end else begin
        localparam integer ADDR_W = $clog2(D);
        reg [ADDR_W-1:0] wr_addr;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) wr_addr <= {ADDR_W{1'b0}};
            else if (wr_addr == D-1) wr_addr <= {ADDR_W{1'b0}};
            else wr_addr <= wr_addr + 1'b1;
        end
        wire [ADDR_W-1:0] rd_addr_minus = wr_addr - (D-1);
        wire               underflow    = (wr_addr < (D-1));
        wire [ADDR_W-1:0] rd_addr       = underflow ? (wr_addr + D - (D-1)) : rd_addr_minus;

        reg [D-1:0] vpipe;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) vpipe <= {D{1'b0}};
            else       vpipe <= {vpipe[D-2:0], vin};
        end

        reg [$clog2(D+1)-1:0] warmup_cnt;
        reg warmed;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                warmup_cnt <= 0; warmed <= 1'b0;
            end else if (!warmed) begin
                warmup_cnt <= warmup_cnt + 1'b1;
                if (warmup_cnt == D-1) warmed <= 1'b1;
            end
        end

        (* ram_style = "block" *) reg [W-1:0] mem [0:D-1];
        always @(posedge clk) begin
            mem[wr_addr] <= din;
        end

        reg [W-1:0] dout_r;
        always @(posedge clk) begin
            dout_r <= mem[rd_addr];
        end

        assign dout = dout_r;
        assign vout = warmed ? vpipe[D-1] : 1'b0;
    end endgenerate
endmodule

// ------------------------------------------------------------
// 간단 FP16 스칼라 곱 래퍼 (valid-only, II=1 기반)
// ------------------------------------------------------------
module mul_scalar #(
    parameter integer DW = 16,
    parameter integer LAT_MUL = 6
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire         valid_i,
    input  wire [DW-1:0]a_i,
    input  wire [DW-1:0]b_i,
    output wire [DW-1:0]y_o,
    output wire         valid_o
);
    // 실제 프로젝트에선 Vivado FP16 Mult IP로 대체
    // 여기선 파이프 지연만 모델링(연결 포맷 통일 목적)
    pipe_bus #(.W(DW), .D(LAT_MUL)) u_mul_d (
        .clk(clk), .rstn(rstn),
        .din(a_i /* * b_i : IP로 대체*/), .vin(valid_i),
        .dout(y_o), .vout(valid_o)
    );
    // NOTE: 위는 더미. 실제론 FP16 multiplier IP 인스턴스 사용!
endmodule
