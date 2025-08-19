// ================================================================
// SSMBLOCK_TOP — tile-in → 8 tiles collect → sum128 → +xD → y_final
//  - B=H=P=1 (scalars: dt,dA,x,D; tile vectors: B_tile,C_tile,hprev_tile)
//  - II=1 가정 (내부 FP16 IP도 throughput=1)
//  - N_TOTAL=128, N_TILE=16 → TILES_PER_GROUP=8
// 고쳐야할것: 지금 hC_buffer에 값이 5개? 밖에 저장이 안되고있음
// xD 딜레이 타이민 정상인지 확인하기
// xD 였나 input이 다음 그룹으로 넘어갈때 신호 초기화 시키지 않아도 되는가? 확인하기
// ================================================================
module SSMBLOCK_TOP #(
    parameter integer DW        = 16,
    parameter integer H_TILE    = 1,
    parameter integer P_TILE    = 1,
    parameter integer N_TILE    = 4,
    parameter integer N_TOTAL   = 128,
    // Latency params (IP 설정에 맞춰 조정)
    parameter integer LAT_DX_M  = 6,   // dx: dt*x (mul)
    parameter integer LAT_DBX_M = 6,   // dBx: dx*B (mul)
    parameter integer LAT_DAH_M = 6,   // dAh: dA*hprev (mul)
    parameter integer LAT_ADD_A = 11,  // h_next: dAh+dBx
    parameter integer LAT_HC_M  = 6    // hC: h_next*C (mul)
)(
    input  wire                   clk,
    input  wire                   rstn,

    // 타일 유효(연속 타일 스트리밍), 마지막 타일 표시는 TB가 관리(여기선 불필요)
    input  wire                   tile_valid_i,
    output wire                   tile_ready_o,   // 필요시 backpressure, 기본 1

    // Scalars
    input  wire [DW-1:0]          dt_i,
    input  wire [DW-1:0]          dA_i,
    input  wire [DW-1:0]          x_i,
    input  wire [DW-1:0]          D_i,

    // Tile vectors (N_TILE)
    input  wire [N_TILE*DW-1:0]   B_tile_i,
    input  wire [N_TILE*DW-1:0]   C_tile_i,
    input  wire [N_TILE*DW-1:0]   hprev_tile_i,

    // 최종 출력: y = sum_{n=0..127} hC[n] + x*D  (N=128 처리 끝날 때 1펄스)
    output wire [DW-1:0]          y_final_o,
    output wire                   y_final_valid_o
);
    // ------------------------------------------------------------
    localparam integer TILES_PER_GROUP = N_TOTAL / N_TILE; // 8

    // ============================================================
    // 1) dx = dt * x   (scalar)
    // ============================================================
    wire [DW-1:0] dx_w;
    wire          v_dx;

    dx #(.DW(DW), .MUL_LAT(LAT_DX_M)) u_dx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (tile_valid_i), // 첫 타일부터 계속 공급(II=1)
        .dt_i    (dt_i),
        .x_i     (x_i),
        .dx_o    (dx_w),
        .valid_o (v_dx)
    );

    // ============================================================
    // 2) dBx = dx * B[n] (N_TILE 병렬)
    // ============================================================
    wire [N_TILE*DW-1:0] dBx_w;
    wire                 v_dBx;
    localparam integer B_W = N_TILE*DW;
    integer bi;
    reg [B_W-1:0] B_tile_buffer [0:LAT_DX_M];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (bi = 0; bi <= LAT_DX_M; bi = bi + 1) begin
                B_tile_buffer[bi] <= {B_W{1'b0}};
            end
        end else begin
            // B 입력 시프트
            B_tile_buffer[0] <= B_tile_i;
            for (bi = 1; bi <= LAT_DX_M; bi = bi + 1) begin
                B_tile_buffer[bi] <= B_tile_buffer[bi-1];
            end
        end
    end
    wire [B_W-1:0] B_tile_aligned = B_tile_buffer[LAT_DX_M];

    dBx #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DBX_M)) u_dBx (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (v_dx),
        .dx_i    (dx_w),
        .Bmat_i  (B_tile_aligned),  // B_mat가 LAT_M 만큼 딜레이된 값이 들어가야함 - 6사이클
        .dBx_o   (dBx_w),
        .valid_o (v_dBx)
    );

    // ============================================================
    // 3) dAh = dA * hprev[n] (N_TILE 병렬) + dBx 경로와 정렬
    // ============================================================
    wire [N_TILE*DW-1:0] dAh_raw_w, dAh_w;
    wire                 v_dAh_raw,  v_dAh;

    dAh #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_DAH_M)) u_dAh (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (tile_valid_i),
        .dA_i     (dA_i),
        .hprev_i  (hprev_tile_i),
        .dAh_o    (dAh_raw_w),
        .valid_o  (v_dAh_raw)
    );

    // dAh가 더 빨리 나오면 그만큼 늦춰서 h_next 입력 타이밍 맞춤
    localparam integer DLY_DAH_ALIGN = (LAT_DX_M + LAT_DBX_M) - LAT_DAH_M;
    wire [N_TILE*DW-1:0] dAh_dly_w;
    wire                 v_dAh_dly;

    pipe_bus #(.W(N_TILE*DW), .D((DLY_DAH_ALIGN>0)?DLY_DAH_ALIGN:0)) u_dly_dAh_bus (
        .clk   (clk), .rstn(rstn),
        .din   (dAh_raw_w), .vin(v_dAh_raw),
        .dout  (dAh_dly_w), .vout(v_dAh_dly)
    );

    assign dAh_w = (DLY_DAH_ALIGN>0) ? dAh_dly_w : dAh_raw_w;
    assign v_dAh = (DLY_DAH_ALIGN>0) ? v_dAh_dly : v_dAh_raw;

    // ============================================================
    // 4) h_next = dBx + dAh (lane-wise)
    // ============================================================
    wire [N_TILE*DW-1:0] hnext_w;
    wire                 v_hnext;

    h_next #(.DW(DW), .N_TILE(N_TILE), .ADD_LAT(LAT_ADD_A)) u_hnext (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (v_dBx & v_dAh),
        .dBx_i    (dBx_w),
        .dAh_i    (dAh_w),
        .hnext_o  (hnext_w),
        .valid_o  (v_hnext)
    );

    // ============================================================
    // 5) hC = h_next * C[n] (lane-wise) → 타일 hC[16]
    // ============================================================
    wire [N_TILE*DW-1:0] hC_tile_o;
    wire                 v_hC;
    localparam integer C_W = N_TILE*DW;
    integer ci;
    reg [C_W-1:0] C_tile_buffer [0:LAT_DX_M+LAT_DBX_M+LAT_ADD_A];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (ci = 0; ci <= LAT_DX_M+LAT_DBX_M+LAT_ADD_A; ci = ci + 1) begin
                C_tile_buffer[ci] <= {C_W{1'b0}};
            end
        end else begin
            // B 입력 시프트
            C_tile_buffer[0] <= C_tile_i;
            for (ci = 1; ci <= LAT_DX_M+LAT_DBX_M+LAT_ADD_A; ci = ci + 1) begin
                C_tile_buffer[ci] <= C_tile_buffer[ci-1];
            end
        end
    end
    wire [C_W-1:0] C_tile_aligned = C_tile_buffer[LAT_DX_M+LAT_DBX_M+LAT_ADD_A];

    hC #(.DW(DW), .N_TILE(N_TILE), .MUL_LAT(LAT_HC_M)) u_hC (
        .clk      (clk),
        .rstn     (rstn),
        .valid_i  (v_hnext),
        .hnext_i  (hnext_w),
        .C_i      (C_tile_aligned),  // C도 여기까지만큼 딜레이 시켜서 넣어야함 - 23사이클
        .hC_o     (hC_tile_o),
        .valid_o  (v_hC)
    );

    // ============================================================
    // 6) 타일 수집기: hC[16]을 8장 모아 128-lane 버스 구성
    // ============================================================
    reg  [N_TILE*DW-1:0] hC_buf [0:TILES_PER_GROUP-1]; // 8개 타일 버퍼
    reg  [2:0]           tile_ptr;   // 0..7
    reg                  grp_emit;   // 이번 싸이클에 8타일이 모였다는 펄스
    wire                 accept_tile = v_hC;

    // xD = x*D (한 그룹의 첫 타일에서 한 번 계산 후 보관)
    // reg  [DW-1:0] xD_hold;
    // reg           xD_hold_v;
    wire [DW-1:0] xD_w;
    wire          v_xD_w;

    // 필요하면 여길 네가 쓰는 xD 래퍼로 교체해도 됨
    xD u_mul_xD (  // x, D input들도 딜레이해서 사용해야함
        .clk       (clk),
        .rstn      (rstn), 
        .valid_i  (accept_tile && (tile_ptr==3'd0)),
        .x_i         (x_i),
        .D_i         (D_i),
        .xD_o    (xD_w),
        .valid_o (v_xD_w)
    );

    integer xDi;
    localparam integer SHIFT_xD = LAT_DBX_M+LAT_ADD_A+LAT_HC_M+77;
    reg [DW-1:0] xD_tile_buffer [0:SHIFT_xD];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (xDi = 0; xDi <= SHIFT_xD; xDi = xDi + 1) begin
                xD_tile_buffer[xDi] <= {DW{1'b0}};
            end
        end else begin
            // B 입력 시프트
            xD_tile_buffer[0] <= xD_w;
            for (xDi = 1; xDi <= SHIFT_xD; xDi = xDi + 1) begin
                xD_tile_buffer[xDi] <= xD_tile_buffer[xDi-1];
            end
        end
    end
    wire [DW-1:0] xD_tile_aligned = xD_tile_buffer[SHIFT_xD];


    integer ti;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            tile_ptr  <= 3'd0;
            grp_emit  <= 1'b0;
            // xD_hold   <= {DW{1'b0}};
            // xD_hold_v <= 1'b0;
            for (ti=0; ti<TILES_PER_GROUP; ti=ti+1) hC_buf[ti] <= {N_TILE*DW{1'b0}};
        end else begin
            grp_emit <= 1'b0;

            if (accept_tile) begin
                // 현재 타일 저장
                hC_buf[tile_ptr] <= hC_tile_o;

                // xD 보관 (첫 타일에서 계산 완료되면 래치)
                // if (v_xD_w) begin
                    // xD_hold   <= xD_w;
                    // xD_hold_v <= 1'b1;
                // end

                // 타일 포인터 증가 및 그룹 완료
                if (tile_ptr == TILES_PER_GROUP-1) begin
                    tile_ptr <= 3'd0;
                    grp_emit <= 1'b1;    // 8번째 타일이 막 들어온 싸이클
                end else begin
                    tile_ptr <= tile_ptr + 3'd1;
                end
            end
        end
    end

    // 8개 타일을 128-lane 버스로 평탄화
    wire [N_TOTAL*DW-1:0] hC_128_bus;
    assign hC_128_bus = {
        hC_buf[7], hC_buf[6], hC_buf[5], hC_buf[4],
        hC_buf[3], hC_buf[2], hC_buf[1], hC_buf[0]
    };

    // ============================================================
    // 7) 128합 트리: y_tmp = Σ_{n=0..127} hC[n]
    //    (네 모듈 포트명에 맞춰 연결. 아래는 예시)
    // ============================================================
    wire [DW-1:0] y_tmp_w;
    wire          y_tmp_v;

    fp16_adder_tree_128 u_sum128 (
        .clk       (clk),
        .rst       (rstn),          // 네 모듈이 rstn이면 포트명만 바꿔
        .valid_in  (grp_emit),      // 8타일 모였을 때 1싸이클 펄스
        .in_flat   (hC_128_bus),    // 128*DW 입력
        .sum       (y_tmp_w),       // Σ128 결과
        .valid_out (y_tmp_v)        // 트리 내부 지연 후 1
    );

    // ============================================================
    // 8) 최종 y = y_tmp + xD  (그룹 완료 시점에 1펄스)
    // ============================================================
    wire [DW-1:0] y_final_w;
    wire          v_y_final_w;

    y_out u_add_yfinal (
        .clk       (clk),
        .rstn      (rstn),
        .valid_i  (y_tmp_v),
        .ytmp_i         (y_tmp_w),
        .xD_i         (xD_tile_aligned),
        .y_o    (y_final_w),
        .valid_o (v_y_final_w)
    );

    assign y_final_o        = y_final_w;
    assign y_final_valid_o  = v_y_final_w;

    // 타일 입력 항상 수락 (필요시 내부 ready와 AND 하세요)
    assign tile_ready_o = 1'b1;

endmodule


// ------------------------------------------------------------
// Data+valid pipeline utility (데이터+valid를 D싸이클 지연)
// ------------------------------------------------------------
module pipe_bus #(
    parameter integer W = 16,
    parameter integer D = 0
)(
    input  wire             clk,
    input  wire             rstn,
    input  wire [W-1:0]     din,
    input  wire             vin,
    output wire [W-1:0]     dout,
    output wire             vout
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
