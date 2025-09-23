// ------------------------------------------------------------------
// exp_sp_shared.v
// 단일 코어(softplus_or_exp16_core)를 SP/EXP가 공유 (time-multiplexed)
// - Throughput=1 가정 (매 싸이클 입력 수락 가능)
// - 같은 싸이클에 SP/EXP 동시 요청시 1-딥 스키드 버퍼로 처리
// - 태그(0=SP, 1=EXP)를 코어 지연만큼 밀어 출력 경로 디멀티플렉스
// ------------------------------------------------------------------
module exp_sp_shared #(
    parameter integer DW       = 16,
    parameter integer H_TILE   = 1,   // vector lanes for (h)
    // 코어 지연: exp가 더 크면 L_CORE=LAT_EXP로 잡는 것을 권장
    parameter integer LAT_EXP  = 32,
    parameter integer LAT_SP   = 28
)(
    input  wire                   clk,
    input  wire                   rstn,

    // ---- SP 요청 (delta -> softplus) ----
    input  wire                   sp_req_v,                 // v_delta_w
    input  wire [H_TILE*DW-1:0]   sp_x,                     // delta_w
    output wire                   sp_rsp_v,                 // v_delta_sp_w (shared)
    output wire [H_TILE*DW-1:0]   sp_y,                     // delta_sp_w

    // ---- EXP 요청 (dA_tmp -> exp) ----
    input  wire                   exp_req_v,                // v_dA_tmp_w
    input  wire [H_TILE*DW-1:0]   exp_x,                    // dA_tmp_w
    output wire                   exp_rsp_v,                // v_dA_w (shared)
    output wire [H_TILE*DW-1:0]   exp_y                     // dA_w
);
    // ===== 파라미터/상수 =====
    localparam integer L_CORE = (LAT_EXP > LAT_SP) ? LAT_EXP : LAT_SP;

    // ===== 1-딥 스키드 버퍼(각 포트) =====
    reg                  sp_buf_v,  exp_buf_v;
    reg [H_TILE*DW-1:0]  sp_buf_x,  exp_buf_x;

    wire sp_avail = sp_buf_v | sp_req_v;
    wire exp_avail= exp_buf_v | exp_req_v;

    // 입력 합치기: 간단 라운드로빈(여기선 SP 우선 → EXP로 토글)
    reg rr_sel; // 0이면 SP 먼저, 1이면 EXP 먼저 시도
    wire pick_sp  = sp_avail & (~exp_avail | (rr_sel==1'b0));
    wire pick_exp = exp_avail & (~sp_avail | (rr_sel==1'b1));

    wire [H_TILE*DW-1:0] picked_x   = pick_sp  ? (sp_buf_v ? sp_buf_x : sp_x)
                                               : (exp_buf_v ? exp_buf_x : exp_x);
    wire                 picked_tag = pick_sp ? 1'b0 : 1'b1; // 0=SP, 1=EXP
    wire                 accept     = sp_avail | exp_avail;  // 코어는 매 싸이클 1개 수락

    // 스키드 버퍼 적재/소비
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sp_buf_v <= 1'b0; exp_buf_v <= 1'b0; rr_sel <= 1'b0;
        end else begin
            // 새로 들어온 요청을 버퍼에 저장(해당 포트를 이번 싸이클에 안뽑으면)
            if (sp_req_v && !(pick_sp && !sp_buf_v)) begin
                sp_buf_v <= 1'b1;
                sp_buf_x <= sp_x;
            end else if (pick_sp && sp_buf_v) begin
                sp_buf_v <= 1'b0; // 버퍼에서 꺼내씀
            end

            if (exp_req_v && !(pick_exp && !exp_buf_v)) begin
                exp_buf_v <= 1'b1;
                exp_buf_x <= exp_x;
            end else if (pick_exp && exp_buf_v) begin
                exp_buf_v <= 1'b0;
            end

            // 라운드로빈 토글
            if (accept) rr_sel <= ~rr_sel;
        end
    end

    // ===== 코어 구동 =====
    // 태그 파이프: 코어 지연만큼 태그를 밀어 출력 분기
    reg [L_CORE-1:0] tag_pipe;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) tag_pipe <= {L_CORE{1'b0}};
        else       tag_pipe <= {tag_pipe[L_CORE-2:0], (accept ? picked_tag : 1'b0)};
    end
    wire tag_out = tag_pipe[L_CORE-1]; // 0=SP, 1=EXP

    // 입력 유효 신호(accept)로 한 싸이클당 1개 투입
    wire                 core_vi = accept;
    wire [H_TILE*DW-1:0] core_xi = picked_x;
    wire                 core_mode = ~picked_tag; 
    // convention: mode=1 => softplus(SP), mode=0 => exp(EXP). (원한다면 반대로 맞춰도 됨)

    wire                 core_vo;
    wire [H_TILE*DW-1:0] core_yo;

    // 네가 가진 코어로 교체:
    // softplus_or_exp16_core #(.DW(DW), .H_TILE(H_TILE), .LAT_EXP(LAT_EXP), .LAT_SP(LAT_SP)) u_core ( ... );
    softplus_or_exp16_core #(.DW(DW), .H_TILE(H_TILE)) u_core (
        .clk     (clk),
        .rstn    (rstn),
        .valid_i (core_vi),
        .mode_i  (core_mode),     // 1=SP, 0=EXP
        .x_i     (core_xi),
        .y_o     (core_yo),
        .valid_o (core_vo)
    );

    // ===== 출력 디멀티플렉스 =====
    assign sp_rsp_v  = core_vo & (tag_out==1'b0);
    assign exp_rsp_v = core_vo & (tag_out==1'b1);

    // 둘 다 같은 데이터에서 갈라짐(무효일 땐 무시)
    assign sp_y   = core_yo;
    assign exp_y  = core_yo;

endmodule
