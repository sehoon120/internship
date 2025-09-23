// ------------------------------------------------------------------
// exp_sp_shared_simple.v
// - 조건: 같은 싸이클 동시요청 없음 (sp_req_v XOR exp_req_v)
// - sp_req_v 또는 exp_req_v가 1인 싸이클에만 코어에 입력(valid_i=1)
// - mode는 그 싸이클에만 1(softplus) / 0(exp)로 셋
// - 둘 다 뜨거나 둘 다 안 뜨면 예외: x를 코어에 전달(시뮬 경고)
// ------------------------------------------------------------------
module exp_sp_shared_simple #(
    parameter integer DW      = 16,
    // 아래 파라미터들은 코어 내부 지연 정보(공유기 동작엔 직접 영향 없음)
    parameter integer LAT_MUL = 1,
    parameter integer LAT_ADD = 1,
    parameter integer LAT_DIV = 1,
    parameter integer LAT_EXP = 6 + LAT_MUL*3 + LAT_ADD*3
)(
    input  wire         clk,
    input  wire         rstn,

    // ---- SP 요청 ----
    input  wire         sp_req_v,
    input  wire [DW-1:0] sp_x,
    output wire         sp_rsp_v,
    output wire [DW-1:0] sp_y,

    // ---- EXP 요청 ----
    input  wire         exp_req_v,
    input  wire [DW-1:0] exp_x,
    output wire         exp_rsp_v,
    output wire [DW-1:0] exp_y
);
    // 1) 입력 선택 (요구 1,3,4)
    wire core_vi   = sp_req_v | exp_req_v;                     // (1)
    // mode: 1=softplus, 0=exp, 예외 시 X
    wire core_mode = sp_req_v ? 1'b1 : (exp_req_v ? 1'b0 : 1'bx);  // (4)
    // 데이터 선택: 예외 시 X
    wire [DW-1:0] core_xi = sp_req_v ? sp_x : (exp_req_v ? exp_x : {DW{1'bx}}); // (3)

`ifdef SIM
    // 동시요청/무요청 예외 체크
    always @(posedge clk) if (rstn) begin
        if (sp_req_v & exp_req_v)
            $display("[%0t][ERR] exp_sp_shared_simple: simultaneous sp/exp requests!", $time);
        if (~sp_req_v & ~exp_req_v)
            ; // 무요청은 허용하려면 그대로 두고, 금지하려면 경고 찍자:
            // $display("[%0t][WARN] exp_sp_shared_simple: no request this cycle.", $time);
    end
`endif

    // 2) 단일 코어 인스턴스 (요구 1,2)
    wire [DW-1:0] y_o_S, y_o_e;
    wire          v_o_S, v_o_e;

    softplus_or_exp16 #(
        .DW(DW),
        .LAT_MUL(LAT_MUL),
        .LAT_ADD(LAT_ADD),
        .LAT_DIV(LAT_DIV),
        .LAT_EXP(LAT_EXP)
    ) u_core (
        .clk              (clk),
        .rstn             (rstn),
        .valid_i          (core_vi),       // (1)
        .mode_softplus_i  (core_mode),     // (4) 입력 싸이클에만 의미 있음
        .x_i              (core_xi),       // (3)

        .y_o_S            (y_o_S),
        .valid_o_S        (v_o_S),
        .y_o_e            (y_o_e),
        .valid_o_e        (v_o_e),
        .mode_softplus_o  ()               // 미사용
    );

    // 3) 출력 분기: 코어가 모드별 valid 제공 (요구 2)
    assign sp_y      = y_o_S;
    assign sp_rsp_v  = v_o_S;

    assign exp_y     = y_o_e;
    assign exp_rsp_v = v_o_e;

endmodule
