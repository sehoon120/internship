// ============================================================================
// dAh_mul : per-(h,p,n) FP16 mul over H_TILE × P_TILE × N_TILE lanes
//   dAh(h,p,n) = dA(h) * hprev(h,p,n)
//   입력  : dA_i         (h)
//           hprev_i      (h*p*n)
//   출력  : dAh_o        (h*p*n)  // 패킹 순서: ((h*P_TILE + p)*N_TILE + n)
// ----------------------------------------------------------------------------
module dAh_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128,
  parameter integer M_LAT   = 6   // 참고용(내부 wrapper latency)
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,
  input  wire [H_TILE*DW-1:0]                dA_i,       // (h)
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hprev_i,    // (h*p*n)
  output wire [H_TILE*P_TILE*N_TILE*DW-1:0]  dAh_o,      // (h*p*n)
  output wire                                valid_o
);

  // 슬라이스 배열
  wire [DW-1:0] dA_h       [0:H_TILE-1];
  wire [DW-1:0] hprev_hpn  [0:H_TILE*P_TILE*N_TILE-1];
  wire [DW-1:0] dAh_hpn    [0:H_TILE*P_TILE*N_TILE-1];
  wire          vout_lane  [0:H_TILE*P_TILE*N_TILE-1];

  genvar h, p, n;
  generate
    // dA(h) 슬라이스
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h_dA
      assign dA_h[h] = dA_i[DW*(h+1)-1 -: DW];
    end

    // hprev(h,p,n) 슬라이스
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h_prev
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p_prev
        for (n = 0; n < N_TILE; n = n + 1) begin : g_n_prev
          localparam integer IDX_HPN = (h*P_TILE + p)*N_TILE + n;
          assign hprev_hpn[IDX_HPN] = hprev_i[DW*(IDX_HPN+1)-1 -: DW];
        end
      end
    end

    // (h,p,n) 곱: dAh(h,p,n) = dA(h) * hprev(h,p,n)
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        for (n = 0; n < N_TILE; n = n + 1) begin : g_n
          localparam integer IDX_HPN = (h*P_TILE + p)*N_TILE + n;

          fp16_mult_wrapper u_mul (
            .clk       (clk),
            .valid_in  (valid_i),
            .a         (dA_h[h]),          // 같은 h 값을 모든 (p,n)에 브로드캐스트
            .b         (hprev_hpn[IDX_HPN]),
            .result    (dAh_hpn[IDX_HPN]),
            .valid_out (vout_lane[IDX_HPN])
          );

          // 패킹: ((h*P_TILE + p)*N_TILE + n) 순서
          assign dAh_o[DW*(IDX_HPN+1)-1 -: DW] = dAh_hpn[IDX_HPN];
        end
      end
    end
  endgenerate

  // 동일 latency 가정 → AND 결합
  assign valid_o = vout_lane[0];  // &vout_lane;

`ifdef SIM
  // (옵션) lane valid 동기성 체크
  integer k;
  always @(posedge clk) if (rstn) begin
    for (k = 1; k < H_TILE*P_TILE*N_TILE; k = k + 1) begin
      if (vout_lane[k] !== vout_lane[0]) begin
        $display("[%0t] WARN(dAh_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
      end
    end
  end
`endif

endmodule
