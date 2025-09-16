// ============================================================================
// hC_mul : per-(h,p,n) FP16 mul over H_TILE × P_TILE × N_TILE lanes
//   hC(h,p,n) = hnext(h,p,n) * C(n)
//   입력  : hnext_i   (h*p*n), C_tile_i (n)
//   출력  : hC_o      (h*p*n)
//   패킹  : ((h*P_TILE + p)*N_TILE + n)
// ----------------------------------------------------------------------------
module hC_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128,
  parameter integer M_LAT   = 6   // 참고용(내부 fp16_mul_wrapper latency)
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,
  input  wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hnext_i,    // (h*p*n)
  input  wire [N_TILE*DW-1:0]                C_tile_i,   // (n)
  output wire [H_TILE*P_TILE*N_TILE*DW-1:0]  hC_o,       // (h*p*n)
  output wire                                valid_o
);

  // 슬라이스 배열
  wire [DW-1:0] hnext_hpn  [0:H_TILE*P_TILE*N_TILE-1];
  wire [DW-1:0] C_n        [0:N_TILE-1];
  wire [DW-1:0] hC_hpn     [0:H_TILE*P_TILE*N_TILE-1];
  wire          vout_lane  [0:H_TILE*P_TILE*N_TILE-1];

  genvar h, p, n;
  generate
    // C(n) 슬라이스
    for (n = 0; n < N_TILE; n = n + 1) begin : g_n_C
      assign C_n[n] = C_tile_i[DW*(n+1)-1 -: DW];
    end

    // hnext 슬라이스 & 곱
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        for (n = 0; n < N_TILE; n = n + 1) begin : g_n
          localparam integer IDX_HPN = (h*P_TILE + p)*N_TILE + n;

          assign hnext_hpn[IDX_HPN] = hnext_i[DW*(IDX_HPN+1)-1 -: DW];

          fp16_mult_wrapper u_mul (
            .clk       (clk),
            .valid_in  (valid_i),
            .a         (hnext_hpn[IDX_HPN]),
            .b         (C_n[n]),             // 같은 n을 모든 (h,p)에 브로드캐스트
            .result    (hC_hpn[IDX_HPN]),
            .valid_out (vout_lane[IDX_HPN])
          );

          assign hC_o[DW*(IDX_HPN+1)-1 -: DW] = hC_hpn[IDX_HPN];
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
        $display("[%0t] WARN(hC_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
      end
    end
  end
`endif

endmodule
