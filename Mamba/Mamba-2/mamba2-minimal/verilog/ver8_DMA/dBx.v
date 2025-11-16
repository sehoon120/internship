// ============================================================================
// dBx_mul : per-(h,p,n) FP16 mul over H_TILE × P_TILE × N_TILE lanes
//   dBx(h,p,n) = dx(h,p) * B(n)
//   입력  : dx_i      (h*p)
//          B_tile_i  (n)
//   출력  : dBx_o     (h*p*n)  // 패킹 순서: ((h*P_TILE + p)*N_TILE + n)
// ----------------------------------------------------------------------------
module dBx_mul #(
  parameter integer DW      = 16,
  parameter integer H_TILE  = 1,
  parameter integer P_TILE  = 1,
  parameter integer N_TILE  = 128,
  parameter integer M_LAT   = 6   // 참고용(내부 wrapper latency)
)(
  input  wire                                clk,
  input  wire                                rstn,
  input  wire                                valid_i,
  input  wire [H_TILE*P_TILE*DW-1:0]         dx_i,       // (h*p)
  input  wire [N_TILE*DW-1:0]                B_tile_i,   // (n)
  output wire [H_TILE*P_TILE*N_TILE*DW-1:0]  dBx_o,      // (h*p*n)
  output wire                                valid_o
);

  // 슬라이스 배열
  wire [DW-1:0] dx_hp     [0:H_TILE*P_TILE-1];
  wire [DW-1:0] B_n       [0:N_TILE-1];
  wire [DW-1:0] dBx_hpn   [0:H_TILE*P_TILE*N_TILE-1];
  wire          vout_lane [0:H_TILE*P_TILE*N_TILE-1];

  genvar h, p, n;
  generate
    // dx(h,p) 슬라이스
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h_dx
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p_dx
        localparam integer IDX_HP = h*P_TILE + p;
        assign dx_hp[IDX_HP] = dx_i[DW*(IDX_HP+1)-1 -: DW];
      end
    end

    // B(n) 슬라이스
    for (n = 0; n < N_TILE; n = n + 1) begin : g_n_B
      assign B_n[n] = B_tile_i[DW*(n+1)-1 -: DW];
    end

    // (h,p,n) 곱: dBx(h,p,n) = dx(h,p) * B(n)
    for (h = 0; h < H_TILE; h = h + 1) begin : g_h
      for (p = 0; p < P_TILE; p = p + 1) begin : g_p
        localparam integer IDX_HP = h*P_TILE + p;
        for (n = 0; n < N_TILE; n = n + 1) begin : g_n
          localparam integer IDX_HPN = (IDX_HP*N_TILE) + n;

          fp16_mult_wrapper u_mul (
            .clk       (clk),
            .valid_in  (valid_i),
            .a         (dx_hp[IDX_HP]), // 같은 (h,p)를 모든 n에 브로드캐스트
            .b         (B_n[n]),
            .result    (dBx_hpn[IDX_HPN]),
            .valid_out (vout_lane[IDX_HPN])
          );

          // 패킹: ((h*P_TILE + p)*N_TILE + n) 순서
          assign dBx_o[DW*(IDX_HPN+1)-1 -: DW] = dBx_hpn[IDX_HPN];
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
        $display("[%0t] WARN(dBx_mul): lane valid mismatch: lane%0d=%b lane0=%b",
                 $time, k, vout_lane[k], vout_lane[0]);
      end
    end
  end
`endif

endmodule
