// top_dma_path.v : BRAM ↔ MM2S → (AXIS mod) → S2MM ↔ BRAM
module top_dma_path #(
  parameter ADDR_W = 12
)(
  input  wire        clk, rstn,
  // ctrl
  input  wire        go_mm2s,
  input  wire        go_s2mm,
  input  wire [31:0] byte_len,
  input  wire [ADDR_W-1:0] in_base,
  input  wire [ADDR_W-1:0] out_base,
  output wire        mm2s_done,
  output wire        s2mm_done
);
  // BRAM
  wire a_en   = 1'b1;
  wire b_en   = 1'b1;
  wire [127:0] a_rdata;
  wire [ADDR_W-1:0] rd_addr, wr_addr;
  wire rd_en, wr_en;
  wire [127:0] wr_data;

  bram_dp_128 #(.ADDR_W(ADDR_W)) u_bram (
    .clk(clk),
    .a_en(a_en), .a_addr(rd_addr), .a_rdata(a_rdata),
    .b_en(b_en), .b_we(wr_en), .b_addr(wr_addr), .b_wdata(wr_data)
  );

  // MM2S
  wire [127:0] m_tdata;
  wire m_tvalid, m_tready, m_tlast;
  wire mm2s_busy;

  mm2s_128 #(.ADDR_W(ADDR_W)) u_mm2s (
    .clk(clk), .rstn(rstn),
    .start(go_mm2s), .byte_len(byte_len), .base(in_base),
    .busy(mm2s_busy), .done(mm2s_done),
    .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(a_rdata),
    .m_tdata(m_tdata), .m_tvalid(m_tvalid), .m_tready(m_tready), .m_tlast(m_tlast)
  );

  // AXIS module (fp16 add)
  wire [127:0] p_tdata;
  wire p_tvalid, p_tready, p_tlast;

  axis_fp16_add1p0_128 u_filter (
    .aclk(clk), .aresetn(rstn),
    .s_tdata(m_tdata), .s_tvalid(m_tvalid), .s_tready(m_tready), .s_tlast(m_tlast),
    .m_tdata(p_tdata), .m_tvalid(p_tvalid), .m_tready(p_tready), .m_tlast(p_tlast)
  );

  // S2MM
  wire s2mm_busy;
  assign p_tready = 1'b1; // 간단: 항상 수신 가능 → 실제론 s2mm_busy와 연동 가능

  s2mm_128 #(.ADDR_W(ADDR_W)) u_s2mm (
    .clk(clk), .rstn(rstn),
    .start(go_s2mm), .byte_len(byte_len), .base(out_base),
    .busy(s2mm_busy), .done(s2mm_done),
    .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
    .s_tdata(p_tdata), .s_tvalid(p_tvalid), .s_tready(/*unused*/),
    .s_tlast(p_tlast)
  );

endmodule
