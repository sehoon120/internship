// mm2s_128.v : memory-to-stream (128b) simple DMA surrogate
module mm2s_128 #(
  parameter ADDR_W = 12
)(
  input  wire        clk, rstn,
  // control
  input  wire        start,
  input  wire [31:0] byte_len,      // 전송 바이트 수
  input  wire [ADDR_W-1:0] base,    // BRAM word addr (128b 단위)
  output reg         busy, done,
  // BRAM read port
  output reg         rd_en,
  output reg [ADDR_W-1:0] rd_addr,
  input  wire [127:0] rd_data,
  // AXIS out
  output reg [127:0] m_tdata,
  output reg         m_tvalid,
  input  wire        m_tready,
  output reg         m_tlast
);
  localparam WBYTES = 16; // 128b
  reg [31:0] bytes_left;
  reg        have_data;

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      busy<=0; done<=0; rd_en<=0; rd_addr<=0;
      m_tdata<=0; m_tvalid<=0; m_tlast<=0; bytes_left<=0; have_data<=0;
    end else begin
      done <= 1'b0;
      if (!busy) begin
        if (start) begin
          busy <= 1'b1;
          bytes_left <= byte_len;
          rd_addr <= base;
          rd_en <= 1'b1;  // 1st read
          have_data <= 1'b0;
          m_tvalid <= 1'b0;
          m_tlast  <= 1'b0;
        end
      end else begin
        // read latency 1: capture on next cycle
        if (rd_en) begin
          // next cycle rd_data valid
          rd_en <= 1'b0;
          have_data <= 1'b1;
          m_tdata <= rd_data;
          m_tvalid <= 1'b1;
          m_tlast  <= (bytes_left <= WBYTES);
        end

        if (m_tvalid && m_tready) begin
          // one beat consumed
          m_tvalid <= 1'b0;
          have_data <= 1'b0;
          if (bytes_left <= WBYTES) begin
            // done
            busy <= 1'b0; done <= 1'b1;
          end else begin
            bytes_left <= bytes_left - WBYTES;
            rd_addr <= rd_addr + 1;
            rd_en <= 1'b1;
          end
        end
      end
    end
  end
endmodule
