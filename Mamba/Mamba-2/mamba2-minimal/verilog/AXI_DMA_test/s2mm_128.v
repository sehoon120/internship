// s2mm_128.v : stream-to-memory (128b) simple DMA surrogate
module s2mm_128 #(
  parameter ADDR_W = 12
)(
  input  wire        clk, rstn,
  // control
  input  wire        start,
  input  wire [31:0] byte_len,
  input  wire [ADDR_W-1:0] base,
  output reg         busy, done,
  // BRAM write port
  output reg         wr_en,
  output reg [ADDR_W-1:0] wr_addr,
  output reg [127:0] wr_data,
  // AXIS in
  input  wire [127:0] s_tdata,
  input  wire         s_tvalid,
  output wire         s_tready,
  input  wire         s_tlast
);
  localparam WBYTES = 16;
  assign s_tready = busy; // 진행 중일 때만 수신

  reg [31:0] bytes_left;

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      busy<=0; done<=0; wr_en<=0; wr_addr<=0; wr_data<=0; bytes_left<=0;
    end else begin
      done <= 1'b0;
      wr_en <= 1'b0;
      if (!busy) begin
        if (start) begin
          busy <= 1'b1;
          bytes_left <= byte_len;
          wr_addr <= base;
        end
      end else begin
        if (s_tvalid && s_tready) begin
          wr_data <= s_tdata;
          wr_en   <= 1'b1;
          wr_addr <= wr_addr + 1;
          if (bytes_left <= WBYTES) begin
            busy <= 1'b0; done <= 1'b1;
          end else begin
            bytes_left <= bytes_left - WBYTES;
          end
        end
      end
    end
  end
endmodule
