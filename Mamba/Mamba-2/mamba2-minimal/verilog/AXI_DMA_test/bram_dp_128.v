// bram_dp_128.v : simple dual-port BRAM (sync read, write-first)
module bram_dp_128 #(
  parameter ADDR_W = 12  // 4KB @128b : 4096/16 = 256 words -> 8 bits로도 가능
)(
  input  wire              clk,
  // Port A (read)
  input  wire              a_en,
  input  wire [ADDR_W-1:0] a_addr,
  output reg  [127:0]      a_rdata,
  // Port B (write)
  input  wire              b_en,
  input  wire              b_we,
  input  wire [ADDR_W-1:0] b_addr,
  input  wire [127:0]      b_wdata
);
  reg [127:0] mem [0:(1<<ADDR_W)-1];
  always @(posedge clk) begin
    if (a_en) a_rdata <= mem[a_addr];
    if (b_en && b_we) mem[b_addr] <= b_wdata;
  end
endmodule
