`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09/18/2025 03:19:31 PM
// Design Name: 
// Module Name: pipe_bus_bram
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


// ------------------------------------------------------------ 
// Data+valid pipeline with BRAM (fixed D-cycle delay, wall-clock)
//   - Stores W-bit samples in a circular buffer
//   - Uses delayed write address as read address
//   - Vivado will map to BRAM when depth/width is large
// ------------------------------------------------------------

module pipe_bus_bram #(
    parameter integer W = 256,
    parameter integer D = 6,
    parameter integer USE_V = 1  // 1: data+valid, 0: data-only
)(
    input  wire         clk,
    input  wire         rstn,
    input  wire [W-1:0] din,
    input  wire         vin,     // USE_V=0이면 무시 가능
    output wire [W-1:0] dout,
    output wire         vout     // USE_V=0이면 1로 고정(워밍업 지나면)
);
    generate if (D == 0) begin
        assign dout = din;
        assign vout = (USE_V? vin : 1'b1);
    end else begin
        localparam integer ADDR_W = $clog2(D);
        reg [ADDR_W-1:0] wr_addr;

        // 0..D-1 래핑 카운터
        always @(posedge clk or negedge rstn) begin
            if (!rstn) wr_addr <= {ADDR_W{1'b0}};
            else if (wr_addr == D-1) wr_addr <= {ADDR_W{1'b0}};
            else                     wr_addr <= wr_addr + 1'b1;
        end

        // **수정 포인트**: rd_addr = (wr_addr + 1) mod D  (D-1 오프셋 보정의 단순화)
        wire [ADDR_W-1:0] rd_addr = (wr_addr == D-1) ? {ADDR_W{1'b0}} : (wr_addr + 1'b1);

        // 워밍업: 초기 D사이클 동안만 출력 무시(연속 스트리밍 가정)
        reg [$clog2(D+1)-1:0] warmup_cnt;
        reg warmed;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                warmup_cnt <= 0; warmed <= 1'b0;
            end else if (!warmed) begin
                warmup_cnt <= warmup_cnt + 1'b1;
                if (warmup_cnt == D-1) warmed <= 1'b1;
            end
        end

        // BRAM
        (* ram_style = "block" *) reg [W-1:0] mem [0:D-1];

        // write: USE_V=1이면 vin일 때만 쓰고, 0이면 매 클럭 쓰기
        always @(posedge clk) begin
            if (USE_V) begin
                if (vin) mem[wr_addr] <= din;
            end else begin
                mem[wr_addr] <= din;
            end
        end

        // 1-cycle read
        reg [W-1:0] dout_r;
        always @(posedge clk) begin
            dout_r <= mem[rd_addr];
        end
        assign dout = dout_r;

        // vout
        if (USE_V) begin : GEN_V
            // valid 파이프 대신 간단화: vin을 D+1 사이클 지연해도 되지만,
            // BRAM read 1사이클 포함했으니 D단 파이프면 맞음
            reg [D-1:0] vpipe;
            always @(posedge clk or negedge rstn) begin
                if (!rstn) vpipe <= {D{1'b0}};
                else       vpipe <= {vpipe[D-2:0], vin};
            end
            assign vout = warmed ? vpipe[D-1] : 1'b0;
        end else begin : GEN_NO_V
            assign vout = warmed ? 1'b1 : 1'b0; // 연속 스트리밍: 워밍업 후 상시 1
        end
    end endgenerate
endmodule
