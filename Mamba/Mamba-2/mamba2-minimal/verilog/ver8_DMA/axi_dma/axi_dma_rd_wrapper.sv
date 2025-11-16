`timescale 1 ns / 1 ps
module axi_dma_rd_wrapper #(
    parameter NUM_MO_BUF = 4,
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 256,
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    `include "assign.svh"
    // AXI config, Start AXI transactions (read / write)
    input  wire                   init_read,
    input  wire  [ADDR_WIDTH-1:0] axi_start_addr,
    input  wire  [          31:0] axi_byte_length,
    output logic                  axi_start_ready,
    output logic                  axi_start_valid,
    output logic                  axi_idle,

    // AXIS Interface
    axis.master __axis_mm2s,

    // AXI Interface
    aximm.master __aximm,
    input wire clk,
    input wire rstn
);

    aximm_rd __aximm_rd ();
    `AXI_ASSIGN_AR(assign, __aximm, __aximm_rd)
    `AXI_ASSIGN_R(assign, __aximm_rd, __aximm)
    // module that provides To x 3 x 3 x 3 sized weights
    axi_dma_rd #(
        .NUM_MO_BUF(NUM_MO_BUF),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .data_t(data_t),
        .addr_t(addr_t),
        .trans_t(trans_t)
    ) axi_dma_rd_inst (
        .axi_start_addr(axi_start_addr),
        .axi_byte_length(axi_byte_length),
        .init_read(init_read),
        .axi_idle(axi_idle),
        .axi_start_ready(axi_start_ready),
        .axi_start_valid(axi_start_valid),
        .__axis_mm2s(__axis_mm2s),
        .clk(clk),
        .rstn(rstn),
        .__aximm_rd(__aximm_rd)
    );
endmodule
