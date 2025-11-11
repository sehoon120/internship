`timescale 1 ns / 1 ps
module axi_dma #(
    parameter NUM_RD_MO_BUF = 4,
    parameter NUM_WR_MO_BUF = 4,
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 256,
    parameter integer WR_AXIS_FIFO_DEPTH = 16,
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    `include "assign.svh"

    // AXI config, Start AXI transactions (read / write)
    input  addr_t        axi_read_start_addr,
    input  addr_t        axi_write_start_addr,
    input  logic  [31:0] axi_read_length,
    input  logic  [31:0] axi_write_length,
    input  logic         init_read,
    input  logic         init_write,
    output logic         axi_write_start_ready,
    output logic         axi_write_start_valid,
    output logic         axi_read_start_ready,
    output logic         axi_read_start_valid,
    output logic         axi_dma_wr_idle,
    output logic         axi_dma_rd_idle,

    // AXIS Interface
    axis.master __axis_mm2s,
    axis.slave  __axis_s2mm,

    // AXI MM Interface
    aximm.master __aximm,
    input wire m_axi_aclk,
    input wire m_axi_aresetn
);

    aximm_rd __aximm_rd ();
    `AXI_ASSIGN_AR(assign, __aximm, __aximm_rd)
    `AXI_ASSIGN_R(assign, __aximm_rd, __aximm)
    axi_dma_rd #(
        .NUM_MO_BUF(NUM_RD_MO_BUF),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .data_t    (data_t),
        .addr_t    (addr_t),
        .trans_t   (trans_t)
    ) u_axi_dma_rd (
        .init_read      (init_read),
        .axi_start_addr (axi_read_start_addr),
        .axi_byte_length(axi_read_length),
        .axi_start_ready(axi_read_start_ready),
        .axi_start_valid(axi_read_start_valid),
        .axi_idle       (axi_dma_rd_idle),
        .__axis_mm2s    (__axis_mm2s),
        .clk            (m_axi_aclk),
        .rstn           (m_axi_aresetn),
        .__aximm_rd     (__aximm_rd)
    );

    aximm_wr __aximm_wr ();
    `AXI_ASSIGN_AW(assign, __aximm, __aximm_wr)
    `AXI_ASSIGN_W(assign, __aximm, __aximm_wr)
    `AXI_ASSIGN_B(assign, __aximm_wr, __aximm)
    axi_dma_wr #(
        .NUM_MO_BUF   (NUM_WR_MO_BUF),
        .ADDR_WIDTH   (ADDR_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .WR_FIFO_DEPTH(WR_AXIS_FIFO_DEPTH),
        .data_t       (data_t),
        .addr_t       (addr_t),
        .trans_t      (trans_t)
    ) u_axi_dma_wr (
        .init_write     (init_write),
        .axi_start_addr (axi_write_start_addr),
        .axi_byte_length(axi_write_length),
        .axi_start_ready(axi_write_start_ready),
        .axi_start_valid(axi_write_start_valid),
        .axi_idle       (axi_dma_wr_idle),
        .__axis_s2mm    (__axis_s2mm),
        .clk            (m_axi_aclk),
        .rstn           (m_axi_aresetn),
        .__aximm_wr     (__aximm_wr)
    );
endmodule
