`timescale 1 ns / 1 ps
// 실전용 실제 시스템에 사용

module axi_dma_wrapper #(
    parameter NUM_RD_MO_BUF = 4,
    parameter NUM_WR_MO_BUF = 4,
    parameter WR_AXIS_FIFO_DEPTH = 1024,
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 256,
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t,
    type strb_t = axi_pkg::strb_256_t
) (
    `include "assign.svh"
    `include "port.svh"

    input wire s_axi_aclk,
    input wire s_axi_aresetn,
    `AXILITE_S_PORT(control, axi_pkg::addr_32_t, axi_pkg::data_32_t, axi_pkg::strb_32_t)

    // M_AXIS_MM2S (AXIS rd) Master Stream
    `AXIS_M_PORT(mm2s, data_t)

    // M_AXIS_S2MM (AXIS wr) Slave Stream
    `AXIS_S_PORT(s2mm, data_t)

    // AXI MM ports
    input wire m_axi_aclk,
    input wire m_axi_aresetn,
    `AXI_M_PORT(dma, addr_t, data_t, strb_t)

    // interrupt signal
    output logic axi_dma_wr_intr,
    output logic axi_dma_rd_intr
);

    // axi_dma_config
    addr_t        axi_read_start_addr;
    addr_t        axi_write_start_addr;
    logic  [31:0] axi_read_length;
    logic  [31:0] axi_write_length;
    logic         init_read;
    logic         init_write;
    logic         axi_write_start_ready;
    logic         axi_read_start_ready;
    logic         axi_dma_wr_idle;
    logic         axi_dma_rd_idle;

    // s_axilite_control (AXILITE Control)
    axilite #(
        .ADDR_WIDTH(8),
        .DATA_WIDTH(32)
    ) __axilite_intf ();
    `AXILITE_ASSIGN_SLAVE_TO_FLAT(control, __axilite_intf)
    // m_axis_mm2s (AXIS MM2S) Master Stream
    axis #(.DATA_WIDTH(DATA_WIDTH)) __axis_mm2s ();
    `AXIS_ASSIGN_MASTER_TO_FLAT(mm2s, __axis_mm2s)
    // s_axis_s2mm (AXIS S2MM) Slave Stream
    axis #(.DATA_WIDTH(DATA_WIDTH)) __axis_s2mm ();
    `AXIS_ASSIGN_SLAVE_TO_FLAT(s2mm, __axis_s2mm)
    // m_axi_dma (Full AXI)
    aximm #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) __aximm ();
    `AXI_ASSIGN_MASTER_TO_FLAT(dma, __aximm)


    axi_pkg::reg_map_64_t reg_map;
    axilite_ctrl #(
        .REG_NUM   (64),
        .DATA_WIDTH(32),
        .reg_map_t (axi_pkg::reg_map_64_t)
    ) u_axilite_ctrl (
        .axi_read_start_addr  (axi_read_start_addr),
        .axi_write_start_addr (axi_write_start_addr),
        .axi_read_length      (axi_read_length),
        .axi_write_length     (axi_write_length),
        .init_read            (init_read),
        .init_write           (init_write),
        .axi_write_start_ready(axi_write_start_ready),
        .axi_read_start_ready (axi_read_start_ready),
        .axi_dma_wr_idle      (axi_dma_wr_idle),
        .axi_dma_rd_idle      (axi_dma_rd_idle),
        .s_axi_aclk           (s_axi_aclk),
        .s_axi_aresetn        (s_axi_aresetn),
        .__axilite_intf       (__axilite_intf)
    );

    // axi_dma module
    axi_dma #(
        .NUM_RD_MO_BUF(NUM_RD_MO_BUF),
        .NUM_WR_MO_BUF(NUM_WR_MO_BUF),
        .ADDR_WIDTH   (ADDR_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .data_t       (data_t),
        .addr_t       (addr_t),
        .trans_t      (trans_t)
    ) u_axi_dma (
        .axi_read_start_addr  (axi_read_start_addr),
        .axi_write_start_addr (axi_write_start_addr),
        .axi_read_length      (axi_read_length),
        .axi_write_length     (axi_write_length),
        .init_read            (init_read),
        .init_write           (init_write),
        .axi_write_start_ready(axi_write_start_ready),
        .axi_read_start_ready (axi_read_start_ready),
        .axi_dma_wr_idle      (axi_dma_wr_idle),
        .axi_dma_rd_idle      (axi_dma_rd_idle),
        // AXIS (mm2s, s2mm)
        .__axis_mm2s          (__axis_mm2s),
        .__axis_s2mm          (__axis_s2mm),
        // AXI MM
        .__aximm              (__aximm),
        .m_axi_aclk           (m_axi_aclk),
        .m_axi_aresetn        (m_axi_aresetn)
    );

endmodule
