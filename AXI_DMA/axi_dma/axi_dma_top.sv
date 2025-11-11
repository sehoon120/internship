`timescale 1 ns / 1 ps

module axi_dma_top #(
    localparam ADDR_WIDTH = 64,
    localparam DATA_WIDTH = 256,
    localparam WR_AXIS_FIFO_DEPTH = 1024
) (
    input logic clk_in1_p,  // in std_logic
    input logic clk_in1_n,  // in std_logic
    input logic reset,
    input logic start,
    output logic done_drive,
    output logic axi_dma_status
);
    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [DATA_WIDTH-1:0] data_t;

    logic clk;
    logic rstn;
    assign rstn = !reset;

    //-- Write Interface Signals (for DMA)
    addr_t        axi_write_start_addr;
    logic  [31:0] axi_write_length;
    logic         init_write;

    //-- Read Interface Signals (for DMA)
    addr_t        axi_read_start_addr;
    logic  [31:0] axi_read_length;
    logic         init_read;

    // Ready signals from the DMA (to be driven by the DUT).
    logic         axi_write_start_ready;
    logic         axi_read_start_ready;

    // Idle signals from the DMA (to be driven by the DUT).
    logic         axi_dma_wr_idle;
    logic         axi_dma_rd_idle;
    assign axi_dma_status = axi_dma_wr_idle & axi_dma_rd_idle;


    axis #(.DATA_WIDTH(DATA_WIDTH)) __axis_mm2s ();
    axis #(.DATA_WIDTH(DATA_WIDTH)) __axis_s2mm ();
    aximm #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) __aximm ();

    clock_gen u_clock_gen (
        .clk_in1_p(clk_in1_p),
        .clk_in1_n(clk_in1_n),
        .reset(reset),
        .locked(),
        .clock(clk)
    );

    axi_dma_reg_driver u_axi_dma_reg_driver (
        .clk(clk),
        .rstn(rstn),
        .axi_read_start_addr(axi_read_start_addr),
        .axi_write_start_addr(axi_write_start_addr),
        .axi_read_length(axi_read_length),
        .axi_write_length(axi_write_length),
        .init_read(init_read),
        .init_write(init_write)
    );

    axis_mm2s_receiver #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_mm2s_receiver (
        .clk(clk),
        .rstn(rstn),
        .start(start),
        .__axis_mm2s(__axis_mm2s)
    );

    axis_s2mm_driver #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_s2mm_driver (
        .clk(clk),
        .rstn(rstn),
        .start(start),
        .done_drive(done_drive),
        .__axis_s2mm(__axis_s2mm)
    );

    axi_dma #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .WR_AXIS_FIFO_DEPTH(1024)
    ) axi_dma_256bit (
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
        .__axis_mm2s          (__axis_mm2s),
        .__axis_s2mm          (__axis_s2mm),
        .__aximm              (__aximm),
        .m_axi_aclk           (clk),
        .m_axi_aresetn        (rstn)
    );

    axi_bram_ctrl_0 memory_init (
        .s_axi_aclk(clk),
        .s_axi_aresetn(rstn),
        .s_axi_awaddr(__aximm.awaddr[17:0]),
        .s_axi_awlen(__aximm.awlen),
        .s_axi_awsize(__aximm.awsize),
        .s_axi_awburst(__aximm.awburst),
        .s_axi_awvalid(__aximm.awvalid),
        .s_axi_awready(__aximm.awready),
        .s_axi_wdata(__aximm.wdata),
        .s_axi_wstrb(__aximm.wstrb),
        .s_axi_wlast(__aximm.wlast),
        .s_axi_wvalid(__aximm.wvalid),
        .s_axi_wready(__aximm.wready),
        .s_axi_bresp(__aximm.bresp),
        .s_axi_bvalid(__aximm.bvalid),
        .s_axi_bready(__aximm.bready),
        .s_axi_araddr(__aximm.araddr[17:0]),
        .s_axi_arlen(__aximm.arlen),
        .s_axi_arsize(__aximm.arsize),
        .s_axi_arburst(__aximm.arburst),
        .s_axi_arvalid(__aximm.arvalid),
        .s_axi_arready(__aximm.arready),
        .s_axi_rdata(__aximm.rdata),
        .s_axi_rresp(__aximm.rresp),
        .s_axi_rlast(__aximm.rlast),
        .s_axi_rvalid(__aximm.rvalid),
        .s_axi_rready(__aximm.rready)
    );
endmodule
