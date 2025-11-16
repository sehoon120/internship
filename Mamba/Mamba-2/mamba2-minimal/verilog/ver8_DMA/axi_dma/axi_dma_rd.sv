`timescale 1 ns / 1 ps

module axi_dma_rd #(
    parameter NUM_MO_BUF = 4,
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 256,
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    // AXI config, Start AXI transactions (read / write)
    input  wire                   init_read,
    input  wire  [ADDR_WIDTH-1:0] axi_start_addr,
    input  wire  [          31:0] axi_byte_length,
    output logic                  axi_start_ready,
    output logic                  axi_start_valid,
    output logic                  axi_idle,

    // AXIS Interface
    axis.master __axis_mm2s,  // accelerator

    // AXI Interface
    aximm_rd.master __aximm_rd,  // memory
    input wire clk,
    input wire rstn
);

    logic rnext;
    axi_pkg::resp_t rresp;
    logic [11:0] read_counter;
    logic read_resp_error;

    // logic axi_start_valid;
    trans_t fifo_mo_ar, fifo_mo_r;
    logic mo_fifo_full;
    logic mo_fifo_empty;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            axi_start_valid <= 0;
        end else begin
            if (axi_start_ready) begin
                if (axi_start_valid) begin
                    axi_start_valid <= 0;
                end else if (init_read) begin
                    axi_start_valid <= 1;
                end
            end
        end
    end
    assign axi_idle = axi_start_ready && mo_fifo_empty;

    //----------------------------
    //Read Address Channel
    //----------------------------
    assign __aximm_rd.araddr = fifo_mo_ar.addr;
    assign __aximm_rd.arlen = fifo_mo_ar.len - 1;
    assign __aximm_rd.arsize = $clog2((DATA_WIDTH / 8) - 1);
    assign __aximm_rd.arburst = 2'b01;

    //--------------------------------
    //Read Data (and Response) Channel
    //--------------------------------
    //--------------------------------
    assign __axis_mm2s.tdata = __aximm_rd.rdata;
    assign __axis_mm2s.tvalid = __aximm_rd.rvalid;
    assign __axis_mm2s.tlast = __aximm_rd.rlast;
    assign __aximm_rd.rready = __axis_mm2s.tready;
    //--------------------------------
    assign rresp = __aximm_rd.rresp;
    assign rnext = __aximm_rd.rvalid && __axis_mm2s.tready;
    assign read_resp_error = __axis_mm2s.tready & __aximm_rd.rvalid & rresp[1];

    // Burst length counter. Uses extra counter register bit to indicate
    // terminal count to reduce decode logic
    always @(posedge clk) begin
        if (!rstn) begin
            read_counter <= 0;
        end else begin
            if (rnext) begin
                if (read_counter == fifo_mo_r.len - 1) begin
                    read_counter <= 0;
                end else begin
                    read_counter <= read_counter + 1;
                end
            end
        end
    end

    mo_rd_fifo #(
        .NUM_MO_BUF(NUM_MO_BUF),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .addr_t(addr_t),
        .trans_t(trans_t)
    ) u_mo_rd_fifo (
        .clk             (clk),
        .rstn            (rstn),
        .start_valid     (axi_start_valid),
        .start_ready     (axi_start_ready),
        .start_addr      (axi_start_addr),
        .len             (axi_byte_length),
        // mo_fifo state
        .mo_fifo_full    (mo_fifo_full),
        .mo_fifo_empty   (mo_fifo_empty),
        // ar channel
        .fifo_mo_ar      (fifo_mo_ar),
        .fifo_mo_ar_valid(__aximm_rd.arvalid),
        .fifo_mo_ar_ready(__aximm_rd.arready),
        // r channel
        .fifo_mo_r       (fifo_mo_r),
        .fifo_mo_r_done  (__aximm_rd.rlast)
    );
endmodule
