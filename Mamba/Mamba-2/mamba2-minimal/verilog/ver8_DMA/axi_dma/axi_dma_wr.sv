`timescale 1 ns / 1 ps

module axi_dma_wr #(
    parameter NUM_MO_BUF = 4,
    parameter integer ADDR_WIDTH = 64,
    parameter integer DATA_WIDTH = 256,
    parameter WR_FIFO_DEPTH = 8,
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    // AXI config, Start AXI transactions (write)
    input  logic         init_write,
    input  addr_t        axi_start_addr,
    input  logic  [31:0] axi_byte_length,
    output logic         axi_start_ready,
    output logic         axi_start_valid,
    output logic         axi_idle,

    axis.slave __axis_s2mm,

    input wire clk,
    input wire rstn,
    aximm_wr.master __aximm_wr
);

    axis_fifo_intf #(.DATA_WIDTH(DATA_WIDTH)) __axis_fifo_intf ();

    // w channel
    logic wready;
    logic wvalid;
    logic wlast;
    logic wnext;
    logic [11:0] write_counter;
    // resp channel
    axi_pkg::resp_t bresp;
    logic bvalid;
    logic bready;
    logic write_resp_error;

    // logic axi_start_valid;

    trans_t fifo_mo_aw, fifo_mo_w, fifo_mo_b;
    logic fifo_mo_w_done, fifo_mo_b_done;
    logic mo_fifo_full;
    logic mo_fifo_empty;


    always_ff @(posedge clk) begin
        if (rstn == 1'b0) begin
            axi_start_valid <= 0;
        end else begin
            if (axi_start_ready) begin
                if (axi_start_valid) begin
                    axi_start_valid <= 0;
                end else if (init_write) begin
                    axi_start_valid <= 1;
                end
            end
        end
    end
    assign axi_idle = axi_start_ready && mo_fifo_empty;


    //--------------------------------
    //Write Address Channel
    //--------------------------------
    assign __aximm_wr.awaddr = fifo_mo_aw.addr;
    assign __aximm_wr.awlen = fifo_mo_aw.len - 1;  //axi_burst_length - 1;
    assign __aximm_wr.awsize = $clog2((DATA_WIDTH / 8) - 1);
    assign __aximm_wr.awburst = 2'b01;

    //--------------------------------
    // AXIS FIFO intf
    //--------------------------------
    assign __axis_fifo_intf.din = __axis_s2mm.tdata;
    assign __axis_fifo_intf.wr_en = __axis_s2mm.tvalid;
    assign __axis_fifo_intf.rd_en = wnext;
    assign __axis_s2mm.tready = !__axis_fifo_intf.full;

    //--------------------------------
    //Write Data Channel
    //--------------------------------
    assign __aximm_wr.wdata = __axis_fifo_intf.dout;
    assign __aximm_wr.wvalid = wvalid && !__axis_fifo_intf.empty;
    assign wready = __aximm_wr.wready;
    assign __aximm_wr.wlast = wlast;
    assign __aximm_wr.wstrb = {(DATA_WIDTH / 8) {1'b1}};
    assign wnext = __aximm_wr.wvalid & __aximm_wr.wready;
    assign fifo_mo_w_done = wnext && wlast;

    always @(posedge clk) begin
        if (!rstn) begin
            write_counter <= 0;
            wvalid <= 0;
            wlast <= 0;
        end else begin
            // data counter
            if (wnext) begin
                if (write_counter == fifo_mo_w.len - 1) begin
                    write_counter <= 0;
                end else begin
                    write_counter <= write_counter + 1;
                end
            end

            // wvalid logic
            if (wnext && wlast) begin
                wvalid <= 0;
            end else if (fifo_mo_w.len > 0) begin
                if (write_counter < fifo_mo_w.len) begin
                    wvalid <= 1;
                end
            end

            // wlast
            if (wnext) begin
                if (fifo_mo_w.len == 1) begin
                    if (write_counter == 0) begin
                        wlast <= 1'b0;
                    end
                end else begin
                    if (write_counter == fifo_mo_w.len - 2) begin
                        wlast <= 1'b1;
                    end else if (wlast) begin
                        wlast <= 1'b0;
                    end
                end
            end else begin
                if (fifo_mo_w.len == 1) begin
                    if (write_counter == 0) begin
                        wlast <= 1;
                    end
                end
            end

        end
    end

    //----------------------------
    //Write Response (B) Channel
    //----------------------------
    assign bresp = __aximm_wr.bresp;
    assign bvalid = __aximm_wr.bvalid;
    assign __aximm_wr.bready = bready;

    assign fifo_mo_b_done = bready & bvalid;
    assign write_resp_error = bready & bvalid & bresp[1];
    always @(posedge clk) begin
        if (rstn == 0) begin
            bready <= 1'b0;
        end else begin
            if (bvalid && ~bready) begin
                bready <= 1;
            end else if (bready) begin  // deassert after one clock cycle
                bready <= 0;
            end else begin
                bready <= bready;
            end
        end
    end

    axis_fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH     (WR_FIFO_DEPTH),
        .data_t    (data_t)
    ) u_axis_fifo (
        .clk        (clk),
        .rstn       (rstn),
        .__fifo_intf(__axis_fifo_intf)
    );

    mo_wr_fifo #(
        .NUM_MO_BUF(NUM_MO_BUF),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .addr_t    (addr_t),
        .trans_t   (trans_t)
    ) u_mo_wr_fifo (
        .clk             (clk),
        .rstn            (rstn),
        .start_valid     (axi_start_valid),
        .start_ready     (axi_start_ready),
        .start_addr      (axi_start_addr),
        .len             (axi_byte_length),
        // mo_fifo state
        .mo_fifo_full    (mo_fifo_full),
        .mo_fifo_empty   (mo_fifo_empty),
        // aw channel
        .fifo_mo_aw      (fifo_mo_aw),
        .fifo_mo_aw_valid(__aximm_wr.awvalid),
        .fifo_mo_aw_ready(__aximm_wr.awready),
        // w channel
        .fifo_mo_w       (fifo_mo_w),
        .fifo_mo_w_done  (fifo_mo_w_done),
        // b channel
        .fifo_mo_b       (fifo_mo_b),
        .fifo_mo_b_done  (fifo_mo_b_done)
    );
endmodule
