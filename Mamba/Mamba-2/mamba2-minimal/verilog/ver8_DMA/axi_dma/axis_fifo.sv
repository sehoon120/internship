`timescale 1 ns / 1 ps 

module axis_fifo #(
    parameter DATA_WIDTH = 256,
    parameter DEPTH = 16,
    // Programmable thresholds for nearly full/empty conditions
    parameter ALMOST_FULL_THRESH = 4,
    // Default type; override as needed.
    type data_t = axi_pkg::data_256_t
) (
    input logic clk,
    input logic rstn,
    axis_fifo_intf.slave __fifo_intf
);

    // Xilinx Fifo Instance
    xpm_fifo_sync #(
        .CASCADE_HEIGHT(0),  // DECIMAL
        .DOUT_RESET_VALUE("0"),  // String
        .ECC_MODE("no_ecc"),  // String
        .FIFO_MEMORY_TYPE("auto"),  // String
        .FIFO_READ_LATENCY(0),  // DECIMAL
        .FIFO_WRITE_DEPTH(DEPTH),  // DECIMAL
        .FULL_RESET_VALUE(0),  // DECIMAL
        .PROG_EMPTY_THRESH(10),  // DECIMAL
        .PROG_FULL_THRESH(10),  // DECIMAL
        .RD_DATA_COUNT_WIDTH(1),  // DECIMAL
        .READ_DATA_WIDTH(DATA_WIDTH),  // DECIMAL
        .READ_MODE("fwft"),  // String
        .SIM_ASSERT_CHK     (0),         // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
        .USE_ADV_FEATURES("0707"),  // String
        .WAKEUP_TIME(0),  // DECIMAL
        .WRITE_DATA_WIDTH(DATA_WIDTH),  // DECIMAL
        .WR_DATA_COUNT_WIDTH(1)  // DECIMAL
    ) xpm_fifo_sync_inst (
        .almost_empty(),
        .almost_full(),
        .data_valid(data_valid),
        .dbiterr(),
        .dout(__fifo_intf.dout),
        .empty(__fifo_intf.empty),
        .full(__fifo_intf.full),
        .overflow(),
        .prog_empty(),
        .prog_full(__fifo_intf.prog_full),
        .rd_data_count(),
        .rd_rst_busy(),
        .sbiterr(),
        .underflow(),
        .wr_ack(),
        .wr_data_count(),
        .wr_rst_busy(),
        .din(__fifo_intf.din),
        .injectdbiterr(),
        .injectsbiterr(),
        .rd_en(__fifo_intf.rd_en),
        .rst(!rstn),
        .sleep(),
        .wr_clk(clk),
        .wr_en(__fifo_intf.wr_en)
    );



    // // Local parameter for pointer width.
    // localparam PTR_WIDTH = $clog2(DEPTH);

    // // FIFO memory array.
    // data_t mem[DEPTH-1:0];

    // // FIFO counter tracking the number of stored elements.
    // // Use one extra bit to cover the full count.
    // logic [PTR_WIDTH:0] count;

    // // Write and read pointers.
    // logic [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
    // logic [PTR_WIDTH-1:0] wr_ptr_next, rd_ptr_next;

    // //--------------------------------------------------------------------------
    // // FIFO Count Logic
    // //--------------------------------------------------------------------------
    // always_ff @(posedge clk or negedge rstn) begin
    //     if (!rstn) begin
    //         count <= 0;
    //     end else begin
    //         // When both a write and a read occur simultaneously, the count remains unchanged.
    //         if (__fifo_intf.wr_en && !__fifo_intf.full &&
    //             __fifo_intf.rd_en && !__fifo_intf.empty) begin
    //             count <= count;
    //         end else if (__fifo_intf.wr_en && !__fifo_intf.full) begin
    //             count <= count + 1;
    //         end else if (__fifo_intf.rd_en && !__fifo_intf.empty) begin
    //             count <= count - 1;
    //         end else begin
    //             count <= count;
    //         end
    //     end
    // end

    // //--------------------------------------------------------------------------
    // // Write Pointer Next Calculation (Combinational)
    // //--------------------------------------------------------------------------
    // always_comb begin
    //     if (wr_ptr == DEPTH - 1) wr_ptr_next = 0;
    //     else wr_ptr_next = wr_ptr + 1;
    // end

    // //--------------------------------------------------------------------------
    // // Read Pointer Next Calculation (Combinational)
    // //--------------------------------------------------------------------------
    // always_comb begin
    //     if (rd_ptr == DEPTH - 1) rd_ptr_next = 0;
    //     else rd_ptr_next = rd_ptr + 1;
    // end

    // //--------------------------------------------------------------------------
    // // Write Pointer and Memory Write Logic
    // //--------------------------------------------------------------------------
    // integer i;
    // always_ff @(posedge clk or negedge rstn) begin
    //     if (!rstn) begin
    //         wr_ptr <= 0;
    //         // Optionally initialize the FIFO memory.
    //         for (i = 0; i < DEPTH; i = i + 1) begin
    //             mem[i] <= '0;
    //         end
    //     end else begin
    //         if (__fifo_intf.wr_en && !__fifo_intf.full) begin
    //             mem[wr_ptr] <= __fifo_intf.din;
    //             wr_ptr <= wr_ptr_next;
    //         end
    //     end
    // end

    // //--------------------------------------------------------------------------
    // // Read Pointer Update Logic
    // //--------------------------------------------------------------------------
    // always_ff @(posedge clk or negedge rstn) begin
    //     if (!rstn) begin
    //         rd_ptr <= 0;
    //     end else begin
    //         if (__fifo_intf.rd_en && !__fifo_intf.empty) begin
    //             rd_ptr <= rd_ptr_next;
    //         end
    //     end
    // end

    // //--------------------------------------------------------------------------
    // // FIFO Interface Signal Assignments
    // //--------------------------------------------------------------------------
    // assign __fifo_intf.full      = (count == DEPTH);
    // assign __fifo_intf.empty     = (count == 0);
    // assign __fifo_intf.prog_full = (count >= (DEPTH - ALMOST_FULL_THRESH));
    // assign __fifo_intf.dout      = mem[rd_ptr];
endmodule
