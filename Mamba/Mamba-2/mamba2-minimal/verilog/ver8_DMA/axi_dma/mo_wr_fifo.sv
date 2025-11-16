`timescale 1 ns / 1 ps

module mo_wr_fifo #(
    parameter int NUM_MO_BUF = 4,   // FIFO depth (number of outstanding transactions)
    parameter int ADDR_WIDTH = 64,  // 32 or 64
    parameter int DATA_WIDTH = 256,  // 256, 128, 64, ....
    // Default types; override as needed.
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    input  logic          clk,
    input  logic          rstn,
    // Input interface for a large transaction request:
    input  logic          start_valid,
    output logic          start_ready,
    input  addr_t         start_addr,
    input  logic   [31:0] len,
    // FIFO (output) interface for split transactions:
    output logic          mo_fifo_full,
    output logic          mo_fifo_empty,
    output trans_t        fifo_mo_aw,
    output logic          fifo_mo_aw_valid,
    input  logic          fifo_mo_aw_ready,
    output trans_t        fifo_mo_w,
    input  logic          fifo_mo_w_done,
    output trans_t        fifo_mo_b,
    input  logic          fifo_mo_b_done
);
    localparam int MO_FIFO_SIZE = NUM_MO_BUF + 1;
    localparam DATA_SHIFT = $clog2(DATA_WIDTH / 8);

    typedef enum logic [1:0] {
        IDLE,  // Waiting for a new (large) transaction.
        SPLIT  // Splitting the current transaction into 4K-boundary bursts.
    } state_t;
    state_t state;

    //-------------------------------------------------------------------------
    // FIFO memory and control signals.
    //-------------------------------------------------------------------------
    trans_t fifo_mem[MO_FIFO_SIZE];

    // Head/tail(b_ch_ptr) pointers and counter for circular FIFO.
    // head(push) tail(pop)
    logic [$clog2(MO_FIFO_SIZE)-1:0]
        head, aw_ch_ptr, w_ch_ptr, b_ch_ptr, next_head;

    assign next_head = (head + 1);
    assign mo_fifo_full = (next_head == b_ch_ptr) || (next_head == b_ch_ptr + MO_FIFO_SIZE);  // fifo full
    assign mo_fifo_empty = (head == b_ch_ptr);  // fifo empty
    assign start_ready = (state == IDLE) && (!mo_fifo_full);  // ready for new
    assign fifo_mo_aw_valid = (aw_ch_ptr != head);  //wr_ptr not reached head
    assign fifo_mo_aw = fifo_mem[aw_ch_ptr];
    assign fifo_mo_w = fifo_mem[w_ch_ptr];
    assign fifo_mo_b = fifo_mem[b_ch_ptr];

    //-------------------------------------------------------------------------
    // State machine for splitting input transaction.
    //-------------------------------------------------------------------------

    // Registers to hold the “current” large transaction while splitting.
    addr_t current_addr;
    logic [31:0] remaining;  // byte
    logic [31:0] new_remaining;

    logic [11:0] offset;  // byte
    logic [12:0] tx_size_bytes_raw;  // bytes
    logic [12:0] tx_size_bytes;  // bytes
    logic [8:0] tx_len;  // 32bytes for 256bit trans
    logic [12:0] boundary_remaining;

    // Determine how many bytes remain in the current 4K block.
    // The 4K boundary is 4096 bytes, so use the lower 12 bits.
    assign offset = current_addr[11:0];
    // Bytes remaining in the current 4K region:
    assign boundary_remaining = 4096 - offset;
    assign new_remaining = remaining - tx_size_bytes;
    assign tx_size_bytes_raw = (remaining < boundary_remaining) ? remaining: boundary_remaining;
    assign tx_size_bytes = ((tx_size_bytes_raw >> DATA_SHIFT) < axi_pkg::AXI_BURST_MAX) ? tx_size_bytes_raw : (axi_pkg::AXI_BURST_MAX << DATA_SHIFT);
    assign tx_len = tx_size_bytes >> DATA_SHIFT;

    // Main state machine and FIFO push/pop logic.
    always_ff @(posedge clk, negedge rstn) begin
        if (!rstn) begin
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    if (start_valid) begin
                        state <= SPLIT;
                    end
                end
                SPLIT: begin
                    // If FIFO is not ready, simply wait (stay in SPLIT).
                    if (remaining == 0) state <= IDLE;
                    else state <= SPLIT;
                end
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end


    always_ff @(posedge clk, negedge rstn) begin
        if (!rstn) begin
            current_addr <= 0;
            remaining    <= 0;
            head         <= 0;
            aw_ch_ptr    <= 0;
            w_ch_ptr     <= 0;
            b_ch_ptr     <= 0;
            for (int mo_idx = 0; mo_idx < MO_FIFO_SIZE; mo_idx++) begin
                fifo_mem[mo_idx] <= '{0, 0};
            end
        end else begin
            //--------------------------------------------------------------------------
            // FIFO Pop Logic:
            // If the FIFO output is accepted (fifo_valid) while ready.
            if (fifo_mo_aw_valid && fifo_mo_aw_ready) begin
                aw_ch_ptr <= (aw_ch_ptr == MO_FIFO_SIZE - 1) ? 0 : aw_ch_ptr + 1;
            end
            // pop the write ch mo
            if (fifo_mo_w_done) begin
                fifo_mem[w_ch_ptr] <= '{0, 0};
                w_ch_ptr <= (w_ch_ptr == MO_FIFO_SIZE - 1) ? 0 : w_ch_ptr + 1;
            end
            // pop the write ch mo
            if (fifo_mo_b_done) begin
                b_ch_ptr <= (b_ch_ptr == MO_FIFO_SIZE - 1) ? 0 : b_ch_ptr + 1;
            end

            //--------------------------------------------------------------------------
            // Splitting the Input Transaction.
            case (state)
                IDLE: begin
                    if (start_valid) begin
                        current_addr <= start_addr;
                        remaining    <= len;
                    end
                end

                SPLIT: begin
                    // Only generate (push) a split transaction if:
                    //   1. There are still bytes to split (remaining > 0), and
                    //   2. The FIFO has room (fifo full is not asserted).
                    if ((remaining > 0) && !mo_fifo_full) begin
                        // Enqueue the split transaction into the FIFO.
                        // Here we assume that the trans_t is a structure with
                        // two fields: start_addr and len.
                        fifo_mem[head] <= '{current_addr, tx_len};
                        head <= (head == MO_FIFO_SIZE - 1) ? 0 : head + 1;

                        // Compute updated values.
                        current_addr <= current_addr + tx_size_bytes;
                        remaining <= new_remaining;
                    end
                end
            endcase
        end
    end
endmodule
