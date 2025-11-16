`timescale 1 ns / 1 ps

module axilite_ctrl #(
    parameter integer REG_NUM = 64,
    parameter integer DATA_WIDTH = 32,
    parameter integer ADDR_WIDTH = $clog2(REG_NUM),
    type reg_map_t = axi_pkg::reg_map_64_t
) (
    output logic [63:0] axi_read_start_addr,
    output logic [63:0] axi_write_start_addr,
    output logic [31:0] axi_read_length,
    output logic [31:0] axi_write_length,
    output logic init_read,
    output logic init_write,
    input logic axi_write_start_ready,
    input logic axi_read_start_ready,
    input logic axi_dma_wr_idle,
    input logic axi_dma_rd_idle,

    input logic s_axi_aclk,
    input logic s_axi_aresetn,
    axilite.slave __axilite_intf
);

    //-------------------------------------------------------------------------
    // Register Map Structure Definition
    //-------------------------------------------------------------------------
    //   - Register 0 : control                  (RW, 32-bit)
    //   - Register 1 : axi_read_start_addr_lsb  (RW, 32-bit)
    //   - Register 2 : axi_read_start_addr_msb  (RW, 32-bit)
    //   - Register 3 : axi_write_start_addr_lsb (RW, 32-bit)
    //   - Register 4 : axi_write_start_addr_msb (RW, 32-bit)
    //   - Register 5 : axi_read_length          (RW, 32-bit)
    //   - Register 6 : axi_write_length         (RW, 32-bit)
    //   - Register 7 : init_read                (RW, 32-bit)
    //   - Register 8 : init_write               (RW, 32-bit)
    //   - Register 9 : axi_write_start_ready    (RO, 32-bit)
    //   - Register 10: axi_read_start_ready     (RO, 32-bit)
    //   - Register 11: axi_dma_wr_idle          (RO, 32-bit)
    //   - Register 12: axi_dma_rd_idle          (RO, 32-bit)
    typedef enum logic [5:0] {
        CONTROL                  = 0,   // Register 0 (RW)
        AXI_READ_START_ADDR_LSB  = 1,   // Register 1 (RW)
        AXI_READ_START_ADDR_MSB  = 2,   // Register 2 (RW)
        AXI_WRITE_START_ADDR_LSB = 3,   // Register 3 (RW)
        AXI_WRITE_START_ADDR_MSB = 4,   // Register 4 (RW)
        AXI_READ_LENGTH          = 5,   // Register 5 (RW)
        AXI_WRITE_LENGTH         = 6,   // Register 6 (RW)
        INIT_READ                = 7,   // Register 7 (RW)
        INIT_WRITE               = 8,   // Register 8 (RW)
        AXI_WRITE_START_READY    = 9,
        AXI_READ_START_READY     = 10,
        AXI_DMA_WR_IDLE          = 11,
        AXI_DMA_RD_IDLE          = 12
    } reg_map_e;

    //-------------------------------------------------------------------------
    // Local Parameters for Address Decoding
    //-------------------------------------------------------------------------
    localparam integer ADDR_LSB = 2;  // For word alignment (32-bit)
    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8; // Number of byte lanes

    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [STRB_WIDTH-1:0] strb_t;

    addr_t awaddr;
    logic awvalid;
    logic awready;
    data_t wdata;
    strb_t wstrb;
    logic wvalid;
    logic wready;
    axi_pkg::resp_t bresp;
    logic bvalid;
    logic bready;
    addr_t araddr;
    logic arvalid;
    logic arready;
    data_t rdata;
    axi_pkg::resp_t rresp;
    logic rvalid;
    logic rready;

    reg_map_t reg_map;
    data_t wmask;
    logic [ADDR_WIDTH-ADDR_LSB:0] awaddr_reg_idx;

    // AW channel
    assign awaddr = __axilite_intf.awaddr;
    assign awvalid = __axilite_intf.awvalid;
    assign __axilite_intf.awready = awready;
    // W channel
    assign wdata = __axilite_intf.wdata;
    assign wstrb = __axilite_intf.wstrb;
    assign wmask = {{8{wstrb[3]}}, {8{wstrb[2]}}, {8{wstrb[1]}}, {8{wstrb[0]}}};
    assign wvalid = __axilite_intf.wvalid;
    assign __axilite_intf.wready = wready;
    // B channel
    assign bready = __axilite_intf.bready;
    assign __axilite_intf.bresp = bresp;
    assign __axilite_intf.bvalid = bvalid;
    // AR channel
    assign araddr = __axilite_intf.araddr;
    assign arvalid = __axilite_intf.arvalid;
    assign __axilite_intf.arready = arready;
    // R channel
    assign rready = __axilite_intf.rready;
    assign __axilite_intf.rdata = rdata;
    assign __axilite_intf.rresp = rresp;
    assign __axilite_intf.rvalid = rvalid;

    //-------------------------------------------------------------------------
    // AXI-Lite Write Logic
    //-------------------------------------------------------------------------
    // AW channel
    addr_t awaddr_reg;
    assign awaddr_reg_idx = (awaddr_reg >> ADDR_LSB);
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            awready    <= 1'b0;
            awaddr_reg <= '0;
            wready     <= 1'b0;
        end else begin
            // Latch AWADDR when valid and not already processing a write.
            if (!awready && awvalid) begin
                awready    <= 1'b1;
                awaddr_reg <= awaddr;
                wready     <= 1'b1;
            end else begin
                awready <= 1'b0;
                wready  <= 1'b0;
            end
        end
    end
    // W channel
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            for (int idx = 0; idx < 64; idx++) begin
                reg_map[idx] <= '0;
            end
        end else begin
            //--------------------------------------------------------
            //--------------------------------------------------------
            // When W handshakes occur, update the register map.
            if (wready && wvalid) begin
                case (awaddr_reg_idx)
                    CONTROL: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_READ_START_ADDR_LSB: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_READ_START_ADDR_MSB: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_WRITE_START_ADDR_LSB: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_WRITE_START_ADDR_MSB: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_READ_LENGTH: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_WRITE_LENGTH: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    INIT_READ: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    INIT_WRITE: begin
                        reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_WRITE_START_READY: begin
                        // reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_READ_START_READY: begin
                        // reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_DMA_WR_IDLE: begin
                        // reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    AXI_DMA_RD_IDLE: begin
                        // reg_map[awaddr_reg_idx] <= (wdata & wmask) | (reg_map[awaddr_reg_idx] & ~wmask);
                    end
                    default: begin
                        reg_map[awaddr_reg_idx] <= reg_map[awaddr_reg_idx];
                    end
                endcase
            end else begin
                // reset.
                reg_map[INIT_WRITE] <= '0;
                reg_map[INIT_READ]  <= '0;
            end
            // RO registers 
            reg_map[AXI_WRITE_START_READY] <= {30'b0, axi_write_start_ready};
            reg_map[AXI_READ_START_READY]  <= {30'b0, axi_read_start_ready};
            reg_map[AXI_DMA_WR_IDLE]       <= {30'b0, axi_dma_wr_idle};
            reg_map[AXI_DMA_RD_IDLE]       <= {30'b0, axi_dma_rd_idle};
            //--------------------------------------------------------
            //--------------------------------------------------------
        end
    end
    // B channel
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            bvalid <= 1'b0;
            bresp  <= 2'b0;
        end else begin
            if (wready && wvalid) begin
                bvalid <= 1'b1;
                bresp  <= axi_pkg::RESP_OKAY;  // OKAY response.
            end

            // Write response handshake: deassert BVALID when accepted.
            if (bready && bvalid) begin
                bvalid <= 1'b0;
            end
        end
    end

    //-------------------------------------------------------------------------
    // AXI-Lite Read Logic
    //-------------------------------------------------------------------------
    addr_t araddr_reg;
    logic  read_in_progress;
    // AR channel
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            arready          <= 1'b0;
            araddr_reg       <= '0;
            read_in_progress <= 1'b0;
        end else begin
            // Latch ARADDR when valid.
            if (!arready && arvalid && !read_in_progress) begin
                arready          <= 1'b1;
                araddr_reg       <= araddr;
                read_in_progress <= 1'b1;
            end else begin
                if (arready && arvalid) begin
                    arready <= 1'b0;
                end

                if (rready && rvalid) begin
                    read_in_progress <= 1'b0;
                end
            end
        end
    end
    // R channel
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            rdata  <= '0;
            rresp  <= 2'b0;
            rvalid <= 1'b0;
        end else begin
            // When a read is pending and data is not already valid, output the data.
            if (read_in_progress && !rvalid) begin
                rdata  <= reg_map[(araddr_reg>>ADDR_LSB)];
                rresp  <= axi_pkg::RESP_OKAY;
                rvalid <= 1'b1;
            end

            // Read handshake: if the master accepts the data, clear RVALID.
            if (rvalid && rready) begin
                rvalid <= 1'b0;
            end
        end
    end

    //--------------------------------------------------
    //--------------------------------------------------
    // assign axi_read_start_addr = {
    //     reg_map.AXI_READ_START_ADDR_MSB, reg_map.AXI_READ_START_ADDR_LSB
    // };
    // assign axi_write_start_addr = {
    //     reg_map.AXI_WRITE_START_ADDR_MSB, reg_map.AXI_WRITE_START_ADDR_LSB
    // };
    // assign axi_read_length = reg_map.AXI_READ_LENGTH;
    // assign axi_write_length = reg_map.AXI_WRITE_LENGTH;
    // assign init_read = reg_map.INIT_READ[0];
    // assign init_write = reg_map.INIT_WRITE[0];
    // assign reg_map.AXI_WRITE_START_READY[0] = axi_write_start_ready;
    // assign reg_map.AXI_READ_START_READY[0] = axi_read_start_ready;
    // assign reg_map.AXI_DMA_WR_IDLE[0] = axi_dma_wr_idle;
    // assign reg_map.AXI_DMA_RD_IDLE[0] = axi_dma_rd_idle;

    assign axi_read_start_addr = {
        reg_map[AXI_READ_START_ADDR_MSB], reg_map[AXI_READ_START_ADDR_LSB]
    };
    assign axi_write_start_addr = {
        reg_map[AXI_WRITE_START_ADDR_MSB], reg_map[AXI_WRITE_START_ADDR_LSB]
    };
    assign axi_read_length = reg_map[AXI_READ_LENGTH];
    assign axi_write_length = reg_map[AXI_WRITE_LENGTH];
    assign init_read = reg_map[INIT_READ][0];
    assign init_write = reg_map[INIT_WRITE][0];
    //--------------------------------------------------
    //--------------------------------------------------
endmodule
