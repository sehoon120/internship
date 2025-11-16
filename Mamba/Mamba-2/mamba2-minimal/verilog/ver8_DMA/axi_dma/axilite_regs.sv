`timescale 1 ns / 1 ps

module axilite_regs #(
    parameter integer DATA_WIDTH = 32,
    parameter integer REG_NUM = 64,
    parameter integer RW_REG_NUM = 48,
    parameter integer RO_REG_NUM = 16,
    parameter integer ADDR_WIDTH = 8
) (
    input wire s_axi_aclk,
    input wire s_axi_aresetn,
    output logic [DATA_WIDTH-1:0] RW_axilite_regs[RW_REG_NUM],
    input wire [DATA_WIDTH-1:0] RO_axilite_regs[RO_REG_NUM],
    axilite.slave __s_axilite
);

    logic [DATA_WIDTH-1:0] axilite_regs[REG_NUM];

    // AXI4LITE signals
    logic [ADDR_WIDTH-1 : 0] axi_awaddr;
    logic axi_awready;
    logic axi_wready;
    logic [1 : 0] axi_bresp;
    logic axi_bvalid;
    logic [ADDR_WIDTH-1 : 0] axi_araddr;
    logic axi_arready;
    logic [DATA_WIDTH-1 : 0] axi_rdata;
    logic [1 : 0] axi_rresp;
    logic axi_rvalid;

    localparam integer ADDR_LSB = (DATA_WIDTH / 32) + 1;
    localparam integer OPT_MEM_ADDR_BITS = 5;

    wire                     slv_reg_rden;
    wire                     slv_reg_wren;
    logic   [DATA_WIDTH-1:0] reg_data_out;
    integer                  byte_index;
    logic                    aw_en;

    // I/O Connections assignments
    assign __s_axilite.awready = axi_awready;
    assign __s_axilite.wready  = axi_wready;
    assign __s_axilite.bresp   = axi_bresp;
    assign __s_axilite.bvalid  = axi_bvalid;
    assign __s_axilite.arready = axi_arready;
    assign __s_axilite.rdata   = axi_rdata;
    assign __s_axilite.rresp   = axi_rresp;
    assign __s_axilite.rvalid  = axi_rvalid;

    always_ff @(posedge s_axi_aclk) begin  // awready, awen
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end else begin
            if (~axi_awready && __s_axilite.awvalid && __s_axilite.wvalid && aw_en) begin
                axi_awready <= 1'b1;
                aw_en <= 1'b0;
            end else if (__s_axilite.bready && axi_bvalid) begin
                aw_en <= 1'b1;
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end


    always_ff @(posedge s_axi_aclk) begin  // awaddr
        if (!s_axi_aresetn) begin
            axi_awaddr <= 0;
        end else begin
            if (~axi_awready && __s_axilite.awvalid && __s_axilite.wvalid && aw_en) begin
                // Write Address latching
                axi_awaddr <= __s_axilite.awaddr;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin  // wready
        if (!s_axi_aresetn) begin
            axi_wready <= 1'b0;
        end else begin
            if (~axi_wready && __s_axilite.wvalid && __s_axilite.awvalid && aw_en) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end

    assign slv_reg_wren = axi_wready && __s_axilite.wvalid && axi_awready && __s_axilite.awvalid;

    always_ff @(posedge s_axi_aclk) begin  // write to reg
        if (s_axi_aresetn == 1'b0) begin  // reset axilite_regs
            for (int reg_idx = 0; reg_idx < RW_REG_NUM; reg_idx++) begin
                RW_axilite_regs[reg_idx] <= 0;
            end
        end else begin
            if (slv_reg_wren) begin
                for (
                    int byte_idx = 0;
                    byte_idx <= (DATA_WIDTH / 8) - 1;
                    byte_idx++
                ) begin
                    if (__s_axilite.wstrb[byte_idx] == 1) begin
                        RW_axilite_regs[axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB]][(byte_idx*8) +:8] <=   __s_axilite.wdata[(byte_idx*8) +:8] ;
                    end
                end
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_bvalid <= 0;
            axi_bresp  <= 2'b0;
        end else begin
            if (axi_awready && __s_axilite.awvalid && ~axi_bvalid && axi_wready && __s_axilite.wvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b0;  // 'OKAY' response
            end                   // work error responses in future
            else begin
                if (__s_axilite.bready && axi_bvalid) begin
                    axi_bvalid <= 1'b0;
                end
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 32'b0;
        end else begin
            if (~axi_arready && __s_axilite.arvalid) begin
                // indicates that the slave has acceped the valid read address
                axi_arready <= 1'b1;
                // Read address latching
                axi_araddr  <= __s_axilite.araddr;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_rvalid <= 0;
            axi_rresp  <= 0;
        end else begin
            if (axi_arready && __s_axilite.arvalid && ~axi_rvalid) begin
                // Valid read data is available at the read data bus
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b0;  // 'OKAY' response
            end else if (axi_rvalid && __s_axilite.rready) begin
                // Read data is accepted by the master
                axi_rvalid <= 1'b0;
            end
        end
    end

    assign slv_reg_rden = axi_arready & __s_axilite.arvalid & ~axi_rvalid;
    always @(*) begin
        if (axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] < RW_REG_NUM) begin
            reg_data_out = RW_axilite_regs[axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB]];
        end else begin
            reg_data_out = RO_axilite_regs[(axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB] - RW_REG_NUM)];
        end
    end

    // Output register or memory read data
    always @(posedge s_axi_aclk) begin
        if (s_axi_aresetn == 1'b0) begin
            axi_rdata <= 0;
        end else begin
            if (slv_reg_rden) begin
                axi_rdata <= reg_data_out;  // register read data
            end
        end
    end
endmodule
