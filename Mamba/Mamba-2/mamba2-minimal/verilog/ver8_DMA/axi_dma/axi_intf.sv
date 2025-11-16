//-------------------------------------------------------------------------
// AXIMM Interface
//-------------------------------------------------------------------------
interface aximm #(  // 외부 메모리(DDR/NoC)의 AXI4 Memory-Mapped 버스와 연결하는 “완전한” 인터페이스. 채널 5개(AW/W/B, AR/R) 모두 포함.
    // parameter int unsigned axi_id_width   = 0,
    // parameter int unsigned axi_user_width = 0,
    parameter int unsigned ADDR_WIDTH = 64,
    parameter int unsigned DATA_WIDTH = 256
);

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;

    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [STRB_WIDTH-1:0] strb_t;

    // AW port
    // axi_pkg::id_t awid;
    addr_t awaddr;
    axi_pkg::len_t awlen;
    axi_pkg::size_t awsize;
    axi_pkg::burst_t awburst;
    // axi_pkg::user_t awuser;
    logic awvalid;
    logic awready;

    // W port
    data_t wdata;
    strb_t wstrb;
    logic wlast;
    logic wvalid;
    logic wready;

    // B port
    // axi_id_t bid;
    axi_pkg::resp_t bresp;
    // axi_pkg::user_t buser;
    logic bvalid;
    logic bready;

    // AR port
    // axi_pkg::id_t arid;
    addr_t araddr;
    axi_pkg::len_t arlen;
    axi_pkg::size_t arsize;
    axi_pkg::burst_t arburst;
    // axi_pkg::user_t aruser;
    logic arvalid;
    logic arready;

    // R port
    // axi_id_t rid;
    data_t rdata;
    axi_pkg::resp_t rresp;
    logic rlast;
    logic rvalid;
    logic rready;

    modport master(  // 출력은 주소/데이터/valid, 입력은 ready/resp/data
        // AW port
        input awready,
        output awaddr, awlen, awsize, awburst, awvalid,
        // W port
        input wready,
        output wdata, wstrb, wlast, wvalid,
        // B port
        input bresp, bvalid,
        output bready,
        // AR port
        input arready,
        output araddr, arlen, arsize, arburst, arvalid,
        // R port
        input rdata, rresp, rlast, rvalid,
        output rready
    );

    modport slave(  // 반대 방향
        // AW port
        input awaddr, awlen, awsize, awburst, awvalid,
        output awready,
        // W port
        input wdata, wstrb, wlast, wvalid,
        output wready,
        // B port
        input bready,
        output bresp, bvalid,
        // AR port
        input araddr, arlen, arsize, arburst, arvalid,
        output arready,
        // R port
        input rready,
        output rdata, rresp, rlast, rvalid
    );
endinterface

//-------------------------------------------------------------------------
// AXIMM Read Interface
//-------------------------------------------------------------------------
interface aximm_rd #(
    parameter int unsigned ADDR_WIDTH = 64,
    parameter int unsigned DATA_WIDTH = 256
);

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;

    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [STRB_WIDTH-1:0] strb_t;

    // AR port
    // axi_pkg::id_t arid;
    addr_t araddr;
    axi_pkg::len_t arlen;
    axi_pkg::size_t arsize;
    axi_pkg::burst_t arburst;
    // axi_pkg::user_t aruser;
    logic arvalid;
    logic arready;

    // R port
    // axi_id_t rid;
    data_t rdata;
    axi_pkg::resp_t rresp;
    logic rlast;
    logic rvalid;
    logic rready;

    modport master(
        // AR port
        input arready,
        output araddr, arlen, arsize, arburst, arvalid,
        // R port
        input rdata, rresp, rlast, rvalid,
        output rready
    );

    modport slave(
        // AR port
        input araddr, arlen, arsize, arburst, arvalid,
        output arready,
        // R port
        input rready,
        output rdata, rresp, rlast, rvalid
    );
endinterface

//-------------------------------------------------------------------------
// AXIMM Write Interface
//-------------------------------------------------------------------------
interface aximm_wr #(
    // parameter int unsigned axi_id_width   = 0,
    // parameter int unsigned axi_user_width = 0,
    parameter int unsigned ADDR_WIDTH = 64,
    parameter int unsigned DATA_WIDTH = 256
);

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;

    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [ADDR_WIDTH-1:0] addr_t;
    typedef logic [STRB_WIDTH-1:0] strb_t;

    // AW port
    // axi_pkg::id_t awid;
    addr_t awaddr;
    axi_pkg::len_t awlen;
    axi_pkg::size_t awsize;
    axi_pkg::burst_t awburst;
    // axi_pkg::user_t awuser;
    logic awvalid;
    logic awready;

    // W port
    data_t wdata;
    strb_t wstrb;
    logic wlast;
    logic wvalid;
    logic wready;

    // B port
    // axi_id_t bid;
    axi_pkg::resp_t bresp;
    // axi_pkg::user_t buser;
    logic bvalid;
    logic bready;

    modport master(
        // AW port
        input awready,
        output awaddr, awlen, awsize, awburst, awvalid,
        // W port
        input wready,
        output wdata, wstrb, wlast, wvalid,
        // B port
        input bresp, bvalid,
        output bready
    );

    modport slave(
        // AW port
        input awaddr, awlen, awsize, awburst, awvalid,
        output awready,
        // W port
        input wdata, wstrb, wlast, wvalid,
        output wready,
        // B port
        input bready,
        output bresp, bvalid
    );
endinterface

//-------------------------------------------------------------------------
// AXIS Interface
//-------------------------------------------------------------------------
interface axis #(
    parameter DATA_WIDTH = 256
);

    localparam int unsigned STRB_WIDTH = DATA_WIDTH / 8;

    typedef logic [DATA_WIDTH-1:0] data_t;
    typedef logic [STRB_WIDTH-1:0] strb_t;

    logic  tvalid;
    data_t tdata;
    logic  tlast;
    logic  tready;

    modport master(output tvalid, tdata, tlast, input tready);
    modport slave(input tvalid, tdata, tlast, output tready);
endinterface

interface axis_fifo_intf #(
    parameter DATA_WIDTH = 256
);

    typedef logic [DATA_WIDTH -1:0] data_t;

    logic  full;
    logic  prog_full;
    logic  wr_en;
    data_t din;
    logic  empty;
    logic  rd_en;
    data_t dout;

    modport master(
        input full, prog_full, empty, dout,
        output wr_en, din, rd_en
    );

    modport slave(output full, prog_full, empty, dout, input wr_en, din, rd_en);

endinterface

//-------------------------------------------------------------------------
// AXILITE Interface
//-------------------------------------------------------------------------
interface axilite #(
    parameter int unsigned ADDR_WIDTH = 8,
    parameter int unsigned DATA_WIDTH = 32
);

    parameter int unsigned STRB_WIDTH = DATA_WIDTH / 8;

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

    modport master(
        // AW
        input awready,
        output awaddr, awvalid,
        // W
        input wready,
        output wdata, wstrb, wvalid,
        // B
        input bresp, bvalid,
        output bready,
        // AR
        input arready,
        output araddr, arvalid,
        // R
        input rdata, rresp, rvalid,
        output rready
    );
    modport slave(
        // AW
        input awaddr, awvalid,
        output awready,
        // W
        input wdata, wstrb, wvalid,
        output wready,
        // B
        input bready,
        output bresp, bvalid,
        // AR
        input araddr, arvalid,
        output arready,
        // R
        input rready,
        output rdata, rresp, rvalid
    );

    task automatic master_init_reg;
        awaddr  <= 0;
        awvalid <= 0;
        wdata   <= 0;
        wstrb   <= 0;
        wvalid  <= 0;
        bready  <= 0;
        araddr  <= 0;
        arvalid <= 0;
        rready  <= 0;
    endtask

    task automatic write_32b_r(ref logic clk, input addr_t addr,
                               input data_t data);
        logic [1:0] resp;
        //--- Address phase
        @(posedge clk);
        awaddr  <= addr;
        wdata   <= data;
        awvalid <= 1;
        wvalid  <= 1;
        wstrb   <= 4'b1111;
        while (~(awvalid && awready)) begin
            @(posedge clk);
        end
        awvalid <= 0;
        wvalid  <= 0;
        wstrb   <= 0;

        //--- Response phase
        @(posedge clk);
        bready <= 1;
        while (~(bvalid && bready)) begin
            @(posedge clk);
        end

        //--- Check Response phase
        resp = rresp;
        if (resp == 0) begin  // OKAY response
            $display("[PASS] data written at address 0x%h : %h", addr, data);
        end else begin  // ERROR response
            $display("[ERROR] AXI RRESP not equal to 0");
        end
        bready <= 0;
    endtask

    task automatic read_32b_r(ref logic clk, input addr_t addr,
                              ref data_t data);
        logic [1:0] resp;
        //-- Address Phase
        @(posedge clk);
        araddr  <= addr;
        arvalid <= 1;
        rready  <= 0;
        while (arready == 0) begin
            @(posedge clk);
        end
        araddr  <= 0;
        arvalid <= 0;
        rready  <= 1;

        //--- Read Data Phase
        while (~(rvalid && rready)) begin
            @(posedge clk);
        end
        data = rdata;

        //--- Check Response Phase
        resp = rresp;
        if (resp == 0) begin  // OKAY response
            $display("[PASS] data read at address 0x%h : %h", addr, data);
        end else begin  // ERROR response
            $display("[ERROR] AXI RRESP not equal to 0");
        end
        @(posedge clk);
        rready <= 0;
    endtask
endinterface
