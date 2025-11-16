

package axi_pkg;
    /// AXI BURST max length
    parameter AXI_BURST_MAX = 256;

    /// AXI Transaction Burst Width.
    typedef logic [1:0] burst_t;
    /// AXI Transaction Response Type.
    typedef logic [1:0] resp_t;
    /// AXI Transaction Protection Type.
    typedef logic [2:0] prot_t;
    /// AXI Transaction Length Type.
    typedef logic [7:0] len_t;
    /// AXI Transaction Size Type.
    typedef logic [2:0] size_t;


    typedef enum bit [1:0] {
        RESP_OKAY   = 2'b00,  // okay
        RESP_EXOKAY = 2'b01,  // exclusive okay
        RESP_SLVERR = 2'b10,  // slave error
        RESP_DECERR = 2'b11   // decode error
    } resp_err_t;
    typedef logic [511:0] data_512_t;
    typedef logic [255:0] data_256_t;
    typedef logic [127:0] data_128_t;
    typedef logic [63:0] data_64_t;
    typedef logic [31:0] data_32_t;
    typedef logic [15:0] data_16_t;
    typedef logic [7:0] data_8_t;

    typedef logic [31:0] strb_256_t;
    typedef logic [15:0] strb_128_t;
    typedef logic [7:0] strb_64_t;
    typedef logic [3:0] strb_32_t;
    typedef logic [1:0] strb_16_t;
    typedef logic strb_8_t;

    typedef logic [63:0] addr_64_t;
    typedef logic [31:0] addr_32_t;
    typedef logic [15:0] addr_16_t;
    typedef logic [7:0] addr_8_t;
    typedef struct packed {
        addr_64_t   addr;
        logic [8:0] len;
    } trans_64_t;
    typedef struct packed {
        addr_32_t   addr;
        logic [8:0] len;
    } trans_32_t;


    typedef logic [31:0] reg_t;
    typedef reg_t reg_map_256_t[256];
    typedef reg_t reg_map_128_t[128];
    typedef reg_t reg_map_64_t[64];
    typedef reg_t reg_map_32_t[32];
    typedef reg_t reg_map_16_t[16];


    // Commonly used rule types for `axi_xbar` (64-bit addresses).
    typedef struct packed {
        int unsigned idx;
        logic [63:0] start_addr;
        logic [63:0] end_addr;
    } xbar_rule_64_t;

    // Commonly used rule types for `axi_xbar` (32-bit addresses).
    typedef struct packed {
        int unsigned idx;
        logic [31:0] start_addr;
        logic [31:0] end_addr;
    } xbar_rule_32_t;
endpackage


package axi_test_pkg;
    parameter CLK_PERIOD = 10;

    //-------------------------------------------------------------------------
    // Testing
    //-------------------------------------------------------------------------
    // Task to start a read transaction.
    task automatic axi_read_start(
        ref logic clk, ref logic [63:0] axi_read_start_addr,
        ref logic [31:0] axi_read_length, ref logic init_read,
        ref logic axi_read_start_ready, input logic [63:0] addr,
        input logic [31:0] length);
        begin
            wait (axi_read_start_ready);
            @(posedge clk);
            // Load transaction parameters.
            axi_read_start_addr = addr;
            axi_read_length     = length;
            @(posedge clk);
            // Pulse the start signal for one cycle.
            init_read = 1;
            @(posedge clk);
            init_read = 0;
            @(posedge clk);
            #10;
        end
    endtask

    // Task to start a write transaction.
    task automatic axi_write_start(
        ref logic clk, ref logic [63:0] axi_write_start_addr,
        ref logic [31:0] axi_write_length, ref logic init_write,
        ref logic axi_write_start_ready, input logic [63:0] addr,
        input logic [31:0] length);
        begin
            // Wait for axi dma to be ready.
            wait (axi_write_start_ready);
            @(posedge clk);
            // Load transaction parameters.
            axi_write_start_addr = addr;
            axi_write_length     = length;
            @(posedge clk);
            // Pulse the start signal for one cycle.
            init_write = 1;
            @(posedge clk);
            init_write = 0;

            @(posedge clk);
            #10;
        end
    endtask

endpackage
