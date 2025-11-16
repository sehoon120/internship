`ifndef AXI_PORT_SVH_
`define AXI_PORT_SVH_

// (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 <interface_name> AWADDR" *)

//////////////////////////////////////////////////
// Macros creating flat AXI ports
// `AXI_M_PORT(__name, __addr_t, __data_t, __strb_t)
// `AXI_S_PORT(__name, __addr_t, __data_t, __strb_t)
// `AXILITE_M_PORT(__name, __addr_t, __data_t, __strb_t)
// `AXILITE_S_PORT(__name, __addr_t, __data_t, __strb_t)
// `AXIS_M_PORT(__name, __data_t)
// `AXIS_S_PORT(__name, __data_t)
`define AXI_M_PORT(__name, __addr_t, __data_t, __strb_t)                         \
  output logic             m_axi_``__name``_awvalid, \
  output __addr_t          m_axi_``__name``_awaddr,  \
  input  logic             m_axi_``__name``_awready, \
  output axi_pkg::len_t    m_axi_``__name``_awlen,   \
  output axi_pkg::size_t   m_axi_``__name``_awsize,  \
  output axi_pkg::burst_t  m_axi_``__name``_awburst, \
  output logic             m_axi_``__name``_wvalid,  \
  output __data_t          m_axi_``__name``_wdata,   \
  output __strb_t          m_axi_``__name``_wstrb,   \
  output logic             m_axi_``__name``_wlast,   \
  input  logic             m_axi_``__name``_wready,  \
  output logic             m_axi_``__name``_bready,  \
  input  logic             m_axi_``__name``_bvalid,  \
  input  axi_pkg::resp_t   m_axi_``__name``_bresp,   \
  output logic             m_axi_``__name``_arvalid, \
  output __addr_t          m_axi_``__name``_araddr,  \
  output axi_pkg::len_t    m_axi_``__name``_arlen,   \
  output axi_pkg::size_t   m_axi_``__name``_arsize,  \
  output axi_pkg::burst_t  m_axi_``__name``_arburst, \
  output logic             m_axi_``__name``_rready,  \
  input  logic             m_axi_``__name``_arready, \
  input  logic             m_axi_``__name``_rvalid,  \
  input  __data_t          m_axi_``__name``_rdata,   \
  input  axi_pkg::resp_t   m_axi_``__name``_rresp,   \
  input  logic             m_axi_``__name``_rlast,
`define AXI_S_PORT(__name, __addr_t, __data_t, __strb_t)                             \
  input logic              s_axi_``__name``_awvalid,     \
  input __addr_t           s_axi_``__name``_awaddr,      \
  output logic             s_axi_``__name``_awready,     \
  input axi_pkg::len_t     s_axi_``__name``_awlen,       \
  input axi_pkg::size_t    s_axi_``__name``_awsize,      \
  input axi_pkg::burst_t   s_axi_``__name``_awburst,     \
  input logic              s_axi_``__name``_wvalid,      \
  input __data_t           s_axi_``__name``_wdata,       \
  input __strb_t           s_axi_``__name``_wstrb,       \
  input logic              s_axi_``__name``_wlast,       \
  output logic             s_axi_``__name``_wready,      \
  input logic              s_axi_``__name``_bready,      \
  output logic             s_axi_``__name``_bvalid,      \
  output axi_pkg::resp_t   s_axi_``__name``_bresp,       \
  input logic              s_axi_``__name``_arvalid,     \
  input __addr_t           s_axi_``__name``_araddr,      \
  input axi_pkg::len_t     s_axi_``__name``_arlen,       \
  input axi_pkg::size_t    s_axi_``__name``_arsize,      \
  input axi_pkg::burst_t   s_axi_``__name``_arburst,     \
  input logic              s_axi_``__name``_rready,      \
  output logic             s_axi_``__name``_arready,     \
  output logic             s_axi_``__name``_rvalid,      \
  output __data_t          s_axi_``__name``_rdata,       \
  output axi_pkg::resp_t   s_axi_``__name``_rresp,       \
  output logic             s_axi_``__name``_rlast,
`define AXILITE_M_PORT(__name, __addr_t, __data_t, __strb_t)  \
  output __addr_t         m_axilite_``__name``_awaddr,            \
  output logic            m_axilite_``__name``_awvalid,           \
  input logic             m_axilite_``__name``_awready,           \
  output __data_t         m_axilite_``__name``_wdata,             \
  output __strb_t         m_axilite_``__name``_wstrb,             \
  output logic            m_axilite_``__name``_wvalid,            \
  input logic             m_axilite_``__name``_wready,            \
  output logic            m_axilite_``__name``_bready,            \
  input axi_pkg::resp_t   m_axilite_``__name``_bresp,             \
  input logic             m_axilite_``__name``_bvalid,            \
  output __addr_t         m_axilite_``__name``_araddr,            \
  output logic            m_axilite_``__name``_arvalid,           \
  input logic             m_axilite_``__name``_arready,           \
  output logic            m_axilite_``__name``_rready,            \
  input __data_t          m_axilite_``__name``_rdata,             \
  input axi_pkg::resp_t   m_axilite_``__name``_rresp,             \
  input logic             m_axilite_``__name``_rvalid,
`define AXILITE_S_PORT(__name, __addr_t, __data_t, __strb_t)  \
  input __addr_t          s_axilite_``__name``_awaddr,            \
  input logic             s_axilite_``__name``_awvalid,           \
  output logic            s_axilite_``__name``_awready,           \
  input __data_t          s_axilite_``__name``_wdata,             \
  input __strb_t          s_axilite_``__name``_wstrb,             \
  input logic             s_axilite_``__name``_wvalid,            \
  output logic            s_axilite_``__name``_wready,            \
  input logic             s_axilite_``__name``_bready,            \
  output axi_pkg::resp_t  s_axilite_``__name``_bresp,             \
  output logic            s_axilite_``__name``_bvalid,            \
  input __addr_t          s_axilite_``__name``_araddr,            \
  input logic             s_axilite_``__name``_arvalid,           \
  output logic            s_axilite_``__name``_arready,           \
  input logic             s_axilite_``__name``_rready,            \
  output __data_t         s_axilite_``__name``_rdata,             \
  output axi_pkg::resp_t  s_axilite_``__name``_rresp,             \
  output logic            s_axilite_``__name``_rvalid,
`define AXIS_M_PORT(__name, __data_t)                    \
  output __data_t          m_axis_``__name``_tdata,      \
  output logic             m_axis_``__name``_tvalid,     \
  input logic              m_axis_``__name``_tready,     \
  output logic             m_axis_``__name``_tlast,
`define AXIS_S_PORT(__name, __data_t)                    \
  input __data_t           s_axis_``__name``_tdata,      \
  input logic              s_axis_``__name``_tvalid,     \
  output logic             s_axis_``__name``_tready,     \
  input logic              s_axis_``__name``_tlast,

//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Macros creating flat AXI port connections
// `AXI_M_CON_FLAT(__name, __axi_intf)
// `AXI_S_CON_FLAT(__name, __axi_intf)
// `AXILITE_M_CON_FLAT(__name, __axi_intf)
// `AXILITE_S_CON_FLAT(__name, __axi_intf)
// `AXIS_M_CON_FLAT(__name, __axis_intf)
// `AXIS_S_CON_FLAT(__name, __axis_intf)
//////////////////////////////////////////////////
`define AXI_M_CON_FLAT(__name, __axi_intf)                \
  .m_axi_``__name``_awvalid  (``__axi_intf``.awvalid  ),  \
  .m_axi_``__name``_awaddr   (``__axi_intf``.awaddr   ),  \
  .m_axi_``__name``_awready  (``__axi_intf``.awready  ),  \
  .m_axi_``__name``_awlen    (``__axi_intf``.awlen    ),  \
  .m_axi_``__name``_awsize   (``__axi_intf``.awsize   ),  \
  .m_axi_``__name``_awburst  (``__axi_intf``.awburst  ),  \
  .m_axi_``__name``_wvalid   (``__axi_intf``.wvalid   ),  \
  .m_axi_``__name``_wdata    (``__axi_intf``.wdata    ),  \
  .m_axi_``__name``_wstrb    (``__axi_intf``.wstrb    ),  \
  .m_axi_``__name``_wlast    (``__axi_intf``.wlast    ),  \
  .m_axi_``__name``_wready   (``__axi_intf``.wready   ),  \
  .m_axi_``__name``_bready   (``__axi_intf``.bready   ),  \
  .m_axi_``__name``_bvalid   (``__axi_intf``.bvalid   ),  \
  .m_axi_``__name``_bresp    (``__axi_intf``.bresp    ),  \
  .m_axi_``__name``_arvalid  (``__axi_intf``.arvalid  ),  \
  .m_axi_``__name``_araddr   (``__axi_intf``.araddr   ),  \
  .m_axi_``__name``_arlen    (``__axi_intf``.arlen    ),  \
  .m_axi_``__name``_arsize   (``__axi_intf``.arsize   ),  \
  .m_axi_``__name``_arburst  (``__axi_intf``.arburst  ),  \
  .m_axi_``__name``_rready   (``__axi_intf``.rready   ),  \
  .m_axi_``__name``_arready  (``__axi_intf``.arready  ),  \
  .m_axi_``__name``_rvalid   (``__axi_intf``.rvalid   ),  \
  .m_axi_``__name``_rdata    (``__axi_intf``.rdata    ),  \
  .m_axi_``__name``_rresp    (``__axi_intf``.rresp    ),  \
  .m_axi_``__name``_rlast    (``__axi_intf``.rlast    ),
`define AXI_S_CON_FLAT(__name, __axi_intf)                \
  .s_axi_``__name``_awvalid  (``__axi_intf``.awvalid  ),  \
  .s_axi_``__name``_awaddr   (``__axi_intf``.awaddr   ),  \
  .s_axi_``__name``_awready  (``__axi_intf``.awready  ),  \
  .s_axi_``__name``_awlen    (``__axi_intf``.awlen    ),  \
  .s_axi_``__name``_awsize   (``__axi_intf``.awsize   ),  \
  .s_axi_``__name``_awburst  (``__axi_intf``.awburst  ),  \
  .s_axi_``__name``_wvalid   (``__axi_intf``.wvalid   ),  \
  .s_axi_``__name``_wdata    (``__axi_intf``.wdata    ),  \
  .s_axi_``__name``_wstrb    (``__axi_intf``.wstrb    ),  \
  .s_axi_``__name``_wlast    (``__axi_intf``.wlast    ),  \
  .s_axi_``__name``_wready   (``__axi_intf``.wready   ),  \
  .s_axi_``__name``_bready   (``__axi_intf``.bready   ),  \
  .s_axi_``__name``_bvalid   (``__axi_intf``.bvalid   ),  \
  .s_axi_``__name``_bresp    (``__axi_intf``.bresp    ),  \
  .s_axi_``__name``_arvalid  (``__axi_intf``.arvalid  ),  \
  .s_axi_``__name``_araddr   (``__axi_intf``.araddr   ),  \
  .s_axi_``__name``_arlen    (``__axi_intf``.arlen    ),  \
  .s_axi_``__name``_arsize   (``__axi_intf``.arsize   ),  \
  .s_axi_``__name``_arburst  (``__axi_intf``.arburst  ),  \
  .s_axi_``__name``_rready   (``__axi_intf``.rready   ),  \
  .s_axi_``__name``_arready  (``__axi_intf``.arready  ),  \
  .s_axi_``__name``_rvalid   (``__axi_intf``.rvalid   ),  \
  .s_axi_``__name``_rdata    (``__axi_intf``.rdata    ),  \
  .s_axi_``__name``_rresp    (``__axi_intf``.rresp    ),  \
  .s_axi_``__name``_rlast    (``__axi_intf``.rlast    ),
`define AXILITE_M_CON_FLAT(__name, __axilite_intf)                \
  .m_axilite_``__name``_awaddr   (``__axilite_intf``.awaddr   ),  \
  .m_axilite_``__name``_awvalid  (``__axilite_intf``.awvalid  ),  \
  .m_axilite_``__name``_awready  (``__axilite_intf``.awready  ),  \
  .m_axilite_``__name``_wdata    (``__axilite_intf``.wdata    ),  \
  .m_axilite_``__name``_wstrb    (``__axilite_intf``.wstrb    ),  \
  .m_axilite_``__name``_wvalid   (``__axilite_intf``.wvalid   ),  \
  .m_axilite_``__name``_wready   (``__axilite_intf``.wready   ),  \
  .m_axilite_``__name``_bready   (``__axilite_intf``.bready   ),  \
  .m_axilite_``__name``_bresp    (``__axilite_intf``.bresp    ),  \
  .m_axilite_``__name``_bvalid   (``__axilite_intf``.bvalid   ),  \
  .m_axilite_``__name``_araddr   (``__axilite_intf``.araddr   ),  \
  .m_axilite_``__name``_arvalid  (``__axilite_intf``.arvalid  ),  \
  .m_axilite_``__name``_arready  (``__axilite_intf``.arready  ),  \
  .m_axilite_``__name``_rready   (``__axilite_intf``.rready   ),  \
  .m_axilite_``__name``_rdata    (``__axilite_intf``.rdata    ),  \
  .m_axilite_``__name``_rresp    (``__axilite_intf``.rresp    ),  \
  .m_axilite_``__name``_rvalid   (``__axilite_intf``.rvalid   ),
`define AXILITE_S_CON_FLAT(__name, __axilite_intf)                \
  .s_axilite_``__name``_awaddr   (``__axilite_intf``.awaddr   ),  \
  .s_axilite_``__name``_awvalid  (``__axilite_intf``.awvalid  ),  \
  .s_axilite_``__name``_awready  (``__axilite_intf``.awready  ),  \
  .s_axilite_``__name``_wdata    (``__axilite_intf``.wdata    ),  \
  .s_axilite_``__name``_wstrb    (``__axilite_intf``.wstrb    ),  \
  .s_axilite_``__name``_wvalid   (``__axilite_intf``.wvalid   ),  \
  .s_axilite_``__name``_wready   (``__axilite_intf``.wready   ),  \
  .s_axilite_``__name``_bready   (``__axilite_intf``.bready   ),  \
  .s_axilite_``__name``_bresp    (``__axilite_intf``.bresp    ),  \
  .s_axilite_``__name``_bvalid   (``__axilite_intf``.bvalid   ),  \
  .s_axilite_``__name``_araddr   (``__axilite_intf``.araddr   ),  \
  .s_axilite_``__name``_arvalid  (``__axilite_intf``.arvalid  ),  \
  .s_axilite_``__name``_arready  (``__axilite_intf``.arready  ),  \
  .s_axilite_``__name``_rready   (``__axilite_intf``.rready   ),  \
  .s_axilite_``__name``_rdata    (``__axilite_intf``.rdata    ),  \
  .s_axilite_``__name``_rresp    (``__axilite_intf``.rresp    ),  \
  .s_axilite_``__name``_rvalid   (``__axilite_intf``.rvalid   ),
`define AXIS_M_CON_FLAT(__name, __axis_intf)                \
  .m_axis_``__name``_tvalid   (``__axis_intf``.tvalid   ),  \
  .m_axis_``__name``_tdata    (``__axis_intf``.tdata    ),  \
  .m_axis_``__name``_tlast    (``__axis_intf``.tlast    ),  \
  .m_axis_``__name``_tready   (``__axis_intf``.tready   ),
`define AXIS_S_CON_FLAT(__name, __axis_intf)                \
  .s_axis_``__name``_tvalid   (``__axis_intf``.tvalid   ),  \
  .s_axis_``__name``_tdata    (``__axis_intf``.tdata    ),  \
  .s_axis_``__name``_tlast    (``__axis_intf``.tlast    ),  \
  .s_axis_``__name``_tready   (``__axis_intf``.tready   ),


`endif
