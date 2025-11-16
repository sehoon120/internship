`ifndef AXI_ASSIGN_SVH_
`define AXI_ASSIGN_SVH_

/////////////////////////////////////////////////////////////////////////////////////
// Internal implementation for assigning one AXI struct or interface to
// another struct or interface.
/////////////////////////////////////////////////////////////////////////////////////
// The path to the signals on each side is defined by the `__sep*` arguments.
// The `__opt_as` argument allows to use this standalone (with `__opt_as = assign`)
// or in assignments inside processes (with `__opt_as` void).
// AXI
`define __AXI_TO_AW(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)   \
  __opt_as __lhs``__lhs_sep``awaddr  = __rhs``__rhs_sep``awaddr;    \
  __opt_as __lhs``__lhs_sep``awlen   = __rhs``__rhs_sep``awlen;     \
  __opt_as __lhs``__lhs_sep``awsize  = __rhs``__rhs_sep``awsize;    \
  __opt_as __lhs``__lhs_sep``awburst = __rhs``__rhs_sep``awburst;
`define __AXI_TO_W(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)    \
  __opt_as __lhs``__lhs_sep``wdata   = __rhs``__rhs_sep``wdata;     \
  __opt_as __lhs``__lhs_sep``wstrb   = __rhs``__rhs_sep``wstrb;     \
  __opt_as __lhs``__lhs_sep``wlast   = __rhs``__rhs_sep``wlast;
`define __AXI_TO_B(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``bresp   = __rhs``__rhs_sep``bresp;
`define __AXI_TO_AR(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)   \
  __opt_as __lhs``__lhs_sep``araddr  = __rhs``__rhs_sep``araddr;    \
  __opt_as __lhs``__lhs_sep``arlen   = __rhs``__rhs_sep``arlen;     \
  __opt_as __lhs``__lhs_sep``arsize  = __rhs``__rhs_sep``arsize;    \
  __opt_as __lhs``__lhs_sep``arburst = __rhs``__rhs_sep``arburst;
`define __AXI_TO_R(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)    \
  __opt_as __lhs``__lhs_sep``rdata   = __rhs``__rhs_sep``rdata;     \
  __opt_as __lhs``__lhs_sep``rresp   = __rhs``__rhs_sep``rresp;     \
  __opt_as __lhs``__lhs_sep``rlast   = __rhs``__rhs_sep``rlast;
`define __AXI_TO_REQ(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)  \
  `__AXI_TO_AW(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)        \
  __opt_as __lhs``__lhs_sep``awvalid = __rhs``__rhs_sep``awvalid;   \
  `__AXI_TO_W(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)         \
  __opt_as __lhs``__lhs_sep``wvalid  = __rhs``__rhs_sep``wvalid;    \
  __opt_as __lhs``__lhs_sep``bready  = __rhs``__rhs_sep``bready;    \
  `__AXI_TO_AR(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)        \
  __opt_as __lhs``__lhs_sep``arvalid = __rhs``__rhs_sep``arvalid;   \
  __opt_as __lhs``__lhs_sep``rready  = __rhs``__rhs_sep``rready;
`define __AXI_TO_RESP(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``awready = __rhs``__rhs_sep``awready;   \
  __opt_as __lhs``__lhs_sep``arready = __rhs``__rhs_sep``arready;   \
  __opt_as __lhs``__lhs_sep``wready  = __rhs``__rhs_sep``wready;    \
  __opt_as __lhs``__lhs_sep``bvalid  = __rhs``__rhs_sep``bvalid;    \
  `__AXI_TO_B(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)         \
  __opt_as __lhs``__lhs_sep``rvalid  = __rhs``__rhs_sep``rvalid;    \
  `__AXI_TO_R(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)

/////////////////////////////////////////////////////////////////////////////////////
// Assigning one AXI4+ATOP interface to another, as if you would do `assign slv = mst;`
//
// The channel assignments `AXI_ASSIGN_XX(dst, src)` assign all payload
// and the valid signal of the `XX` channel from the `src` to the `dst` interface
// and they assign the ready signal from the
// `src` to the `dst` interface.
// The interface assignment `AXI_ASSIGN(dst, src)` assigns all channels
// including handshakes as if `src` was the master of `dst`.
/////////////////////////////////////////////////////////////////////////////////////
// Usage Example:
// `AXI_ASSIGN(slv, mst)
// `AXI_ASSIGN_AW(dst, src)
// `AXI_ASSIGN_R(dst, src)
`define AXI_ASSIGN_AW(__opt_as, dst, src)  \
  `__AXI_TO_AW(__opt_as, dst, ., src, .)   \
  assign dst.awvalid = src.awvalid;        \
  assign src.awready = dst.awready;
`define AXI_ASSIGN_W(__opt_as, dst, src)   \
  `__AXI_TO_W(__opt_as, dst, ., src, .)    \
  assign dst.wvalid  = src.wvalid;         \
  assign src.wready  = dst.wready;
`define AXI_ASSIGN_B(__opt_as, dst, src)   \
  `__AXI_TO_B(__opt_as, dst, ., src, .)    \
  assign dst.bvalid  = src.bvalid;         \
  assign src.bready  = dst.bready;
`define AXI_ASSIGN_AR(__opt_as, dst, src)  \
  `__AXI_TO_AR(__opt_as, dst, ., src, .)   \
  assign dst.arvalid = src.arvalid;        \
  assign src.arready = dst.arready;
`define AXI_ASSIGN_R(__opt_as, dst, src)   \
  `__AXI_TO_R(__opt_as, dst, ., src, .)    \
  assign dst.rvalid  = src.rvalid;         \
  assign src.rready  = dst.rready;
`define AXI_ASSIGN(slv, mst)               \
  `AXI_ASSIGN_AW(slv, mst)                 \
  `AXI_ASSIGN_W(slv, mst)                  \
  `AXI_ASSIGN_B(mst, slv)                  \
  `AXI_ASSIGN_AR(slv, mst)                 \
  `AXI_ASSIGN_R(mst, slv)


////////////////////////////////////////////////////////////////////////////////////////////////////
// Assigning an interface from channel or request/response structs outside a process.
//
// The channel macros `AXI_ASSIGN_FROM_XX(axi_if, xx_struct)` assign the payload signals of the
// `axi_if` interface from the signals in `xx_struct`.  They do not assign the handshake signals.
// The request macro `AXI_ASSIGN_FROM_REQ(axi_if, req_struct)` assigns all request channels (AW, W,
// AR) and the request-side handshake signals (AW, W, and AR valid and B and R ready) of the
// `axi_if` interface from the signals in `req_struct`.
// The response macro `AXI_ASSIGN_FROM_RESP(axi_if, resp_struct)` assigns both response channels (B
// and R) and the response-side handshake signals (B and R valid and AW, W, and AR ready) of the
// `axi_if` interface from the signals in `resp_struct`.
//
// Usage Example:
// `AXI_ASSIGN_FROM_REQ(my_if, my_req_struct)
`define AXI_ASSIGN_FROM_AW(axi_if, aw_struct) \
  `__AXI_TO_AW(assign, axi_if, _, aw_struct, .)
`define AXI_ASSIGN_FROM_W(axi_if, w_struct) \
  `__AXI_TO_W(assign, axi_if, _, w_struct, .)
`define AXI_ASSIGN_FROM_B(axi_if, b_struct) \
  `__AXI_TO_B(assign, axi_if, _, b_struct, .)
`define AXI_ASSIGN_FROM_AR(axi_if, ar_struct) \
  `__AXI_TO_AR(assign, axi_if, _, ar_struct, .)
`define AXI_ASSIGN_FROM_R(axi_if, r_struct) \
  `__AXI_TO_R(assign, axi_if, _, r_struct, .)
`define AXI_ASSIGN_FROM_REQ(axi_if, req_struct) \
  `__AXI_TO_REQ(assign, axi_if, _, req_struct, .)
`define AXI_ASSIGN_FROM_RESP(axi_if, resp_struct) \
  `__AXI_TO_RESP(assign, axi_if, _, resp_struct, .)
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Assigning channel or request/response structs from an interface outside a process.
//
// The channel macros `AXI_ASSIGN_TO_XX(xx_struct, axi_if)` assign the signals of `xx_struct` to the
// payload signals of that channel in the `axi_if` interface.  They do not assign the handshake
// signals.
// The request macro `AXI_ASSIGN_TO_REQ(axi_if, req_struct)` assigns all signals of `req_struct`
// (i.e., request channel (AW, W, AR) payload and request-side handshake signals (AW, W, and AR
// valid and B and R ready)) to the signals in the `axi_if` interface.
// The response macro `AXI_ASSIGN_TO_RESP(axi_if, resp_struct)` assigns all signals of `resp_struct`
// (i.e., response channel (B and R) payload and response-side handshake signals (B and R valid and
// AW, W, and AR ready)) to the signals in the `axi_if` interface.
//
// Usage Example:
// `AXI_ASSIGN_TO_REQ(my_req_struct, my_if)

`define AXI_ASSIGN_TO_AW(aw_struct, axi_if) \
  `__AXI_TO_AW(assign, aw_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_W(w_struct, axi_if) \
  `__AXI_TO_W(assign, w_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_B(b_struct, axi_if) \
  `__AXI_TO_B(assign, b_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_AR(ar_struct, axi_if) \
  `__AXI_TO_AR(assign, ar_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_R(r_struct, axi_if) \
  `__AXI_TO_R(assign, r_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_REQ(req_struct, axi_if) \
  `__AXI_TO_REQ(assign, req_struct, ., axi_if, _)
`define AXI_ASSIGN_TO_RESP(resp_struct, axi_if) \
  `__AXI_TO_RESP(assign, resp_struct, ., axi_if, _)
////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// Macros for assigning flattened AXI ports to req/resp AXI structs
// Flat AXI ports are required by the Vivado IP Integrator.
// Vivado naming convention is followed.
//
// Usage Example:
// `AXI_ASSIGN_MASTER_TO_FLAT("my_bus", my_req_struct, my_rsp_struct)
`define AXI_ASSIGN_MASTER_TO_FLAT(pat, __intf) \
  assign m_axi_``pat``_awvalid  = __intf.awvalid; \
  assign m_axi_``pat``_awaddr   = __intf.awaddr;  \
  assign m_axi_``pat``_awlen    = __intf.awlen;   \
  assign m_axi_``pat``_awsize   = __intf.awsize;  \
  assign m_axi_``pat``_awburst  = __intf.awburst; \
  assign __intf.awready = m_axi_``pat``_awready;  \
                                                  \
  assign m_axi_``pat``_wvalid   = __intf.wvalid;  \
  assign m_axi_``pat``_wdata    = __intf.wdata;   \
  assign m_axi_``pat``_wstrb    = __intf.wstrb;   \
  assign m_axi_``pat``_wlast    = __intf.wlast;   \
  assign __intf.wready  = m_axi_``pat``_wready;   \
                                                  \
  assign m_axi_``pat``_bready   = __intf.bready;  \
  assign __intf.bvalid  = m_axi_``pat``_bvalid;   \
  assign __intf.bresp   = m_axi_``pat``_bresp;    \
                                                  \
  assign m_axi_``pat``_arvalid  = __intf.arvalid; \
  assign m_axi_``pat``_araddr   = __intf.araddr;  \
  assign m_axi_``pat``_arlen    = __intf.arlen;   \
  assign m_axi_``pat``_arsize   = __intf.arsize;  \
  assign m_axi_``pat``_arburst  = __intf.arburst; \
  assign __intf.arready = m_axi_``pat``_arready;  \
                                                  \
  assign m_axi_``pat``_rready   = __intf.rready;  \
  assign __intf.rvalid  = m_axi_``pat``_rvalid;   \
  assign __intf.rdata   = m_axi_``pat``_rdata;    \
  assign __intf.rresp   = m_axi_``pat``_rresp;    \
  assign __intf.rlast   = m_axi_``pat``_rlast;
`define AXI_ASSIGN_SLAVE_TO_FLAT(pat, __intf)     \
  assign __intf.awvalid = s_axi_``pat``_awvalid;  \
  assign __intf.awaddr  = s_axi_``pat``_awaddr;   \
  assign __intf.awlen   = s_axi_``pat``_awlen;    \
  assign __intf.awsize  = s_axi_``pat``_awsize;   \
  assign __intf.awburst = s_axi_``pat``_awburst;  \
  assign s_axi_``pat``_awready = __intf.awready;  \
                                                  \
  assign __intf.wvalid   = s_axi_``pat``_wvalid;  \
  assign __intf.wdata    = s_axi_``pat``_wdata;   \
  assign __intf.wstrb    = s_axi_``pat``_wstrb;   \
  assign __intf.wlast    = s_axi_``pat``_wlast;   \
  assign s_axi_``pat``_wready = __intf.wready;    \
                                                  \
  assign __intf.bready = s_axi_``pat``_bready;    \
  assign s_axi_``pat``_bvalid = __intf.bvalid;    \
  assign s_axi_``pat``_bresp = __intf.bresp;      \
                                                  \
  assign __intf.arvalid = s_axi_``pat``_arvalid;  \
  assign __intf.araddr = s_axi_``pat``_araddr;    \
  assign __intf.arlen = s_axi_``pat``_arlen;      \
  assign __intf.arsize = s_axi_``pat``_arsize;    \
  assign __intf.arburst = s_axi_``pat``_arburst;  \
  assign s_axi_``pat``_arready = __intf.arready;  \
                                                  \
  assign __intf.rready = s_axi_``pat``_rready;    \
  assign s_axi_``pat``_rvalid = __intf.rvalid;    \
  assign s_axi_``pat``_rdata = __intf.rdata;      \
  assign s_axi_``pat``_rresp = __intf.rresp;      \
  assign s_axi_``pat``_rlast = __intf.rlast;

`define __AXILITE_TO_AW(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``awaddr  = __rhs``__rhs_sep``awaddr;
`define __AXILITE_TO_W(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)    \
  __opt_as __lhs``__lhs_sep``wdata   = __rhs``__rhs_sep``wdata;     \
  __opt_as __lhs``__lhs_sep``wstrb   = __rhs``__rhs_sep``wstrb;
`define __AXILITE_TO_B(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``bresp   = __rhs``__rhs_sep``bresp;
`define __AXILITE_TO_AR(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``araddr  = __rhs``__rhs_sep``araddr;
`define __AXILITE_TO_R(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)    \
  __opt_as __lhs``__lhs_sep``rdata   = __rhs``__rhs_sep``rdata;     \
  __opt_as __lhs``__lhs_sep``rresp   = __rhs``__rhs_sep``rresp;
`define __AXILITE_TO_REQ(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)  \
  `__AXILITE_TO_AW(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)        \
  __opt_as __lhs``__lhs_sep``awvalid = __rhs``__rhs_sep``awvalid;   \
  `__AXILITE_TO_W(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)         \
  __opt_as __lhs``__lhs_sep``wvalid  = __rhs``__rhs_sep``wvalid;    \
  __opt_as __lhs``__lhs_sep``bready  = __rhs``__rhs_sep``bready;    \
  `__AXILITE_TO_AR(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)        \
  __opt_as __lhs``__lhs_sep``arvalid = __rhs``__rhs_sep``arvalid;   \
  __opt_as __lhs``__lhs_sep``rready  = __rhs``__rhs_sep``rready;
`define __AXILITE_TO_RESP(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep) \
  __opt_as __lhs``__lhs_sep``awready = __rhs``__rhs_sep``awready;   \
  __opt_as __lhs``__lhs_sep``arready = __rhs``__rhs_sep``arready;   \
  __opt_as __lhs``__lhs_sep``wready  = __rhs``__rhs_sep``wready;    \
  __opt_as __lhs``__lhs_sep``bvalid  = __rhs``__rhs_sep``bvalid;    \
  `__AXILITE_TO_B(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)         \
  __opt_as __lhs``__lhs_sep``rvalid  = __rhs``__rhs_sep``rvalid;    \
  `__AXILITE_TO_R(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)

/////////////////////////////////////////////////////////////////////////////////////
// The channel assignments `AXILITE_ASSIGN_XX(dst, src)` assign all payload
// and the valid signal of the `XX` channel from the `src` to the `dst` interface
// and they assign the ready signal from the
// `src` to the `dst` interface.
// The interface assignment `AXILITE_ASSIGN(dst, src)` assigns all channels
// including handshakes as if `src` was the master of `dst`.
/////////////////////////////////////////////////////////////////////////////////////
// Usage Example:
// `AXILITE_ASSIGN(slv, mst)
// `AXILITE_ASSIGN_AW(dst, src)
// `AXILITE_ASSIGN_R(dst, src)
`define AXILITE_ASSIGN_AW(__opt_as, dst, src) \
  `__AXILITE_TO_AW(__opt_as, dst, ., src, .)  \
  assign dst.awvalid = src.awvalid;           \
  assign src.awready = dst.awready;
`define AXILITE_ASSIGN_W(__opt_as, dst, src)  \
  `__AXILITE_TO_W(__opt_as, dst, ., src, .)   \
  assign dst.wvalid  = src.wvalid;            \
  assign src.wready  = dst.wready;
`define AXILITE_ASSIGN_B(__opt_as, dst, src)  \
  `__AXILITE_TO_B(__opt_as, dst, ., src, .)   \
  assign dst.bvalid  = src.bvalid;            \
  assign src.bready  = dst.bready;
`define AXILITE_ASSIGN_AR(__opt_as, dst, src) \
  `__AXILITE_TO_AR(__opt_as, dst, ., src, .)  \
  assign dst.arvalid = src.arvalid;           \
  assign src.arready = dst.arready;
`define AXILITE_ASSIGN_R(__opt_as, dst, src)  \
  `__AXILITE_TO_R(__opt_as, dst, ., src, .)   \
  assign dst.rvalid  = src.rvalid;            \
  assign src.rready  = dst.rready;
`define AXILITE_ASSIGN(slv, mst)              \
  `AXILITE_ASSIGN_AW(slv, mst)                \
  `AXILITE_ASSIGN_W(slv, mst)                 \
  `AXILITE_ASSIGN_B(mst, slv)                 \
  `AXILITE_ASSIGN_AR(slv, mst)                \
  `AXILITE_ASSIGN_R(mst, slv)

////////////////////////////////////////////////////////////////////////////////////////////////////
// Assigning an interface from channel or request/response structs outside a process.
//
// The channel macros `AXILITE_ASSIGN_FROM_XX(axilite_if, xx_struct)` assign the payload signals of the
// `axilite_if` interface from the signals in `xx_struct`.  They do not assign the handshake signals.
// The request macro `AXILITE_ASSIGN_FROM_REQ(axilite_if, req_struct)` assigns all request channels (AW, W,
// AR) and the request-side handshake signals (AW, W, and AR valid and B and R ready) of the
// `axilite_if` interface from the signals in `req_struct`.
// The response macro `AXILITE_ASSIGN_FROM_RESP(axilite_if, resp_struct)` assigns both response channels (B
// and R) and the response-side handshake signals (B and R valid and AW, W, and AR ready) of the
// `axilite_if` interface from the signals in `resp_struct`.
//
// Usage Example:
// `AXILITE_ASSIGN_FROM_REQ(my_if, my_req_struct)
`define AXILITE_ASSIGN_FROM_AW(axilite_if, aw_struct) \
  `__AXILITE_TO_AW(assign, axilite_if, _, aw_struct, .)
`define AXILITE_ASSIGN_FROM_W(axilite_if, w_struct) \
  `__AXILITE_TO_W(assign, axilite_if, _, w_struct, .)
`define AXILITE_ASSIGN_FROM_B(axilite_if, b_struct) \
  `__AXILITE_TO_B(assign, axilite_if, _, b_struct, .)
`define AXILITE_ASSIGN_FROM_AR(axilite_if, ar_struct) \
  `__AXILITE_TO_AR(assign, axilite_if, _, ar_struct, .)
`define AXILITE_ASSIGN_FROM_R(axilite_if, r_struct) \
  `__AXILITE_TO_R(assign, axilite_if, _, r_struct, .)
`define AXILITE_ASSIGN_FROM_REQ(axilite_if, req_struct) \
  `__AXILITE_TO_REQ(assign, axilite_if, _, req_struct, .)
`define AXILITE_ASSIGN_FROM_RESP(axilite_if, resp_struct) \
  `__AXILITE_TO_RESP(assign, axilite_if, _, resp_struct, .)


////////////////////////////////////////////////////////////////////////////////////////////////////
// Assigning channel or request/response structs from an interface outside a process.
//
// The channel macros `AXILITE_ASSIGN_TO_XX(xx_struct, axilite_if)` assign the signals of `xx_struct` to the
// payload signals of that channel in the `axilite_if` interface.  They do not assign the handshake
// signals.
// The request macro `AXILITE_ASSIGN_TO_REQ(axilite_if, req_struct)` assigns all signals of `req_struct`
// (i.e., request channel (AW, W, AR) payload and request-side handshake signals (AW, W, and AR
// valid and B and R ready)) to the signals in the `axilite_if` interface.
// The response macro `AXILITE_ASSIGN_TO_RESP(axilite_if, resp_struct)` assigns all signals of `resp_struct`
// (i.e., response channel (B and R) payload and response-side handshake signals (B and R valid and
// AW, W, and AR ready)) to the signals in the `axilite_if` interface.
//
// Usage Example:
// `AXILITE_ASSIGN_TO_REQ(my_req_struct, my_if)
`define AXILITE_ASSIGN_TO_AW(aw_struct, axilite_if) \
  `__AXILITE_TO_AW(assign, aw_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_W(w_struct, axilite_if) \
  `__AXILITE_TO_W(assign, w_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_B(b_struct, axilite_if) \
  `__AXILITE_TO_B(assign, b_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_AR(ar_struct, axilite_if) \
  `__AXILITE_TO_AR(assign, ar_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_R(r_struct, axilite_if) \
  `__AXILITE_TO_R(assign, r_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_REQ(req_struct, axilite_if) \
  `__AXILITE_TO_REQ(assign, req_struct, ., axilite_if, _)
`define AXILITE_ASSIGN_TO_RESP(resp_struct, axilite_if) \
  `__AXILITE_TO_RESP(assign, resp_struct, ., axilite_if, _)


/////////////////////////////////////////////////////////////////////////////////////
// Macros for assigning flattened AXILITE ports to req/resp AXILITE structs
// Flat AXILITE ports are required by the Vivado IP Integrator.
// Vivado naming convention is followed.
//
// Usage Example:
// `AXILITE_ASSIGN_MASTER_TO_FLAT("my_bus", my_req_struct, my_rsp_struct)
`define AXILITE_ASSIGN_MASTER_TO_FLAT(pat, __intf) \
  assign m_axilite_``pat``_awvalid  = __intf.awvalid; \
  assign m_axilite_``pat``_awaddr   = __intf.awaddr;  \
  assign __intf.awready = m_axilite_``pat``_awready;  \
                                                      \
  assign m_axilite_``pat``_wvalid   = __intf.wvalid;  \
  assign m_axilite_``pat``_wdata    = __intf.wdata;   \
  assign m_axilite_``pat``_wstrb    = __intf.wstrb;   \
  assign __intf.wready  = m_axilite_``pat``_wready;   \
                                                      \
  assign m_axilite_``pat``_bready   = __intf.bready;  \
  assign __intf.bvalid  = m_axilite_``pat``_bvalid;   \
  assign __intf.bresp   = m_axilite_``pat``_bresp;    \
                                                      \
  assign m_axilite_``pat``_arvalid  = __intf.arvalid; \
  assign m_axilite_``pat``_araddr   = __intf.araddr;  \
  assign __intf.arready = m_axilite_``pat``_arready;  \
                                                      \
  assign m_axilite_``pat``_rready   = __intf.rready;  \
  assign __intf.rvalid  = m_axilite_``pat``_rvalid;   \
  assign __intf.rdata   = m_axilite_``pat``_rdata;    \
  assign __intf.rresp   = m_axilite_``pat``_rresp;
`define AXILITE_ASSIGN_SLAVE_TO_FLAT(pat, __intf)     \
  assign __intf.awvalid = s_axilite_``pat``_awvalid;  \
  assign __intf.awaddr  = s_axilite_``pat``_awaddr;   \
  assign s_axilite_``pat``_awready = __intf.awready;  \
                                                      \
  assign __intf.wvalid   = s_axilite_``pat``_wvalid;  \
  assign __intf.wdata    = s_axilite_``pat``_wdata;   \
  assign __intf.wstrb    = s_axilite_``pat``_wstrb;   \
  assign s_axilite_``pat``_wready = __intf.wready;    \
                                                      \
  assign __intf.bready = s_axilite_``pat``_bready;    \
  assign s_axilite_``pat``_bvalid = __intf.bvalid;    \
  assign s_axilite_``pat``_bresp = __intf.bresp;      \
                                                      \
  assign __intf.arvalid = s_axilite_``pat``_arvalid;  \
  assign __intf.araddr = s_axilite_``pat``_araddr;    \
  assign s_axilite_``pat``_arready = __intf.arready;  \
                                                      \
  assign __intf.rready = s_axilite_``pat``_rready;    \
  assign s_axilite_``pat``_rvalid = __intf.rvalid;    \
  assign s_axilite_``pat``_rdata = __intf.rdata;      \
  assign s_axilite_``pat``_rresp = __intf.rresp;

////////////////////////////////////////////////////////////////////////////////////////
`define __AXIS_TO(__opt_as, __lhs, __lhs_sep, __rhs, __rhs_sep)     \
  __opt_as __lhs``__lhs_sep``tdata   = __rhs``__rhs_sep``tdata;     \
  __opt_as __lhs``__lhs_sep``tlast   = __rhs``__rhs_sep``tlast;

`define AXIS_ASSIGN(slv, mst)            \
  `__AXIS_TO(slv, ., mst, .)             \
  assign mst.tready  = slv.tready;       \
  assign slv.tvalid  = mst.tvalid;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Assigning an interface from channel or request/response structs outside a process.
//
// The channel macros `AXIS_ASSIGN_FROM_XX(axis_if, xx_struct)` assign the payload signals of the
// `axis_if` interface from the signals in `xx_struct`.
`define AXIS_ASSIGN_TO(axis_struct, axis_if)         \
  `__AXIS_TO(assign, axis_struct, _, axis_if, .)     \
`define AXIS_ASSIGN_FROM(axis_if, axis_struct)       \
  `__AXIS_TO(assign, axi_if, ., axis_struct, _)

/////////////////////////////////////////////////////////////////////////////////////
// Macros for assigning flattened AXIS ports to req/resp AXIS structs
// Flat AXIS ports are required by the Vivado IP Integrator.
// Vivado naming convention is followed.
//
// Usage Example:
// `AXIS_ASSIGN_MASTER_TO_FLAT("my_bus", my_req_struct, my_rsp_struct)
`define AXIS_ASSIGN_MASTER_TO_FLAT(pat, __intf)    \
  assign m_axis_``pat``_tdata   = __intf.tdata;    \
  assign m_axis_``pat``_tvalid  = __intf.tvalid;   \
  assign __intf.tready = m_axis_``pat``_tready;    \
  assign m_axis_``pat``_tlast   = __intf.tlast;
`define AXIS_ASSIGN_SLAVE_TO_FLAT(pat, __intf)     \
  assign __intf.tdata  = s_axis_``pat``_tdata  ;   \
  assign __intf.tvalid = s_axis_``pat``_tvalid ;   \
  assign s_axis_``pat``_tready = __intf.tready;    \
  assign __intf.tlast  = s_axis_``pat``_tlast  ;

`endif
