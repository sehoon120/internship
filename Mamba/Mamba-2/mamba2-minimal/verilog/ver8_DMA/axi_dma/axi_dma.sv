// 이 모듈이 top
// 그런데 가속기와의 연결은 상위 모듈 하나 더 생성해서 연결. 여기서는 port만 만들고

//           Top Module
// ┌────────────────────────────────────────┐
// │                                        │
// │  ┌───────────┐        ┌────────────┐   │
// │  │   CPU     │        │ Accelerator│   │
// │  └────▲──────┘        └────▲───────┘   │
// │       │ AXI-Lite           │AXIS       │
// │       │ (option)           │           │
// │       │                    │           │
// │  ┌────▼──────┐  AXIS  ┌────▼────────┐  │
// │  │  axi_dma  │───────►│mm2s  (in)   │  │
// │  │ (DMA IP)  │◄───────│s2mm  (out)  │  │
// │  └────▲──────┘        └─────────────┘  │
// │       │ AXI-MM (master)                │
// │  ┌────▼──────┐                         │
// │  │  Memory   │                         │
// │  └───────────┘                         │
// │                                        │
// └────────────────────────────────────────┘


`timescale 1 ns / 1 ps
module axi_dma #(
    parameter NUM_RD_MO_BUF = 4,  // (메모리 명령을 동시에 여러 개 날릴 수 있게 하는 큐/슬롯 수)
    parameter NUM_WR_MO_BUF = 4,
    parameter integer ADDR_WIDTH = 64,  // AXI-MM 주소/data 폭
    parameter integer DATA_WIDTH = 256,
    // S2MM 경로에서 AXIS 입력을 임시 저장하는 FIFO 깊이. 
    // 가속기와 메모리 사이의 속도/버스트 미스매치를 흡수.
    parameter integer WR_AXIS_FIFO_DEPTH = 16,  
    // Default types; override as needed.
    type data_t = axi_pkg::data_256_t,
    type addr_t = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    `include "assign.svh"  // 이 파일도 제공받을 수 있을까?

    // 가속기와 연결
    // AXI config, Start AXI transactions (read / write)
    input  addr_t        axi_read_start_addr,   // r
    input  addr_t        axi_write_start_addr,  // w
    input  logic  [31:0] axi_read_length,       // r
    input  logic  [31:0] axi_write_length,      // w
    input  logic         init_read,             // r
    input  logic         init_write,            // w
    output logic         axi_write_start_ready, // w
    output logic         axi_write_start_valid, // w
    output logic         axi_read_start_ready,  // r
    output logic         axi_read_start_valid,  // r
    output logic         axi_dma_wr_idle,       // w
    output logic         axi_dma_rd_idle,       // r

    // AXIS Interface
    axis.master __axis_mm2s,  // (master): 메모리→가속기 (MM2S, read path)
    axis.slave  __axis_s2mm,  // (slave): 가속기→메모리 (S2MM, write path)

    // AXI MM Interface
    // 이 모듈 바깥에서 만들어진 aximm 인터페이스 인스턴스를, master modport 뷰로 받는 포트
    aximm.master __aximm,  // external memory와 연결을 위함
    input wire m_axi_aclk,
    input wire m_axi_aresetn
);

    aximm_rd __aximm_rd ();  // 읽기 채널(AR/R)만 가진 인터페이스 인스턴스
    `AXI_ASSIGN_AR(assign, __aximm, __aximm_rd)  // 목적지: memory, 출발지: DMA
    `AXI_ASSIGN_R(assign, __aximm_rd, __aximm)
    // init_read + start_addr + byte_length를 입력으로 받아서, 
    // 메모리(AXI-MM Read)에서 데이터를 읽어와 AXI-Stream(__axis_mm2s)으로 뿜어주는 엔진
    axi_dma_rd #(  
        .NUM_MO_BUF(NUM_RD_MO_BUF),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .data_t    (data_t),
        .addr_t    (addr_t),
        .trans_t   (trans_t)
    ) u_axi_dma_rd (
        .init_read      (init_read),
        .axi_start_addr (axi_read_start_addr),
        .axi_byte_length(axi_read_length),
        .axi_start_ready(axi_read_start_ready),
        .axi_start_valid(axi_read_start_valid),
        .axi_idle       (axi_dma_rd_idle),
        .__axis_mm2s    (__axis_mm2s),
        .clk            (m_axi_aclk),
        .rstn           (m_axi_aresetn),
        .__aximm_rd     (__aximm_rd)
    );

    aximm_wr __aximm_wr ();
    `AXI_ASSIGN_AW(assign, __aximm, __aximm_wr)
    `AXI_ASSIGN_W(assign, __aximm, __aximm_wr)
    `AXI_ASSIGN_B(assign, __aximm_wr, __aximm)
    axi_dma_wr #(
        .NUM_MO_BUF   (NUM_WR_MO_BUF),
        .ADDR_WIDTH   (ADDR_WIDTH),
        .DATA_WIDTH   (DATA_WIDTH),
        .WR_FIFO_DEPTH(WR_AXIS_FIFO_DEPTH),
        .data_t       (data_t),
        .addr_t       (addr_t),
        .trans_t      (trans_t)
    ) u_axi_dma_wr (
        .init_write     (init_write),
        .axi_start_addr (axi_write_start_addr),
        .axi_byte_length(axi_write_length),
        .axi_start_ready(axi_write_start_ready),
        .axi_start_valid(axi_write_start_valid),
        .axi_idle       (axi_dma_wr_idle),
        .__axis_s2mm    (__axis_s2mm),
        .clk            (m_axi_aclk),
        .rstn           (m_axi_aresetn),
        .__aximm_wr     (__aximm_wr)
    );
endmodule
