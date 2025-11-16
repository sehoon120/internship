`timescale 1ns / 1ps

// maybe TB용 더미 모듈

// axi_dma_reg_driver
// 용도: DMA에 시작 주소/길이 세팅 + init 펄스를 넣어주는 테스트 시퀀서.
// 성격: 생산 환경의 AXI-Lite 레지스터 제어를 임시로 대체하는 TB용 드라이버.

// axis_mm2s_receiver
// 용도: MM2S 스트림을 받아주는 더미 소비자(tready=1로 받아줌).
// 성격: 가속기(입력)나 실제 다운스트림 모듈이 없을 때 싱크 역할을 하는 TB용.

// axis_s2mm_driver
// 용도: S2MM 스트림에 데이터를 만들어 밀어주는 더미 소스(카운터 패턴).
// 성격: 가속기(출력)나 실제 업스트림 모듈 대신 소스 역할을 하는 TB용.

// AXI DMA Driver: axi_dma signal driver for axi_dma operation
module axi_dma_reg_driver #(
    // Default types; override as needed.
    type addr_t  = axi_pkg::addr_64_t,
    type trans_t = axi_pkg::trans_64_t
) (
    input logic clk,
    input logic rstn,

    output addr_t        axi_read_start_addr,
    output addr_t        axi_write_start_addr,
    output logic  [31:0] axi_read_length,
    output logic  [31:0] axi_write_length,
    output logic         init_read,
    output logic         init_write
);

    logic [3:0] counter;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            counter <= 0;
            axi_read_start_addr <= 0;
            axi_write_start_addr <= 0;
            axi_read_length <= 0;
            axi_write_length <= 0;
            init_read <= 0;
            init_write <= 0;
        end else begin
            if (counter == 8) begin
                init_read <= 1;
                init_write <= 1;
                counter <= counter + 1;
            end else if (counter == 9) begin
                init_read  <= 0;
                init_write <= 0;
            end else begin
                axi_read_start_addr <= 0;
                axi_write_start_addr <= 0;
                axi_read_length <= 16;
                axi_write_length <= 16;

                counter <= counter + 1;
            end
        end
    end

endmodule

// S2MM Receiver: Minimal receiver for AXIS write stream
module axis_mm2s_receiver #(  // start==1이면 tready=1로 고정 → 들어오는 스트림을 무제한 수용.
    parameter int DATA_WIDTH = 256
) (
    input logic clk,
    input logic rstn,
    input logic start,

    axis.slave __axis_mm2s
);

    always_ff @(posedge clk) begin
        if (!rstn) begin
            __axis_mm2s.tready <= 1'b0;
        end else begin
            if (start) begin
                __axis_mm2s.tready <= 1'b1;
            end
        end
    end
endmodule


// MM2S Driver: Minimal driver for AXIS read stream
module axis_s2mm_driver #(
    parameter int DATA_WIDTH = 256,
    parameter int NUM_WORDS  = 16
) (
    input  logic clk,
    input  logic rstn,
    input  logic start,
    output logic done_drive,

    axis.master __axis_s2mm
);

    logic [31:0] word_counter;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            __axis_s2mm.tvalid <= 0;
            __axis_s2mm.tlast <= 0;
            __axis_s2mm.tdata <= 0;
            word_counter <= 0;
            done_drive <= 0;
        end else begin
            if (start) begin
                if (word_counter < NUM_WORDS) begin
                    __axis_s2mm.tvalid <= 1;
                    __axis_s2mm.tdata  <= word_counter;
                    __axis_s2mm.tlast  <= (word_counter == NUM_WORDS - 1);

                    if (__axis_s2mm.tvalid && __axis_s2mm.tready) begin
                        word_counter <= word_counter + 1;
                    end
                end else begin
                    done_drive <= 1;
                    __axis_s2mm.tvalid <= 0;
                    __axis_s2mm.tlast <= 0;
                end
            end
        end
    end
endmodule
