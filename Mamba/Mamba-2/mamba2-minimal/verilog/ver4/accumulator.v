// N = 128 기준으로 생성
// N = 16 or 128에 따라 다른 구조를 사용하도록 수동 교체 필요
// 나중에 이것도 병렬 fp16_adder_tree_128 모듈 사용하도록 개선 가능할것으로 보임
// 이거 지금 N=32 인 상태임 그런데 이거 안건드렸는데 왜 되는거지????
// 원인 확인해보기

module accumulator #(
    parameter B  = 1,
    parameter H  = 4,
    parameter P  = 4,
    parameter N  = 128,
    parameter DW = 16,
    parameter ADD_LAT = 11,
    parameter TREE_DEPTH = 7,    // N = 128일시 교체
    // parameter TREE_DEPTH = 5,                      // log2(N)
    parameter TREE_LAT = ADD_LAT * TREE_DEPTH      // total latency through the tree
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   start,
    input  wire [B*H*P*N*DW-1:0]  hC_flat,
    output wire [B*H*P*DW-1:0]    hC_sum_flat,
    output reg                    done
);

    // Unpack hC_flat
    wire [DW-1:0] hC [0:B*H*P*N-1];
    reg  [DW-1:0] sum [0:B*H*P-1];
    genvar g;
    generate
        for (g = 0; g < B*H*P*N; g = g + 1) begin
            assign hC[g] = hC_flat[(g+1)*DW-1 -: DW];
        end
        for (g = 0; g < B*H*P; g = g + 1) begin
            assign hC_sum_flat[(g+1)*DW-1 -: DW] = sum[g];
        end
    endgenerate

    // Internal control
    localparam TOTAL_TILES = B*H*P;
    reg [$clog2(TOTAL_TILES):0] tile_index_in;
    reg [$clog2(TOTAL_TILES):0] tile_index_out;
    reg [DW-1:0] tree_input [0:N-1];
    reg [N*DW-1:0] tree_input_flat;
    integer i;

    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            tree_input_flat[(i+1)*DW-1 -: DW] = tree_input[i];
        end
    end

    // Tile feeding
    reg feeding;
    reg valid_in;

    // Adder tree instantiation
    wire [DW-1:0] tree_sum;
    wire valid_out;
    fp16_adder_tree_128 #(.DW(DW), .N(N)) tree (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .in_flat(tree_input_flat),
        .sum(tree_sum),
        .valid_out(valid_out)
    );

    // Write output when valid_out is high
    reg done_pulse;
    reg done_pulse_ack;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tile_index_out <= 0;
            done <= 0;
            done_pulse <= 0;
            done_pulse_ack <= 0;
            feeding <= 0;
            tile_index_in <= 0;
            valid_in <= 0;
        end else begin
            if (start) begin
                // 새로운 run 시작 시 초기화
                tile_index_out   <= 0;
                tile_index_in    <= 0;
                feeding          <= 1;
                valid_in         <= 0;
                done             <= 0;
                done_pulse       <= 0;
                done_pulse_ack   <= 0;
            end else begin
                // 2) 트리 출력 수거 (경계 가드)
                if (valid_out && (tile_index_out < TOTAL_TILES)) begin
                    sum[tile_index_out] <= tree_sum;
                    tile_index_out      <= tile_index_out + 1;
                end

                // 3) 입력 피딩
                if (feeding && (tile_index_in < TOTAL_TILES)) begin
                    for (integer i2 = 0; i2 < N; i2 = i2 + 1)
                        tree_input[i2] <= hC[tile_index_in*N + i2];
                    valid_in       <= 1;
                    tile_index_in  <= tile_index_in + 1;
                end else begin
                    valid_in <= 0;
                end

                // 4) 피딩 종료
                if (feeding && (tile_index_in == TOTAL_TILES)) begin
                    feeding <= 0;
                end

                // 5) done 펄스 생성 (모든 출력 배출 + 파이프라인 비움)
                if (!done_pulse_ack &&
                    (tile_index_out == TOTAL_TILES) &&
                    !feeding && !valid_in && !valid_out) begin
                    done_pulse     <= 1;
                    done_pulse_ack <= 1;
                end else begin
                    done_pulse <= 0;
                end

                done <= done_pulse; // 1사이클 펄스
            end
        end
    end

endmodule


// N = 128 FP16 adder tree using fp16_add_wrapper
// Assumes latency-insensitive, pipelined valid_in → valid_out structure
// 128일시 N = 128로 교체, 주석 해제 등

module fp16_adder_tree_128 #(parameter DW=16, N=128)(    // N에 따라 교체
    input  wire clk,
    input  wire rst,
    input  wire valid_in,
    input  wire [N*DW-1:0] in_flat,  // 128 inputs, each DW-bit
    output wire [DW-1:0]     sum,      // final accumulated sum
    output wire              valid_out
);

    // Unpack input
    wire [DW-1:0] in_level0 [0:127];
    // Declare wires for each tree level
    wire [DW-1:0] level1 [0:63];
    wire [DW-1:0] level2 [0:31];
    wire [DW-1:0] level3 [0:15];
    wire [DW-1:0] level4 [0:7];
    wire [DW-1:0] level5 [0:3];
    wire [DW-1:0] level6 [0:1];
    wire [DW-1:0] level7;
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : UNPACK
            assign in_level0[i] = in_flat[(i+1)*DW-1 -: DW]; // 128일시 교체
            // assign level2[i] = in_flat[(i+1)*DW-1 -: DW];
        end
    endgenerate

    

    wire valid_l1, valid_l2, valid_l3, valid_l4, valid_l5, valid_l6, valid_l7;

    assign valid_l1 = valid_in;  // 128일시 교체
    // assign valid_l3 = valid_in;

    // 128일시 주석 해제
    // Level 1 (128 → 64)
    generate
        for (i = 0; i < 64; i = i + 1) begin : L1
            fp16_add_wrapper add (
                .clk(clk),
                .a(in_level0[2*i]),
                .b(in_level0[2*i+1]),
                .valid_in(valid_l1),
                .result(level1[i]),
                .valid_out(valid_l2)
            );
        end
    endgenerate

    // Level 2 (64 → 32)
    generate
        for (i = 0; i < 32; i = i + 1) begin : L2
            fp16_add_wrapper add (
                .clk(clk),
                .a(level1[2*i]),
                .b(level1[2*i+1]),
                .valid_in(valid_l2),
                .result(level2[i]),
                .valid_out(valid_l3)
            );
        end
    endgenerate

    // Level 3 (32 → 16)
    generate
        for (i = 0; i < 16; i = i + 1) begin : L3
            fp16_add_wrapper add (
                .clk(clk),
                .a(level2[2*i]),
                .b(level2[2*i+1]),
                .valid_in(valid_l3),
                .result(level3[i]),
                .valid_out(valid_l4)
            );
        end
    endgenerate

    // Level 4 (16 → 8)
    generate
        for (i = 0; i < 8; i = i + 1) begin : L4
            fp16_add_wrapper add (
                .clk(clk),
                .a(level3[2*i]),
                .b(level3[2*i+1]),
                .valid_in(valid_l4),
                .result(level4[i]),
                .valid_out(valid_l5)
            );
        end
    endgenerate

    // Level 5 (8 → 4)
    generate
        for (i = 0; i < 4; i = i + 1) begin : L5
            fp16_add_wrapper add (
                .clk(clk),
                .a(level4[2*i]),
                .b(level4[2*i+1]),
                .valid_in(valid_l5),
                .result(level5[i]),
                .valid_out(valid_l6)
            );
        end
    endgenerate

    // Level 6 (4 → 2)
    generate
        for (i = 0; i < 2; i = i + 1) begin : L6
            fp16_add_wrapper add (
                .clk(clk),
                .a(level5[2*i]),
                .b(level5[2*i+1]),
                .valid_in(valid_l6),
                .result(level6[i]),
                .valid_out(valid_l7)
            );
        end
    endgenerate

    // Level 7 (2 → 1)
    fp16_add_wrapper final_add (
        .clk(clk),
        .a(level6[0]),
        .b(level6[1]),
        .valid_in(valid_l7),
        .result(level7),
        .valid_out(valid_out)
    );

    assign sum = level7;

endmodule
