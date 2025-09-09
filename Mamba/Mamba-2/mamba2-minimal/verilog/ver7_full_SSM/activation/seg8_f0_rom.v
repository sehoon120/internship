module seg8_f0_rom #(parameter integer DW=16)(
    input  wire       clk, input wire rstn,
    input  wire       valid_i,
    input  wire [2:0] seg_i,
    output reg  [DW-1:0] f0_o,
    output reg          valid_o
);
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin f0_o<=0; valid_o<=1'b0; end
        else begin
            valid_o <= valid_i;
            case(seg_i)
              3'd0: f0_o <= 16'h0000; // 0.0
              3'd1: f0_o <= 16'h2E00; // ~0.125
              3'd2: f0_o <= 16'h3200; // ~0.25
              3'd3: f0_o <= 16'h3400; // ~0.375
              3'd4: f0_o <= 16'h3800; // 0.5
              3'd5: f0_o <= 16'h3980; // ~0.625
              3'd6: f0_o <= 16'h3A80; // ~0.75
              3'd7: f0_o <= 16'h3B40; // ~0.875
            endcase
        end
    end
endmodule
