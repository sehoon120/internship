module pwl8_rom #(
    parameter integer DW=16
)(
    input  wire       clk,
    input  wire       rstn,
    input  wire       valid_i,
    input  wire [2:0] seg_i,
    output reg  [DW-1:0] y0_o,
    output reg  [DW-1:0] slope_o,
    output reg          valid_o
);
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin y0_o<=0; slope_o<=0; valid_o<=1'b0; end
        else begin
            valid_o <= valid_i;
            case (seg_i)
                3'd0: begin y0_o <= 16'h3C00; slope_o <= 16'h39CB; end  // y0=2^0=1.0
                3'd1: begin y0_o <= 16'h3C5D; slope_o <= 16'h3A51; end
                3'd2: begin y0_o <= 16'h3CC2; slope_o <= 16'h3AE3; end
                3'd3: begin y0_o <= 16'h3D30; slope_o <= 16'h3B83; end
                3'd4: begin y0_o <= 16'h3DA8; slope_o <= 16'h3C19; end
                3'd5: begin y0_o <= 16'h3E2B; slope_o <= 16'h3C77; end
                3'd6: begin y0_o <= 16'h3EBA; slope_o <= 16'h3CDF; end
                3'd7: begin y0_o <= 16'h3F56; slope_o <= 16'h3D50; end
            endcase
        end
    end
endmodule
