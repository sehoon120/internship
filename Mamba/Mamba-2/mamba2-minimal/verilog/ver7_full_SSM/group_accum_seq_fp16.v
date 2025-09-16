// group_accum_seq_fp16 : accumulate TILES_PER_GROUP hp-vectors into one
// - Accepts one y_tile per cycle (valid_i), no backpressure
// - Internally buffers tiles and reduces using hp_add_vec (throughput=1)
// - Pulses valid_o with the final sum after the last tile of the group drains
module group_accum_seq_fp16 #(
  parameter integer DW               = 16,
  parameter integer H_TILE           = 1,
  parameter integer P_TILE           = 1,
  parameter integer TILES_PER_GROUP  = 8  // N_TOTAL / N_TILE
)(
  input  wire                              clk,
  input  wire                              rstn,

  input  wire                              valid_i,      // y_tile_i valid (1/cycle ok)
  input  wire [H_TILE*P_TILE*DW-1:0]       y_tile_i,     // (hp)
  input  wire                              last_i,       // assert with valid_i on last tile of the group

  output reg  [H_TILE*P_TILE*DW-1:0]       group_sum_o,  // (hp) final group sum
  output reg                               valid_o       // 1-pulse when group_sum_o is valid
);
  localparam integer HPW = H_TILE*P_TILE*DW;

  // Simple FIFO for up to TILES_PER_GROUP tiles (each word is (hp) bus)
  reg [HPW-1:0] fifo_mem [0:TILES_PER_GROUP-1];
  integer head, tail, count;

  // Track we've seen the last tile marker for this group
  reg last_seen;

  // Running group sum and state
  reg                 sum_valid;           // 0: no sum yet, 1: sum in group_sum_r
  reg [HPW-1:0]       group_sum_r;

  // Adder control
  reg                 add_fire;            // single-cycle fire to hp_add_vec
  reg [HPW-1:0]       add_a, add_b;
  wire[HPW-1:0]       add_y;
  wire                add_v;

  // Instantiate hp vector adder
  hp_add_vec #(.DW(DW), .H_TILE(H_TILE), .P_TILE(P_TILE)) u_hpadd (
    .clk     (clk),
    .rstn    (rstn),
    .valid_i (add_fire),
    .a_i     (add_a),
    .b_i     (add_b),
    .y_o     (add_y),
    .valid_o (add_v)
  );

  // Book-keeping: push into FIFO on valid_i
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      head <= 0; tail <= 0; count <= 0; last_seen <= 1'b0;
    end else begin
      if (valid_i) begin
        fifo_mem[tail] <= y_tile_i;
        tail <= tail + 1;
        count <= count + 1;
        if (last_i) last_seen <= 1'b1;
      end
      // pop happens when we explicitly do it below
    end
  end

  // Simple sequencer:
  // 1) If no sum yet and fifo has data -> pop one and load group_sum_r
  // 2) Else if sum exists and fifo has data and adder is idle -> fire add of (sum, next)
  // 3) When add result valid -> update group_sum_r
  // 4) When last_seen && fifo empty && adder idle && sum_valid -> raise valid_o 1-cycle
  reg adder_busy;      // tracks an in-flight add
  reg pop_now;         // internal pop strobes
  reg [HPW-1:0] pop_data;

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      sum_valid   <= 1'b0;
      group_sum_r <= {HPW{1'b0}};
      add_fire    <= 1'b0;
      add_a       <= {HPW{1'b0}};
      add_b       <= {HPW{1'b0}};
      adder_busy  <= 1'b0;
      pop_now     <= 1'b0;
      pop_data    <= {HPW{1'b0}};
      group_sum_o <= {HPW{1'b0}};
      valid_o     <= 1'b0;
    end else begin
      // defaults
      add_fire <= 1'b0;
      pop_now  <= 1'b0;
      valid_o  <= 1'b0;

      // consume adder result
      if (add_v) begin
        group_sum_r <= add_y;
        adder_busy  <= 1'b0;
        sum_valid   <= 1'b1;
      end

      // launch new add if possible
      if (!adder_busy && sum_valid && (count > 0)) begin
        // pop next tile
        pop_data <= fifo_mem[head];
        head     <= head + 1;
        count    <= count - 1;
        pop_now  <= 1'b1;

        // fire adder this cycle with (sum, popped)
        add_a    <= group_sum_r;
        add_b    <= fifo_mem[head];  // safe because head used *before* increment in this cycle?
        // NOTE: safer to use pop_data; register order fix below
        add_fire <= 1'b1;
        adder_busy <= 1'b1;
      end

      // if no sum yet, bootstrap with first element
      if (!sum_valid && (count > 0)) begin
        // pop one into group_sum_r (no adder)
        group_sum_r <= fifo_mem[head];
        head        <= head + 1;
        count       <= count - 1;
        sum_valid   <= 1'b1;
      end

      // group done condition
      if (last_seen && (count == 0) && !adder_busy && sum_valid) begin
        group_sum_o <= group_sum_r;
        valid_o     <= 1'b1;
        // prepare for next group
        sum_valid   <= 1'b0;
        last_seen   <= 1'b0;
      end
    end
  end

endmodule
