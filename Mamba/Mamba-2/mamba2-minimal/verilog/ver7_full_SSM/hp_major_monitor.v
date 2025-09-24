`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// hp_major_monitor.v  (pure Verilog, no SystemVerilog)
//  - ev_valid/ev_addr 스트림의 "연속 run", "gap", "가상 burst(64-beat cap, BUS_BYTES 정렬, 4KB 경계)"를 계측
//  - CSV: hp_major_wr.csv 로 덤프 (파일명 고정; 필요하면 바꿔도 됨)
// -----------------------------------------------------------------------------
module hp_major_monitor #(
  parameter DATA_BYTES   = 2,     // FP16 = 2 bytes
  parameter BUS_BYTES    = 32,    // 256b = 32B/beat (추정용)
  parameter RUN_HIST_MAX = 64,     // run 길이 히스토그램 최대 bin
  parameter [8*64-1:0] FNAME = "hp_major_wr.csv", // 파일명
  parameter [8*8-1:0]  TAG   = "WR"               // 태그
)(
  input        clk,
  input        rstn,
  input        ev_valid,          // 이벤트(쓰기 1개 또는 읽기 요청 1개) 발생 펄스
  input [63:0] ev_addr,           // 바이트 주소(또는 elem_idx*DATA_BYTES)
  input        stream_start,      // 옵션: 경계 펄스 (없으면 1'b0)
  input        stream_end         // 옵션: 경계 펄스 (없으면 1'b0)
);

  // 상태
  reg        in_run;
  reg [63:0] prev_addr;
  reg [63:0] run_start_addr;
  integer    run_len;

  // 누적 카운터
  integer total_events;
  integer gap_cnt;
  integer sum_run_len;
  integer run_cnt;
  integer max_run_len;
  integer virt_burst_cnt;
  integer virt_bytes_total;
  integer unaligned_bus_cnt;

  // 히스토그램 (0 bin은 미사용; 1..RUN_HIST_MAX)
  integer hist_run_len [0:RUN_HIST_MAX];
  integer hist_run_len_over;

  // 가상 burst 개수 추정 함수 (64-beat cap, BUS_BYTES 정렬, 4KB 경계 고려)
  function integer estimate_bursts;
    input [63:0] start_addr;
    input integer run_elems;
    integer bursts;
    integer remain;        // bytes 남은 양
    integer align_gap;     // BUS_BYTES 정렬까지 필요한 바이트
    integer b4kb;          // 4KB 경계까지 남은 바이트
    integer bmax;          // 64-beat * BUS_BYTES
    integer bcap;          // 이번 전송에서 보낼 수 있는 최대 바이트
    integer chunk;         // 이번에 소진할 바이트
    reg [63:0] addr;       // 진행 중 주소 (bytes)
  begin
    bursts = 0;
    remain = run_elems * DATA_BYTES;
    addr   = start_addr;
    bmax   = 64 * BUS_BYTES;

    while (remain > 0) begin
      // (1) 버스 정렬 맞추기
      if ((addr % BUS_BYTES) != 0) begin
        align_gap = BUS_BYTES - (addr % BUS_BYTES);
        chunk     = (remain < align_gap) ? remain : align_gap;
        bursts    = bursts + 1;
        addr      = addr + chunk;
        remain    = remain - chunk;
      end else begin
        // (2) 4KB 경계/64-beat cap 고려하여 한 번에 보낼 chunk 결정
        b4kb = 4096 - (addr % 4096);
        bcap = (b4kb < bmax) ? b4kb : bmax;
        chunk = (remain < bcap) ? remain : bcap;
        bursts = bursts + 1;
        addr   = addr + chunk;
        remain = remain - chunk;
      end
    end
    estimate_bursts = bursts;
  end
  endfunction

  // run 종료 처리
  task close_run;
    integer rl;
  begin
    if (!in_run || run_len == 0) begin
      // nothing
    end else begin
      rl = run_len;

      // 히스토그램
      if (rl <= RUN_HIST_MAX) hist_run_len[rl] = hist_run_len[rl] + 1;
      else                    hist_run_len_over = hist_run_len_over + 1;

      // 최대/평균용 누적
      if (rl > max_run_len) max_run_len = rl;
      sum_run_len      = sum_run_len + rl;
      run_cnt          = run_cnt + 1;

      // 가상 burst 추정
      virt_burst_cnt   = virt_burst_cnt + estimate_bursts(run_start_addr, rl);
      virt_bytes_total = virt_bytes_total + (rl * DATA_BYTES);

      // reset run
      in_run  = 1'b0;
      run_len = 0;
    end
  end
  endtask

  integer i;

  // 초기화
  initial begin
    in_run = 1'b0; run_len = 0; prev_addr = 64'd0; run_start_addr = 64'd0;
    total_events = 0; gap_cnt = 0; sum_run_len = 0; run_cnt = 0; max_run_len = 0;
    virt_burst_cnt = 0; virt_bytes_total = 0; unaligned_bus_cnt = 0;
    for (i = 0; i <= RUN_HIST_MAX; i = i + 1) hist_run_len[i] = 0;
    hist_run_len_over = 0;
  end

  // 메인 로직
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      in_run <= 1'b0; run_len <= 0; prev_addr <= 64'd0; run_start_addr <= 64'd0;
      total_events <= 0; gap_cnt <= 0; sum_run_len <= 0; run_cnt <= 0; max_run_len <= 0;
      virt_burst_cnt <= 0; virt_bytes_total <= 0; unaligned_bus_cnt <= 0;
      for (i = 0; i <= RUN_HIST_MAX; i = i + 1) hist_run_len[i] <= 0;
      hist_run_len_over <= 0;
    end else begin
      if (stream_start) close_run();

      if (ev_valid) begin
        total_events <= total_events + 1;

        if (!in_run) begin
          // 새 run 시작
          in_run         <= 1'b1;
          run_len        <= 1;
          run_start_addr <= ev_addr;
          if ((ev_addr % BUS_BYTES) != 0) unaligned_bus_cnt <= unaligned_bus_cnt + 1;
        end else begin
          // 연속성 검사
          if (ev_addr == (prev_addr + DATA_BYTES)) begin
            run_len <= run_len + 1;
          end else begin
            // run 끊김
            close_run();
            gap_cnt <= gap_cnt + 1;

            // 새 run 시작
            in_run         <= 1'b1;
            run_len        <= 1;
            run_start_addr <= ev_addr;
            if ((ev_addr % BUS_BYTES) != 0) unaligned_bus_cnt <= unaligned_bus_cnt + 1;
          end
        end

        prev_addr <= ev_addr;
      end

      if (stream_end) close_run();
    end
  end

  // CSV 덤프 (파일명을 파라미터 사용)
  task dump_report;
    integer fd;
    real avg_run, trans_per_kb;
    integer i;
  begin
    close_run(); // 열린 run 마감
    avg_run      = (run_cnt > 0) ? (1.0 * sum_run_len / run_cnt) : 0.0;
    trans_per_kb = (virt_bytes_total > 0) ? (1.0 * virt_burst_cnt / (virt_bytes_total / 1024.0)) : 0.0;

    fd = $fopen(FNAME, "w");
    $fdisplay(fd, "TAG,total_events,run_cnt,avg_run,max_run,gap_cnt,virt_bursts,virt_kB,trans_per_kB,unaligned_starts");
    $fdisplay(fd, "%0s,%0d,%0d,%f,%0d,%0d,%0d,%f,%f,%0d",
                 TAG, total_events, run_cnt, avg_run, max_run_len, gap_cnt,
                 virt_burst_cnt, (1.0*virt_bytes_total)/1024.0, trans_per_kb, unaligned_bus_cnt);
    $fdisplay(fd, "RUN_LEN,COUNT");
    for (i = 1; i <= RUN_HIST_MAX; i = i + 1)
      if (hist_run_len[i] > 0) $fdisplay(fd, "%0d,%0d", i, hist_run_len[i]);
    if (hist_run_len_over > 0) $fdisplay(fd, "GT_%0d,%0d", RUN_HIST_MAX, hist_run_len_over);
    $fclose(fd);
    $display("hp_major_monitor: report dumped to %0s", FNAME);
  end
  endtask
endmodule
