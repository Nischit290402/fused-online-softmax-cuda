### Table A. Median time per call (µs, CUDA events, mean of the two sweep runs; lower is better)

| S | composed | torch.softmax | B1 (3-pass) | B2 (online, 2-pass) | B3 (row in smem) |
|---|---|---|---|---|---|
| 1,024 | 164.5 | 27.9 | 37.3 | 46.5 | 58.1 |
| 2,048 | 291.0 | 81.5 | 126.9 | 136.6 | 153.6 |
| 4,096 | 1,090.1 | 427.0 | 443.2 | 504.0 | 471.0 |
| 8,192 | 4,494.7 | 1,558.8 | 2,317.3 | 2,235.8 | 2,083.7 |
| 16,384 | 17,568.0 | 4,557.7 | 11,238.1 | 9,322.1 | 16,173.6 |
| 32,768 | 141,172.7 | 17,978.8 | 45,424.3 | 35,647.8 | skip (row > smem) |

### Table B. Effective bandwidth (GB/s, 4·S² bytes / time; theoretical peak 352 GB/s)

| S | composed | torch.softmax | B1 | B2 | B3 | best custom / torch |
|---|---|---|---|---|---|---|
| 1,024 | 26 | 151 | 112 | 90 | 72 | 75% |
| 2,048 | 58 | 206 | 132 | 123 | 109 | 64% |
| 4,096 | 62 | 157 | 151 | 133 | 142 | 96% |
| 8,192 | 60 | 172 | 116 | 120 | 129 | 75% |
| 16,384 | 61 | 236 | 96 | 115 | 66 | 49% |
| 32,768 | 30 | 239 | 95 | 120 | skip | 50% |

### Table C. Nsight Compute (ncu time is at locked base clocks, so it is slower than Table A)

| variant | S | block | kernels/call | DRAM B/elem | model B/elem | L2 hit % | DRAM % peak | SM % peak | occupancy % | regs/thread | time (µs) | top stall reasons (warps per issue) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| composed | 8,192 | 256 | 5 | 15.94 | 16 | 39 | 54 | 44 | 93 | 33 | 5,669 | long_scoreboard 10.48; wait 2.13; not_selected 1.15 |
| torch.softmax | 2,048 | 256 | 1 | 3.79 | 4 | 50 | 50 | 45 | 66 | 80 | 92 | long_scoreboard 3.27; wait 2.00; short_scoreboard 1.06 |
| torch.softmax | 8,192 | 256 | 1 | 3.99 | 4 | 50 | 35 | 57 | 98 | 35 | 2,167 | not_selected 3.27; math_pipe_throttle 2.11; wait 1.97 |
| torch.softmax | 16,384 | 256 | 1 | 4.00 | 4 | 50 | 50 | 49 | 97 | 41 | 6,136 | not_selected 3.23; long_scoreboard 2.49; math_pipe_throttle 2.38 |
| B1 | 2,048 | 256 | 1 | 3.75 | 8 | 50 | 31 | 54 | 97 | 22 | 147 | long_scoreboard 6.15; wait 2.28; short_scoreboard 1.56 |
| B1 | 8,192 | 256 | 1 | 4.12 | 8 | 71 | 25 | 34 | 97 | 22 | 3,194 | long_scoreboard 15.56; wait 2.20; not_selected 1.01 |
| B1 | 16,384 | 256 | 1 | 7.99 | 8 | 32 | 41 | 29 | 99 | 22 | 14,787 | long_scoreboard 21.20; wait 2.18; selected 1.00 |
| B2 | 2,048 | 256 | 1 | 3.77 | 6 | 51 | 26 | 71 | 96 | 24 | 176 | long_scoreboard 3.60; wait 1.77; not_selected 1.65 |
| B2 | 8,192 | 256 | 1 | 4.12 | 6 | 64 | 27 | 49 | 98 | 24 | 2,914 | long_scoreboard 10.04; wait 1.81; not_selected 1.10 |
| B2 | 16,384 | 256 | 1 | 6.59 | 6 | 33 | 42 | 44 | 99 | 24 | 22,977 | long_scoreboard 12.25; wait 1.82; not_selected 1.04 |
| B3 | 2,048 | 256 | 1 | 3.77 | 4 | 50 | 25 | 72 | 96 | 20 | 183 | long_scoreboard 2.99; wait 1.82; not_selected 1.77 |
| B3 | 8,192 | 256 | 1 | 3.99 | 4 | 50 | 28 | 58 | 74 | 20 | 2,683 | long_scoreboard 4.44; wait 1.86; selected 1.00 |
| B3 | 16,384 | 256 | 1 | 4.00 | 4 | 56 | 12 | 23 | 25 | 20 | 25,475 | long_scoreboard 4.28; wait 1.81; selected 1.00 |

### Table D. Peak GPU memory above the input (MiB)

| S | input | composed | torch.softmax, B1, B2, B3 |
|---|---|---|---|
| 1,024 | 2 | 6 | 2 |
| 2,048 | 8 | 24 | 8 |
| 4,096 | 32 | 96 | 32 |
| 8,192 | 128 | 384 | 128 |
| 16,384 | 512 | 1,536 | 512 |
| 32,768 | 2,048 | 6,144 | 2,048 |

### Table E. Knobs at S = 8192 (median µs, mean of two runs)

| variant | block 128 | block 256 | block 512 | block 1024 | `__expf` at 8192 | `__expf` at 16384 |
|---|---|---|---|---|---|---|
| B1 | 2,842 | 2,317 | 1,649 | 2,066 | 2,287 (1.3% faster) | 10,839 (3.6% faster) |
| B2 | 2,354 | 2,236 | 2,033 | 2,344 | 2,010 (10.1% faster) | 8,580 (8.0% faster) |

Run-to-run spread: the largest relative difference between run 1 and run 2 is 53.6% (B3, S = 1024, block 256).
