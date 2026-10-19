### Version 2: median time per call (µs, mean of two runs, block 256, accurate exp)

| S | torch.softmax | B2 | B3 (v1 launch) | B3v2 | B4 | B5 | best v2 / B2 | best v2 / torch |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 27.2 | 47.5 | 58.1 | 49.8 | 34.6 | 38.3 | 1.37x faster | 1.27x the time |
| 2,048 | 78.8 | 138.5 | 153.6 | 140.1 | 99.6 | 101.5 | 1.39x faster | 1.26x the time |
| 4,096 | 443.5 | 481.3 | 471.0 | 454.3 | 235.6 | 337.7 | 2.04x faster | 0.53x the time |
| 8,192 | 1,553.3 | 2,168.4 | 2,083.7 | 1,767.1 | 1,191.6 | 1,257.2 | 1.82x faster | 0.77x the time |
| 16,384 | 4,473.2 | 9,176.7 | 16,173.6 | 8,426.9 | 4,963.9 | 4,951.7 | 1.85x faster | 1.11x the time |
| 32,768 | 18,159.3 | 36,154.8 | skip | skip | 19,713.1 | 19,764.8 | 1.83x faster | 1.09x the time |

### B3 threads per block launched by v2

| S | requested | launched |
|---|---|---|
| 1,024 | 256 | 256 |
| 2,048 | 256 | 256 |
| 4,096 | 256 | 256 |
| 8,192 | 256 | 320 |
| 16,384 | 256 | 1024 |
| 32,768 | 256 | skip |

### B4 block size and fast exp2 (S = 8192 unless stated, median µs)

| block 128 | block 256 | block 512 | block 1024 | B4 ex2.approx 8192 | B4 ex2.approx 16384 | B5 ex2.approx 8192 | B5 ex2.approx 16384 |
|---|---|---|---|---|---|---|---|
| 1,258 | 1,192 | 882 | 1,048 | 1,183 | 4,981 | 1,272 | 4,988 |

### Nsight Compute, version 2 (B2 rows are the v1 reports, same kernel)

| variant | S | DRAM B/elem | DRAM % peak | SM % peak | occupancy % | regs | time (µs) | top stall reasons |
|---|---|---|---|---|---|---|---|---|
| B2 | 2,048 | 3.77 | 26 | 71 | 96 | 24 | 176 | long_scoreboard 3.60; wait 1.77; not_selected 1.65 |
| B2 | 8,192 | 4.12 | 27 | 49 | 98 | 24 | 2,914 | long_scoreboard 10.04; wait 1.81; not_selected 1.10 |
| B2 | 16,384 | 6.00 | 38 | 44 | 99 | 24 | 12,122 |  |
| B3v2 | 2,048 | 3.77 | 25 | 72 | 96 | 20 | 183 | long_scoreboard 3.01; wait 1.82; not_selected 1.77 |
| B3v2 | 8,192 | 3.99 | 31 | 65 | 91 | 20 | 2,475 | long_scoreboard 4.45; wait 1.87; not_selected 1.32 |
| B3v2 | 16,384 | 4.05 | 28 | 62 | 98 | 20 | 11,168 | long_scoreboard 3.74; not_selected 2.27; wait 1.88 |
| B4 | 2,048 | 3.74 | 52 | 74 | 92 | 29 | 87 | wait 1.97; not_selected 1.84; long_scoreboard 1.75 |
| B4 | 8,192 | 5.02 | 81 | 46 | 96 | 29 | 1,183 | long_scoreboard 9.65; wait 1.66; not_selected 1.19 |
| B4 | 16,384 | 6.00 | 94 | 38 | 97 | 29 | 4,875 | long_scoreboard 14.09; wait 1.55; selected 1.00 |
| B5 | 2,048 | 5.65 | 91 | 37 | 87 | 45 | 78 | long_scoreboard 13.77; wait 1.31; not_selected 1.01 |
| B5 | 8,192 | 6.00 | 93 | 34 | 94 | 45 | 1,241 | long_scoreboard 17.84; wait 1.19; mio_throttle 1.02 |
| B5 | 16,384 | 6.00 | 93 | 34 | 96 | 45 | 4,931 | long_scoreboard 19.05; mio_throttle 1.24; not_selected 1.20 |
