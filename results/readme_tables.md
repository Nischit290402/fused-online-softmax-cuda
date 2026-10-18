### Table A. Median time per call (µs, CUDA events, mean of the two sweep runs; lower is better)

| S | composed | torch.softmax | B1 (3-pass) | B2 (online, 2-pass) | B3 (row in smem) |
|---|---|---|---|---|---|
| 1,024 | 167.1 | 50.6 | 56.6 | 68.3 | 67.2 |
| 2,048 | 316.7 | 101.5 | 134.6 | 153.8 | 159.6 |
| 4,096 | 1,099.5 | 453.4 | 457.9 | 516.3 | 479.5 |
| 8,192 | 4,401.1 | 1,538.6 | 2,334.3 | 2,151.0 | 2,074.1 |
| 16,384 | 16,866.6 | 4,487.1 | 11,044.5 | 8,934.4 | 15,794.8 |
| 32,768 | 106,698.3 | 17,299.6 | 44,263.3 | 35,463.6 | skip (row > smem) |

### Table B. Effective bandwidth (GB/s, 4·S² bytes / time; theoretical peak 352 GB/s)

| S | composed | torch.softmax | B1 | B2 | B3 | best custom / torch |
|---|---|---|---|---|---|---|
| 1,024 | 25 | 83 | 74 | 61 | 62 | 89% |
| 2,048 | 53 | 165 | 125 | 109 | 105 | 75% |
| 4,096 | 61 | 148 | 147 | 130 | 140 | 99% |
| 8,192 | 61 | 174 | 115 | 125 | 129 | 74% |
| 16,384 | 64 | 239 | 97 | 120 | 68 | 50% |
| 32,768 | 40 | 248 | 97 | 121 | skip | 49% |

### Table C. Nsight Compute (ncu time is at locked base clocks, so it is slower than Table A)

| variant | S | kernels/call | DRAM B/elem | model B/elem | L2 hit % | DRAM % peak | SM % peak | occupancy % | regs/thread | time (µs) |
|---|---|---|---|---|---|---|---|---|---|---|
| composed | 8,192 | 5 | 15.94 | 16 | 39 | 54 | 44 | 93 | 33 | 5,663 |
| torch.softmax | 2,048 | 1 | 3.79 | 4 | 50 | 50 | 46 | 66 | 80 | 92 |
| torch.softmax | 8,192 | 1 | 3.99 | 4 | 50 | 35 | 57 | 98 | 35 | 2,170 |
| torch.softmax | 16,384 | 1 | 4.00 | 4 | 50 | 50 | 49 | 97 | 41 | 6,113 |
| B1 | 2,048 | 1 | 3.76 | 8 | 50 | 31 | 54 | 97 | 22 | 146 |
| B1 | 8,192 | 1 | 4.12 | 8 | 71 | 25 | 34 | 98 | 22 | 3,190 |
| B1 | 16,384 | 1 | 7.99 | 8 | 25 | 41 | 29 | 99 | 22 | 14,812 |
| B2 | 2,048 | 1 | 3.77 | 6 | 51 | 26 | 71 | 95 | 24 | 175 |
| B2 | 8,192 | 1 | 4.15 | 6 | 64 | 27 | 48 | 98 | 24 | 2,928 |
| B2 | 16,384 | 1 | 6.00 | 6 | 33 | 38 | 44 | 99 | 24 | 12,122 |
| B3 | 2,048 | 1 | 3.77 | 4 | 50 | 25 | 72 | 96 | 20 | 183 |
| B3 | 8,192 | 1 | 3.99 | 4 | 50 | 28 | 58 | 74 | 20 | 2,685 |
| B3 | 16,384 | 1 | 4.00 | 4 | 50 | 12 | 23 | 25 | 20 | 25,499 |

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
| B1 | 2,820 | 2,334 | 1,696 | 1,937 | 2,274 (2.6% faster) | 10,641 (3.7% faster) |
| B2 | 2,246 | 2,151 | 1,988 | 2,260 | 2,040 (5.2% faster) | 8,503 (4.8% faster) |

Run-to-run spread: the largest relative difference between run 1 and run 2 is 21.4% (torch.softmax, S = 2048, block 256).
