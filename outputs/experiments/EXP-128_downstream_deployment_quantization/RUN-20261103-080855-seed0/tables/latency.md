| variant | inference | batch | device | precision | threads | ms_per_batch | ms_per_img | peak_transient_gpu_mb | gpu_util_mem_used |
|---|---|---|---|---|---|---|---|---|---|
| DINO | greedy | 1 | GPU | fp32 |  | 18.904 | 18.904 | 1.438 | 100, 12029
0, 18 |
| DINO | greedy | 1 | GPU | bf16 |  | 19.316 | 19.316 | 5.234 | 100, 12039
0, 18 |
| DINO | greedy | 1 | GPU | fp16 |  | 19.192 | 19.192 | 5.234 | 100, 12045
0, 18 |
| DINO | greedy | 1 | CPU | fp32 | 1 | 6.706 | 6.706 | nan |  |
| DINO | greedy | 1 | CPU | fp32 | 24 | 931.005 | 931.005 | nan |  |
| DINO | greedy | 1 | CPU | int8_dynamic | 1 | 7.363 | 7.363 | nan |  |
| DINO | greedy | 1 | CPU | int8_dynamic | 24 | 1047.011 | 1047.011 | nan |  |
| DINO | greedy | 64 | GPU | fp32 |  | 21.148 | 0.330 | 29.000 | 100, 12085
0, 18 |
| DINO | greedy | 64 | GPU | bf16 |  | 19.911 | 0.311 | 27.500 | 100, 12089
0, 18 |
| DINO | greedy | 64 | GPU | fp16 |  | 20.676 | 0.323 | 27.500 | 100, 12089
0, 18 |
| DINO | greedy | 64 | CPU | fp32 | 1 | 183.193 | 2.862 | nan |  |
| DINO | greedy | 64 | CPU | fp32 | 24 | 2289.291 | 35.770 | nan |  |
| DINO | greedy | 64 | CPU | int8_dynamic | 1 | 172.257 | 2.692 | nan |  |
| DINO | greedy | 64 | CPU | int8_dynamic | 24 | 2167.991 | 33.875 | nan |  |
| DINO | orbit | 1 | GPU | fp32 |  | 111.634 | 111.634 | 11.009 | 100, 12089
0, 18 |
| DINO | orbit | 1 | GPU | bf16 |  | 111.094 | 111.094 | 10.707 | 100, 12089
0, 18 |
| DINO | orbit | 1 | GPU | fp16 |  | 109.905 | 109.905 | 10.707 | 100, 12089
0, 18 |
| DINO | orbit | 1 | CPU | fp32 | 1 | 129.846 | 129.846 | nan |  |
| DINO | orbit | 1 | CPU | fp32 | 24 | 7381.000 | 7381.000 | nan |  |
| DINO | orbit | 1 | CPU | int8_dynamic | 1 | 126.640 | 126.640 | nan |  |
| DINO | orbit | 1 | CPU | int8_dynamic | 24 | 7843.974 | 7843.974 | nan |  |
| DINO | orbit | 64 | GPU | fp32 |  | 261.650 | 4.088 | 704.461 | 100, 12909
0, 18 |
| DINO | orbit | 64 | GPU | bf16 |  | 151.066 | 2.360 | 446.084 | 100, 12909
0, 18 |
| DINO | orbit | 64 | GPU | fp16 |  | 153.248 | 2.394 | 446.084 | 100, 12909
0, 18 |
| DINO | orbit | 64 | CPU | fp32 | 1 | 11091.337 | 173.302 | nan |  |
| DINO | orbit | 64 | CPU | fp32 | 24 | 15983.984 | 249.750 | nan |  |
| DINO | orbit | 64 | CPU | int8_dynamic | 1 | 11006.742 | 171.980 | nan |  |
| DINO | orbit | 64 | CPU | int8_dynamic | 24 | 14997.006 | 234.328 | nan |  |
| PoE | greedy | 1 | GPU | fp32 |  | 35.649 | 35.649 | 2.812 | 100, 12913
0, 18 |
| PoE | greedy | 1 | GPU | bf16 |  | 35.947 | 35.947 | 9.062 | 100, 12919
0, 18 |
| PoE | greedy | 1 | GPU | fp16 |  | 35.957 | 35.957 | 9.062 | 100, 12919
0, 18 |
| PoE | greedy | 1 | CPU | fp32 | 1 | 20.552 | 20.552 | nan |  |
| PoE | greedy | 1 | CPU | fp32 | 24 | 2008.990 | 2008.990 | nan |  |
| PoE | greedy | 1 | CPU | int8_dynamic | 1 | 20.929 | 20.929 | nan |  |
| PoE | greedy | 1 | CPU | int8_dynamic | 24 | 2266.996 | 2266.996 | nan |  |
| PoE | greedy | 64 | GPU | fp32 |  | 43.613 | 0.681 | 117.000 | 100, 12919
0, 18 |
| PoE | greedy | 64 | GPU | bf16 |  | 38.487 | 0.601 | 86.259 | 100, 12921
0, 18 |
| PoE | greedy | 64 | GPU | fp16 |  | 38.278 | 0.598 | 86.259 | 99, 12921
0, 18 |
| PoE | greedy | 64 | CPU | fp32 | 1 | 948.113 | 14.814 | nan |  |
| PoE | greedy | 64 | CPU | fp32 | 24 | 4662.232 | 72.847 | nan |  |
| PoE | greedy | 64 | CPU | int8_dynamic | 1 | 906.718 | 14.167 | nan |  |
| PoE | greedy | 64 | CPU | int8_dynamic | 24 | 4430.860 | 69.232 | nan |  |
| PoE | orbit | 1 | GPU | fp32 |  | 191.849 | 191.849 | 41.270 | 100, 12921
0, 18 |
| PoE | orbit | 1 | GPU | bf16 |  | 191.506 | 191.506 | 32.259 | 100, 12921
0, 18 |
| PoE | orbit | 1 | GPU | fp16 |  | 190.283 | 190.283 | 32.259 | 100, 12921
0, 18 |
| PoE | orbit | 1 | CPU | fp32 | 1 | 577.605 | 577.605 | nan |  |
| PoE | orbit | 1 | CPU | fp32 | 24 | 15449.002 | 15449.002 | nan |  |
| PoE | orbit | 1 | CPU | int8_dynamic | 1 | 567.600 | 567.600 | nan |  |
| PoE | orbit | 1 | CPU | int8_dynamic | 24 | 16116.003 | 16116.003 | nan |  |
| PoE | orbit | 64 | GPU | fp32 |  | 891.718 | 13.933 | 2641.149 | 100, 15239
0, 18 |
| PoE | orbit | 64 | GPU | bf16 |  | 436.165 | 6.815 | 1617.724 | 100, 15241
0, 18 |
| PoE | orbit | 64 | GPU | fp16 |  | 425.301 | 6.645 | 1617.724 | 100, 15241
0, 18 |
| PoE | orbit | 64 | CPU | fp32 | 1 | 50175.115 | 783.986 | nan |  |
| PoE | orbit | 64 | CPU | fp32 | 24 | 38437.083 | 600.579 | nan |  |
| PoE | orbit | 64 | CPU | int8_dynamic | 1 | 50187.555 | 784.181 | nan |  |
| PoE | orbit | 64 | CPU | int8_dynamic | 24 | 36056.192 | 563.378 | nan |  |