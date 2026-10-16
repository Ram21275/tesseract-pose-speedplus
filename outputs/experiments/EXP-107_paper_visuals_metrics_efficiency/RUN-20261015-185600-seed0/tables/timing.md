| component | batch | ms_per_batch | ms_per_image | images_per_s | params_M | gpu_util_mem_at_start |
|---|---|---|---|---|---|---|
| DINOv3-L features (frozen) | 1 | 6.012 | 6.012 | 166.326 | 303.079 | 6, 3833 |
| MoGe-2 ViT-L features (frozen) | 1 | 20.536 | 20.536 | 48.696 | 330.902 | 4, 4325 |
| DINOv3-L features (frozen) | 64 | 155.945 | 2.437 | 410.400 | 303.079 | 98, 4685 |
| MoGe-2 ViT-L features (frozen) | 64 | 1578.782 | 24.668 | 40.538 | 330.902 | 52, 32467 |
| tree A+B encode + per-level PoE + greedy decode | 1 | 12.753 | 12.753 | 78.410 | 4.520 | 14, 4161 |
| tree A+B beam 8 (candidate set) | 1 | 13.481 | 13.481 | 74.178 | None | 14, 4163 |
| flow: one velocity evaluation (2 heads, M = 32) | 1 | 1.344 | 1.344 | 743.895 | 3.346 | 14, 4203 |
| flow: full sampling NFE 10, M 32 | 1 | 15.078 | 15.078 | 66.322 | None | 23, 4203 |
| KDE mode (M 32) | 1 | 0.076 | 0.076 | 13138.022 | None | 23, 4203 |
| tree A+B encode + per-level PoE + greedy decode | 64 | 14.257 | 0.223 | 4489.013 | 4.520 | 23, 4263 |
| tree A+B beam 8 (candidate set) | 64 | 22.319 | 0.349 | 2867.449 | None | 91, 4683 |
| flow: one velocity evaluation (2 heads, M = 32) | 64 | 31.606 | 0.494 | 2024.912 | 3.346 | 91, 7823 |
| flow: full sampling NFE 10, M 32 | 64 | 322.729 | 5.043 | 198.309 | None | 100, 7823 |
| KDE mode (M 32) | 64 | 0.078 | 0.001 | 825072.040 | None | 100, 7823 |