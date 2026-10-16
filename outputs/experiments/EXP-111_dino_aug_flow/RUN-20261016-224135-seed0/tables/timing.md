| batch | component | ms_per_image | fps_or_img_per_s |
|---|---|---|---|
| 1 | DINOv3-L features | 6.196 | 161.405 |
| 1 | DINOv3 tree greedy | 6.876 | 145.442 |
| 1 | flow NFE10 M32 (1 head) | 8.490 | 117.784 |
| 1 | KDE | 0.075 | 13357.023 |
| 1 | TOTAL tree only | 13.071 | 76.504 |
| 1 | TOTAL tree + flow | 21.636 | 46.219 |
| 64 | DINOv3-L features | 2.767 | 361.377 |
| 64 | DINOv3 tree greedy | 0.115 | 8707.227 |
| 64 | flow NFE10 M32 (1 head) | 1.202 | 831.816 |
| 64 | KDE | 0.001 | 879918.330 |
| 64 | TOTAL tree only | 2.882 | 346.976 |
| 64 | TOTAL tree + flow | 4.085 | 244.776 |