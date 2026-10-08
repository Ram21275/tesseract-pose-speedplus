| grid | config | n | haar_decoded_mean_deg | haar_decoded_p95_deg | haar_decoded_max_deg | haar_nearest_mean_deg | gt_decoded_mean_deg | spacing_cv | nn_min_deg | decode |
|---|---|---|---|---|---|---|---|---|---|---|
| tesseract | L3 | 2048 | 10.011 | 15.833 | 23.813 | 9.877 | 10.020 | 0.209 | 11.161 | tree, 28 nodes (greedy) |
| tesseract | L4 | 16384 | 5.012 | 7.921 | 11.831 | 4.949 | 5.016 | 0.213 | 5.312 | tree, 36 nodes (greedy) |
| tesseract | L5 | 131072 | 2.506 | 3.966 | 5.900 | 2.475 | 2.509 | 0.206 | 2.593 | tree, 44 nodes (greedy) |
| spacehop_hopf | 256x12 (paper) | 3072 | 8.794 | 13.998 | 16.943 | 8.794 | 8.788 | 0.085 | 11.851 | flat argmax over 3072 |
| spacehop_hopf | 171x12 (~L3) | 2052 | 9.671 | 14.598 | 18.121 | 9.671 | 9.655 | 0.052 | 14.432 | flat argmax over 2052 |
| spacehop_hopf | 1365x12 (~L4) | 16380 | 6.764 | 13.086 | 15.396 | 6.764 | 6.765 | 0.130 | 5.467 | flat argmax over 16380 |
| spacehop_hopf | 10923x12 (~L5) | 131076 | 6.060 | 12.909 | 15.051 | 6.060 | 6.062 | 0.207 | 1.695 | flat argmax over 131076 |