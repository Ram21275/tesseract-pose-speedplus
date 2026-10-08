| device | batch | depth_L | hypotheses_K | head | params | median_ms | p95_ms | per_image_ms | iters |
|---|---|---|---|---|---|---|---|---|---|
| NVIDIA RTX A400 | 1 | 2 | 256 | tesseract_beam1 | 9025932 | 0.639 | 0.840 | 0.639 | 200 |
| NVIDIA RTX A400 | 1 | 2 | 256 | tesseract_beam4 | 9025932 | 0.674 | 0.736 | 0.674 | 200 |
| NVIDIA RTX A400 | 1 | 2 | 256 | flat_argmax | 8815872 | 0.418 | 0.421 | 0.418 | 200 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | tesseract_beam1 | 9026060 | 0.742 | 0.891 | 0.742 | 200 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | tesseract_beam4 | 9026060 | 0.825 | 0.872 | 0.825 | 200 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | flat_argmax | 9735168 | 0.506 | 0.508 | 0.506 | 200 |
| NVIDIA RTX A400 | 1 | 4 | 16384 | tesseract_beam1 | 9026188 | 0.982 | 1.001 | 0.982 | 200 |
| NVIDIA RTX A400 | 1 | 4 | 16384 | tesseract_beam4 | 9026188 | 1.087 | 1.114 | 1.087 | 200 |
| NVIDIA RTX A400 | 1 | 4 | 16384 | flat_argmax | 17089536 | 0.775 | 0.779 | 0.775 | 200 |
| NVIDIA RTX A400 | 1 | 5 | 131072 | tesseract_beam1 | 9026316 | 1.215 | 1.241 | 1.215 | 200 |
| NVIDIA RTX A400 | 1 | 5 | 131072 | tesseract_beam4 | 9026316 | 1.347 | 1.385 | 1.347 | 200 |
| NVIDIA RTX A400 | 1 | 5 | 131072 | flat_argmax | 75924480 | 3.320 | 3.332 | 3.320 | 200 |
| NVIDIA RTX A400 | 1 | 6 | 1048576 | tesseract_beam1 | 9026444 | 1.451 | 1.483 | 1.451 | 200 |
| NVIDIA RTX A400 | 1 | 6 | 1048576 | tesseract_beam4 | 9026444 | 1.610 | 1.642 | 1.610 | 200 |
| NVIDIA RTX A400 | 1 | 6 | 1048576 | flat_argmax | 546604032 | 23.555 | 23.565 | 23.555 | 200 |
| NVIDIA RTX A400 | 256 | 2 | 256 | tesseract_beam1 | 9025932 | 3.916 | 4.028 | 0.015 | 200 |
| NVIDIA RTX A400 | 256 | 2 | 256 | tesseract_beam4 | 9025932 | 4.904 | 4.995 | 0.019 | 200 |
| NVIDIA RTX A400 | 256 | 2 | 256 | flat_argmax | 8815872 | 3.426 | 3.474 | 0.013 | 200 |
| NVIDIA RTX A400 | 256 | 3 | 2048 | tesseract_beam1 | 9026060 | 4.205 | 4.327 | 0.016 | 200 |
| NVIDIA RTX A400 | 256 | 3 | 2048 | tesseract_beam4 | 9026060 | 5.708 | 5.778 | 0.022 | 200 |
| NVIDIA RTX A400 | 256 | 3 | 2048 | flat_argmax | 9735168 | 3.796 | 3.879 | 0.015 | 200 |
| NVIDIA RTX A400 | 256 | 4 | 16384 | tesseract_beam1 | 9026188 | 4.508 | 4.592 | 0.018 | 200 |
| NVIDIA RTX A400 | 256 | 4 | 16384 | tesseract_beam4 | 9026188 | 6.478 | 6.542 | 0.025 | 200 |
| NVIDIA RTX A400 | 256 | 4 | 16384 | flat_argmax | 17089536 | 6.181 | 6.418 | 0.024 | 200 |
| NVIDIA RTX A400 | 256 | 5 | 131072 | tesseract_beam1 | 9026316 | 4.789 | 4.919 | 0.019 | 200 |
| NVIDIA RTX A400 | 256 | 5 | 131072 | tesseract_beam4 | 9026316 | 7.282 | 7.424 | 0.028 | 200 |
| NVIDIA RTX A400 | 256 | 5 | 131072 | flat_argmax | 75924480 | 26.312 | 27.013 | 0.103 | 200 |
| NVIDIA RTX A400 | 256 | 6 | 1048576 | tesseract_beam1 | 9026444 | 5.109 | 5.208 | 0.020 | 200 |
| NVIDIA RTX A400 | 256 | 6 | 1048576 | tesseract_beam4 | 9026444 | 8.122 | 8.254 | 0.032 | 200 |
| NVIDIA RTX A400 | 256 | 6 | 1048576 | flat_argmax | 546604032 | 185.517 | 189.031 | 0.725 | 200 |
| CPU (1 thread) | 1 | 2 | 256 | tesseract_beam1 | 9025932 | 1.370 | 1.635 | 1.370 | 200 |
| CPU (1 thread) | 1 | 2 | 256 | tesseract_beam4 | 9025932 | 1.414 | 1.746 | 1.414 | 200 |
| CPU (1 thread) | 1 | 2 | 256 | flat_argmax | 8815872 | 0.983 | 1.077 | 0.983 | 200 |
| CPU (1 thread) | 1 | 3 | 2048 | tesseract_beam1 | 9026060 | 1.500 | 1.816 | 1.500 | 200 |
| CPU (1 thread) | 1 | 3 | 2048 | tesseract_beam4 | 9026060 | 1.589 | 1.812 | 1.589 | 200 |
| CPU (1 thread) | 1 | 3 | 2048 | flat_argmax | 9735168 | 1.152 | 1.218 | 1.152 | 200 |
| CPU (1 thread) | 1 | 4 | 16384 | tesseract_beam1 | 9026188 | 1.637 | 2.105 | 1.637 | 200 |
| CPU (1 thread) | 1 | 4 | 16384 | tesseract_beam4 | 9026188 | 1.753 | 1.995 | 1.753 | 200 |
| CPU (1 thread) | 1 | 4 | 16384 | flat_argmax | 17089536 | 2.260 | 2.652 | 2.260 | 200 |
| CPU (1 thread) | 1 | 5 | 131072 | tesseract_beam1 | 9026316 | 1.746 | 2.023 | 1.746 | 200 |
| CPU (1 thread) | 1 | 5 | 131072 | tesseract_beam4 | 9026316 | 1.922 | 2.033 | 1.922 | 200 |
| CPU (1 thread) | 1 | 5 | 131072 | flat_argmax | 75924480 | 12.160 | 12.901 | 12.160 | 200 |
| CPU (1 thread) | 1 | 6 | 1048576 | tesseract_beam1 | 9026444 | 1.863 | 2.039 | 1.863 | 200 |
| CPU (1 thread) | 1 | 6 | 1048576 | tesseract_beam4 | 9026444 | 2.097 | 2.549 | 2.097 | 200 |
| CPU (1 thread) | 1 | 6 | 1048576 | flat_argmax | 546604032 | 91.781 | 93.364 | 91.781 | 200 |
| CPU (1 thread) | 256 | 2 | 256 | tesseract_beam1 | 9025932 | 34.578 | 34.831 | 0.135 | 200 |
| CPU (1 thread) | 256 | 2 | 256 | tesseract_beam4 | 9025932 | 42.522 | 42.757 | 0.166 | 200 |
| CPU (1 thread) | 256 | 2 | 256 | flat_argmax | 8815872 | 31.601 | 31.765 | 0.123 | 200 |
| CPU (1 thread) | 256 | 3 | 2048 | tesseract_beam1 | 9026060 | 35.745 | 36.026 | 0.140 | 200 |
| CPU (1 thread) | 256 | 3 | 2048 | tesseract_beam4 | 9026060 | 47.697 | 48.309 | 0.186 | 200 |
| CPU (1 thread) | 256 | 3 | 2048 | flat_argmax | 9735168 | 35.433 | 35.648 | 0.138 | 200 |
| CPU (1 thread) | 256 | 4 | 16384 | tesseract_beam1 | 9026188 | 37.367 | 37.735 | 0.146 | 200 |
| CPU (1 thread) | 256 | 4 | 16384 | tesseract_beam4 | 9026188 | 53.014 | 53.414 | 0.207 | 200 |
| CPU (1 thread) | 256 | 4 | 16384 | flat_argmax | 17089536 | 64.087 | 64.752 | 0.250 | 200 |
| CPU (1 thread) | 256 | 5 | 131072 | tesseract_beam1 | 9026316 | 38.620 | 38.931 | 0.151 | 200 |
| CPU (1 thread) | 256 | 5 | 131072 | tesseract_beam4 | 9026316 | 58.062 | 58.421 | 0.227 | 200 |
| CPU (1 thread) | 256 | 5 | 131072 | flat_argmax | 75924480 | 345.780 | 355.751 | 1.351 | 173 |
| CPU (1 thread) | 256 | 6 | 1048576 | tesseract_beam1 | 9026444 | 40.617 | 41.035 | 0.159 | 200 |
| CPU (1 thread) | 256 | 6 | 1048576 | tesseract_beam4 | 9026444 | 63.807 | 64.164 | 0.249 | 200 |
| CPU (1 thread) | 256 | 6 | 1048576 | flat_argmax | 546604032 | 2583.859 | 2602.903 | 10.093 | 24 |
| NVIDIA RTX 6000 Ada Generation | 1 | 2 | 256 | tesseract_beam1 | 9025932 | 0.499 | 0.519 | 0.499 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 2 | 256 | tesseract_beam4 | 9025932 | 0.531 | 0.543 | 0.531 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 2 | 256 | flat_argmax | 8815872 | 0.054 | 0.054 | 0.054 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 3 | 2048 | tesseract_beam1 | 9026060 | 0.748 | 0.765 | 0.748 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 3 | 2048 | tesseract_beam4 | 9026060 | 0.795 | 0.811 | 0.795 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 3 | 2048 | flat_argmax | 9735168 | 0.058 | 0.058 | 0.058 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 4 | 16384 | tesseract_beam1 | 9026188 | 0.991 | 1.008 | 0.991 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 4 | 16384 | tesseract_beam4 | 9026188 | 1.052 | 1.069 | 1.052 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 4 | 16384 | flat_argmax | 17089536 | 0.060 | 0.060 | 0.060 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 5 | 131072 | tesseract_beam1 | 9026316 | 1.233 | 1.257 | 1.233 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 5 | 131072 | tesseract_beam4 | 9026316 | 1.309 | 1.336 | 1.309 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 5 | 131072 | flat_argmax | 75924480 | 0.398 | 0.399 | 0.398 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 6 | 1048576 | tesseract_beam1 | 9026444 | 1.472 | 1.510 | 1.472 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 6 | 1048576 | tesseract_beam4 | 9026444 | 1.544 | 1.584 | 1.544 | 200 |
| NVIDIA RTX 6000 Ada Generation | 1 | 6 | 1048576 | flat_argmax | 546604032 | 2.534 | 2.545 | 2.534 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 2 | 256 | tesseract_beam1 | 9025932 | 0.538 | 0.855 | 0.002 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 2 | 256 | tesseract_beam4 | 9025932 | 0.572 | 0.590 | 0.002 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 2 | 256 | flat_argmax | 8815872 | 0.231 | 0.236 | 0.001 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 3 | 2048 | tesseract_beam1 | 9026060 | 0.795 | 0.814 | 0.003 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 3 | 2048 | tesseract_beam4 | 9026060 | 0.859 | 0.876 | 0.003 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 3 | 2048 | flat_argmax | 9735168 | 0.238 | 0.243 | 0.001 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 4 | 16384 | tesseract_beam1 | 9026188 | 1.055 | 1.112 | 0.004 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 4 | 16384 | tesseract_beam4 | 9026188 | 1.136 | 1.165 | 0.004 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 4 | 16384 | flat_argmax | 17089536 | 0.442 | 0.458 | 0.002 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 5 | 131072 | tesseract_beam1 | 9026316 | 1.299 | 1.325 | 0.005 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 5 | 131072 | tesseract_beam4 | 9026316 | 1.403 | 1.438 | 0.005 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 5 | 131072 | flat_argmax | 75924480 | 1.950 | 1.953 | 0.008 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 6 | 1048576 | tesseract_beam1 | 9026444 | 1.561 | 1.606 | 0.006 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 6 | 1048576 | tesseract_beam4 | 9026444 | 1.685 | 1.718 | 0.007 | 200 |
| NVIDIA RTX 6000 Ada Generation | 256 | 6 | 1048576 | flat_argmax | 546604032 | 14.667 | 15.338 | 0.057 | 200 |
| NVIDIA RTX A400 | 1 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 64.667 | 64.769 | 64.667 | 50 |
| NVIDIA RTX A400 | 32 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 1124.476 | 1133.406 | 35.140 | 50 |
| NVIDIA RTX 6000 Ada Generation | 1 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 5.889 | 6.061 | 5.889 | 50 |
| NVIDIA RTX 6000 Ada Generation | 32 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 75.693 | 77.973 | 2.365 | 50 |