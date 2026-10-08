| device | batch | depth_L | hypotheses_K | head | params | median_ms | p95_ms | per_image_ms | iters |
|---|---|---|---|---|---|---|---|---|---|
| NVIDIA RTX A400 | 1 | 2 | 256 | tesseract_beam1 | 9025932 | 0.636 | 0.641 | 0.636 | 5 |
| NVIDIA RTX A400 | 1 | 2 | 256 | tesseract_beam4 | 9025932 | 0.702 | 0.734 | 0.702 | 5 |
| NVIDIA RTX A400 | 1 | 2 | 256 | flat_argmax | 8815872 | 0.422 | 0.434 | 0.422 | 5 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | tesseract_beam1 | 9026060 | 0.778 | 0.787 | 0.778 | 5 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | tesseract_beam4 | 9026060 | 0.843 | 0.861 | 0.843 | 5 |
| NVIDIA RTX A400 | 1 | 3 | 2048 | flat_argmax | 9735168 | 0.520 | 0.529 | 0.520 | 5 |
| NVIDIA RTX A400 | 1 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 64.639 | 64.708 | 64.639 | 50 |
| NVIDIA RTX A400 | 32 | - | - | backbone DINOv3-L/16 @256 (shared by both arms) | 303079424 | 1122.052 | 1132.002 | 35.064 | 50 |