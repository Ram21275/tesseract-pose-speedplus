| module | params | fp32_MB | fp16_MB | int8_dynamic_MB | int8w_ideal_MB | linear_or_mha_modules | converted_by_dynamic_int8 |
|---|---|---|---|---|---|---|---|
| tree DINOv3 | 2391564 | 9.205 | 4.612 | 6.165 | 2.326 | 20 | 20 |
| tree MoGe-2 | 2128404 | 8.389 | 4.204 | 6.096 | 2.069 | 20 | 20 |
| translation block | 331011 | 1.267 | 0.636 | 1.078 | 0.323 | 4 | 4 |