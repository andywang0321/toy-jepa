# Phase 2 seeds

| metric | pixel | JEPA |
|---|---:|---:|
| c_t_state_k3_position_mae | 6.8643 ± 0.2018 (n=5) | 5.1823 ± 0.0846 (n=5) |
| c_t_state_k24_position_mae | 6.6547 ± 0.1638 (n=5) | 5.1541 ± 0.1476 (n=5) |
| c_t_state_k48_position_mae | 6.7289 ± 0.1805 (n=5) | 5.6124 ± 0.1679 (n=5) |
| c_t_state_final_position_mae | 8.8976 ± 0.1181 (n=5) | 7.6131 ± 0.0973 (n=5) |
| z_hat_state_final_position_mae | 8.2607 ± 0.1071 (n=5) | 7.4359 ± 0.1796 (n=5) |
| c_t_digit_accuracy | 0.0308 ± 0.0076 (n=5) | 0.0198 ± 0.0022 (n=5) |
| z_hat_digit_accuracy | 0.0352 ± 0.0031 (n=5) | 0.0190 ± 0.0032 (n=5) |
| c_t_bounce_accuracy | 0.8044 ± 0.0086 (n=5) | 0.8404 ± 0.0158 (n=5) |
| c_t_bounce_balanced_accuracy | 0.7605 ± 0.0128 (n=5) | 0.7979 ± 0.0163 (n=5) |
| c_t_variance | 0.5314 ± 0.0199 (n=5) | 0.2649 ± 0.0154 (n=5) |
| c_t_cosine | 0.4903 ± 0.0191 (n=5) | 0.7143 ± 0.0169 (n=5) |
| c_t_rank | 59.6408 ± 0.4486 (n=5) | 23.5561 ± 0.2816 (n=5) |

Final position MAE by context length:

| m | pixel c_t | JEPA c_t | pixel z_hat | JEPA z_hat |
|---:|---:|---:|---:|---:|
| 3 | 8.897 ± 0.241 | 8.455 ± 0.227 | 7.783 ± 0.212 | 7.584 ± 0.163 |
| 6 | 9.088 ± 0.339 | 8.227 ± 0.174 | 7.668 ± 0.269 | 7.418 ± 0.203 |
| 12 | 8.733 ± 0.221 | 8.120 ± 0.059 | 7.419 ± 0.389 | 7.320 ± 0.345 |

![comparison](p2_comparison.png)
