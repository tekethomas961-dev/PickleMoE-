# Experiment Summary

| Run | Model | Best Epoch | Params | Active Params | Test Loss | Acc | Macro F1 | Balanced Acc | CV(load) | Max/Mean | Gate Entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final_dense_mlp | mlp | 63 | 12004 | 12004 | 0.000373 | 1.0 | 1.0 | 1.0 |  |  |  |
| final_moe_k1_lb | moe-K=1, lambda_lb=0.01 | 42 | 20596 | 7336 | 0.011049 | 1.0 | 1.0 | 1.0 | 0.174105 | 1.125 | 1.38379 |
| final_moe_k2_lb | moe-K=2, lambda_lb=0.01 | 67 | 20596 | 11756 | 0.010445 | 1.0 | 1.0 | 1.0 | 0.16322 | 1.2375 | 0.374603 |
| final_moe_k1_no_lb | moe-K=1, lambda_lb=0.0 | 83 | 20596 | 7336 | 0.000274 | 1.0 | 1.0 | 1.0 | 0.776812 | 2.15 | 1.359131 |
| final_moe_k2_no_lb | moe-K=2, lambda_lb=0.0 | 67 | 20596 | 11756 | 0.000204 | 1.0 | 1.0 | 1.0 | 0.339692 | 1.5875 | 0.384336 |

Figures are stored under each run directory's `figures/` folder.