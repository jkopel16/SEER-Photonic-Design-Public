# Ablation grid

## Surrogate rows (shared evaluator; test split, grouped sigma-stratified unless footnoted)

| row | set | TTA | n | MAE | RMSE | rho | RMS(s)/RMSE | PICP 1s | 2s | 3s |
|---|---|---|---|---|---|---|---|---|---|---|
| deployed v2 (reference; #4 TTA) | test | off | 272 | 0.005442 | 0.006913 | 0.701 | 1.003 | 0.651 | 0.952 | 0.996 |
| deployed v2 (reference; #4 TTA) | test | on | 272 | 0.005442 | 0.006912 | 0.701 | 1.005 | 0.651 | 0.952 | 0.996 |
| #8 NLL + shared splits | test | off | 272 | 0.005265 | 0.006702 | 0.706 | 1.025 | 0.673 | 0.956 | 1.000 |
| #8 NLL + shared splits | test | on | 272 | 0.005287 | 0.006716 | 0.694 | 1.027 | 0.669 | 0.960 | 1.000 |
| #9 SmoothL1 + k-fold | test | off | 272 | 0.005409 | 0.006820 | 0.685 | 0.243 | 0.210 | 0.338 | 0.467 |
| #9 SmoothL1 + k-fold | test | on | 272 | 0.005415 | 0.006822 | 0.694 | 0.210 | 0.173 | 0.312 | 0.415 |
| #9 SmoothL1 + shared | test | off | 272 | 0.005304 | 0.006693 | 0.697 | 0.259 | 0.199 | 0.375 | 0.496 |
| #9 SmoothL1 + shared | test | on | 272 | 0.005319 | 0.006712 | 0.699 | 0.219 | 0.158 | 0.327 | 0.438 |
| #10 D4 train-aug off | test | off | 272 | 0.005992 | 0.007565 | 0.626 | 1.000 | 0.654 | 0.960 | 1.000 |
| #10 D4 train-aug off | test | on | 272 | 0.006112 | 0.007808 | 0.621 | 1.010 | 0.702 | 0.956 | 1.000 |
| #11 FFT channel off | test | off | 272 | 0.006254 | 0.007790 | 0.559 | 1.104 | 0.728 | 0.963 | 1.000 |
| #11 FFT channel off | test | on | 272 | 0.006286 | 0.007788 | 0.564 | 1.108 | 0.717 | 0.963 | 1.000 |
| #12 naive split (own test set) | test | off | 272 | 0.005573 | 0.007032 | 0.598 | 0.971 | 0.662 | 0.949 | 0.996 |
| #12 naive split (own test set) | test | on | 272 | 0.005570 | 0.007033 | 0.592 | 0.975 | 0.665 | 0.949 | 0.996 |
| #13 ensemble k=1 (shared) | test | off | 272 | 0.005981 | 0.007589 | 0.612 | 1.104 | 0.702 | 0.982 | 1.000 |
| #13 ensemble k=1 (shared) | test | on | 272 | 0.006007 | 0.007614 | 0.618 | 1.103 | 0.702 | 0.974 | 1.000 |
| #13 ensemble k=3 (shared) | test | off | 272 | 0.005208 | 0.006694 | 0.695 | 0.972 | 0.662 | 0.949 | 1.000 |
| #13 ensemble k=3 (shared) | test | on | 272 | 0.005274 | 0.006721 | 0.696 | 0.978 | 0.643 | 0.949 | 1.000 |
| #14 raster 256 px | test | off | 272 | 0.006023 | 0.007779 | 0.608 | 1.024 | 0.665 | 0.963 | 0.996 |
| #14 raster 256 px | test | on | 272 | 0.006030 | 0.007764 | 0.619 | 1.027 | 0.669 | 0.963 | 0.996 |
| #16 circular pad + shift aug | test | off | 272 | 0.005739 | 0.007319 | 0.647 | 1.019 | 0.669 | 0.967 | 0.996 |
| #16 circular pad + shift aug | test | on | 272 | 0.005702 | 0.007301 | 0.656 | 1.022 | 0.669 | 0.967 | 0.996 |
| #16b circular, retuned (260-trial sweep) | test | off | 272 | 0.005801 | 0.007348 | 0.660 | 1.147 | 0.739 | 0.982 | 1.000 |
| #16b circular, retuned (260-trial sweep) | test | on | 272 | 0.005763 | 0.007307 | 0.667 | 1.157 | 0.732 | 0.978 | 1.000 |
| #18 no attention | test | off | 272 | 0.005540 | 0.007073 | 0.679 | 0.985 | 0.651 | 0.949 | 0.996 |
| #18 no attention | test | on | 272 | 0.005553 | 0.007084 | 0.681 | 0.984 | 0.643 | 0.952 | 0.996 |
| #19 CBAM | test | off | 272 | 0.005903 | 0.007448 | 0.638 | 1.101 | 0.699 | 0.967 | 1.000 |
| #19 CBAM | test | on | 272 | 0.005843 | 0.007408 | 0.638 | 1.114 | 0.702 | 0.971 | 1.000 |
| #20 ECA | test | off | 272 | 0.005616 | 0.007083 | 0.660 | 1.121 | 0.706 | 0.963 | 1.000 |
| #20 ECA | test | on | 272 | 0.005584 | 0.007059 | 0.665 | 1.125 | 0.706 | 0.967 | 1.000 |
| #21 self-attn (all blocks) | test | off | 272 | 0.006660 | 0.008483 | 0.499 | 1.073 | 0.713 | 0.967 | 1.000 |
| #21 self-attn (all blocks) | test | on | 272 | 0.006693 | 0.008511 | 0.488 | 1.073 | 0.713 | 0.967 | 1.000 |
| #22 recon multi-task | test | off | 272 | 0.005471 | 0.006879 | 0.691 | 0.987 | 0.647 | 0.949 | 0.996 |
| #22 recon multi-task | test | on | 272 | 0.005463 | 0.006879 | 0.702 | 0.990 | 0.651 | 0.949 | 0.996 |
| #23 structure factor only | test | off | 272 | 0.005782 | 0.007404 | 0.664 | 0.964 | 0.662 | 0.945 | 0.996 |
| #23 structure factor only | test | on | 272 | 0.005808 | 0.007437 | 0.659 | 0.963 | 0.665 | 0.949 | 0.996 |
| #24 self-attn (stage 4 only) | test | off | 272 | 0.005940 | 0.007571 | 0.638 | 1.146 | 0.721 | 0.974 | 1.000 |
| #24 self-attn (stage 4 only) | test | on | 272 | 0.005884 | 0.007465 | 0.646 | 1.179 | 0.739 | 0.982 | 1.000 |
| #25 replicate (init seeds +1000) | test | off | 272 | 0.005388 | 0.006825 | 0.692 | 1.002 | 0.662 | 0.960 | 0.996 |
| #25 replicate (init seeds +1000) | test | on | 272 | 0.005352 | 0.006805 | 0.693 | 1.007 | 0.658 | 0.956 | 0.996 |
| #25 replicate (init seeds +2000) | test | off | 272 | 0.005541 | 0.006991 | 0.674 | 0.998 | 0.665 | 0.949 | 0.996 |
| #25 replicate (init seeds +2000) | test | on | 272 | 0.005557 | 0.007024 | 0.680 | 0.994 | 0.662 | 0.952 | 0.996 |
| #15 cell holdout (jitter_s0125) | holdout_jitter_s0125 | off | 155 | 0.005623 | 0.007040 | 0.673 | 1.068 | 0.710 | 0.974 | 1.000 |
| #15 cell holdout (jitter_s0125) | holdout_jitter_s0125 | on | 155 | 0.005559 | 0.006961 | 0.684 | 1.085 | 0.735 | 0.974 | 0.994 |
| #15 cell holdout (jitter_s0125) | test | off | 257 | 0.005497 | 0.006991 | 0.677 | 0.970 | 0.661 | 0.946 | 0.996 |
| #15 cell holdout (jitter_s0125) | test | on | 257 | 0.005449 | 0.006944 | 0.683 | 0.980 | 0.665 | 0.946 | 0.996 |

## FDTD arms (verified E; mean +- SE at arm level)

| arm | n | mean E | SE | max E | claimable | col |
|---|---|---|---|---|---|---|
| production LCB k=0.2 (jitter_s015) | 20 | 2.6472 | 0.0025 | 2.6675 | 20 | true_E60 |
| #5 kappa=0 (jitter_s015) | 20 | 2.6503 | 0.0020 | 2.6618 | 20 | true_E60 |
| #26 screen-only (jitter_s015) | 20 | 2.6171 | 0.0013 | 2.6296 | 17 | true_E60 |
| #7 single member (jitter_s015) | 20 | 2.6493 | 0.0015 | 2.6636 | 20 | true_E60 |
| #6 random (jitter_s006) | 20 | 2.6128 | 0.0019 | 2.6242 | 2 | true_E60 |
|     optimizer ref (jitter_s006) | 20 | 2.6393 | 0.0015 | 2.6508 | 20 | true_E60 |
| #6 random (jitter_s008) | 20 | 2.6235 | 0.0018 | 2.6341 | 2 | true_E60 |
|     optimizer ref (jitter_s008) | 20 | 2.6481 | 0.0017 | 2.6657 | 20 | true_E60 |
| #6 random (jitter_s010) | 20 | 2.6189 | 0.0021 | 2.6351 | 3 | true_E60 |
|     optimizer ref (jitter_s010) | 20 | 2.6482 | 0.0018 | 2.6626 | 20 | true_E60 |
| #6 random (jitter_s0125) | 20 | 2.6148 | 0.0020 | 2.6275 | 5 | true_E60 |
|     optimizer ref (jitter_s0125) | 20 | 2.6450 | 0.0017 | 2.6609 | 20 | true_E60 |
| #6 random (jitter_s015) | 20 | 2.6029 | 0.0021 | 2.6214 | 2 | true_E60 |
|     optimizer ref (jitter_s015) | 20 | 2.6472 | 0.0025 | 2.6675 | 20 | true_E60 |
| #6 random (radius_s015) | 20 | 2.6009 | 0.0018 | 2.6162 | 4 | true_E60 |
|     optimizer ref (radius_s015) | 20 | 2.6486 | 0.0008 | 2.6547 | 20 | true_E60 |
| #6 random (radius_s020) | 20 | 2.6034 | 0.0020 | 2.6192 | 6 | true_E60 |
|     optimizer ref (radius_s020) | 20 | 2.6510 | 0.0016 | 2.6640 | 20 | true_E60 |
| #6 random (radius_s025) | 20 | 2.5922 | 0.0027 | 2.6215 | 6 | true_E60 |
|     optimizer ref (radius_s025) | 20 | 2.6443 | 0.0018 | 2.6564 | 20 | true_E60 |

```
Footnotes (pre-registered):
  a. naive_split is evaluated on its OWN naive test set -- deliberately
     not sample-for-sample comparable; it measures the optimism of naive
     evaluation.
  b. ensemble-size rows use shared splits: indicative of size scaling
     only; the deployed model is k-fold-rotated and lives on the
     reference row, not this curve. (k=5 shared == the #8 row.)
  c. SmoothL1 rows have no learned s: s columns are ensemble member
     disagreement only (the v1 convention); expected under-coverage IS
     the result.
  d. FDTD arms report mean +- SE (arm level) and per-candidate tables
     separately; per-candidate differences below the 0.30 % within-sigma
     floor are 'not resolvable'. Random-baseline vs optimizer is
     equal-budget only over all 8 cells (160 vs 160 solves).
  e. All s-vs-error columns are RMS-vs-RMS or coverage counts -- no
     error distribution assumed (never s vs mean |error|).
  f. holdout_* rows: 'test' = in-distribution split of the remaining
     cells; the holdout row is the held-out cell (OOD generalization).
  g. attention rows replace the SE gate in every residual block with
     `--attention {none,cbam,eca,sa}`; everything else is the deployed
     v2 recipe.  none = identity (SE removed); cbam = channel + spatial
     (Woo 2018); eca = 1D-conv channel attn, no bottleneck (Wang 2020);
     sa = multi-head spatial self-attention at every block (stage-1
     cost dominates; also swaps the gate for a LayerNorm residual, a
     disclosed confound).  Rows share the deployed test split.
  h. #16 retrains with circular conv padding (torus topology) + cyclic
     shift augmentation; #22 adds a decoder reconstructing the raster
     channel from the pre-GAP features (loss + 0.1 * recon MSE) --
     regression head, validation loss and early stopping unchanged.
     Both use the deployed best_params (no re-sweep), test split shared.

```
