# Curated result figures

Headline figures only; every one is regenerable, and the Source column names
the script output it was copied from. Diagnostics and intermediate figures
stay untracked in their run directories.

| File | Shows | Source |
|---|---|---|
| `e_vs_sigma_scatter.png` | Enhancement vs. disorder strength across Photra-2.7k, per realization | `scripts/FDTD_solver/data_production/figs/fig6_realization_scatter.png` (`run_dataset.py analyze`) |
| `E_histograms.png` | Within-cell enhancement distributions (the arrangement spread the search exploits) | `.../figs/fig11_E_histograms.png` |
| `layout_gallery.png` | Example manufacturable layouts per class and strength | `.../figs/fig8_layout_gallery.png` |
| `spectra_showcase.png` | Absorption spectra of representative layouts vs. the flat film | `.../figs/fig7_spectra_showcase.png` |
| `seer_pred_by_sigma.png` | Deployed SEER predictions vs. FDTD labels by disorder strength | `runs/surrogate_128_fft_nll_sweep/pred_by_sigma.png` |
| `seer_spearman_by_cell.png` | Deployed SEER within-cell ranking (Spearman rho per cell) | `runs/surrogate_128_fft_nll_sweep/spearman_by_cell.png` |
| `seer_uq_calibration.png` | Calibration of the predicted error s (coverage vs. nominal) | `runs/surrogate_128_fft_nll_sweep/uq/uq_calibration.png` |
| `learning_curve.png` | Test error vs. training-set size (floor crossing near N = 1,500) | `runs/learning_curve_seed137_v2/learning_curve.png` |
| `inverse_design_vs_dataset.png` | Verified designed layouts vs. the Photra-2.7k distribution, all cells | `runs/inverse_v2/fig_inverse_v2_vs_dataset.png` |
| `saliency_champion.png` | SEER saliency on the champion layout | `runs/interpretability/champ_v2/figure.png` |
| `saliency_randomization_check.png` | Attribution band concentration, trained vs. weight-randomized control | `runs/interpretability/validation/step1_randomization.png` |
