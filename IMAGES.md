# Image gallery

Supplementary figures for SEER and the Photra-2.7k dataset. Silicon at
production numerics (res 60, a = 650 nm, r/a = 0.35, 7x7 supercell).

## Data scaling and honest uncertainty

![Learning curve](assets/gallery/learning_curve.png)

As the training set grows, ensemble test error (red) crosses below the
0.30% ranking-resolvability floor (grey band) near N = 1,500, while
ranking fidelity rho (blue, right axis) keeps climbing. The purple curve
is the model's own error forecast, which tracks measured error to within
8% for N >= 200 and stays cautious when data is scarce.
