# California Housing Regression with PyTorch

A structured neural-network regression experiment comparing 12 model configurations on the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (sklearn built-in). The goal is to predict median house values (in $100k units) and systematically measure how architecture, loss function, and optimizer choice affect performance.


## What's being compared

Three architectures are tested: 
    - A small two-layer network `[16, 8]`, a
    - A single wide layer `[64]`, 
    - And a deeper network `[128, 64, 32]`. 

Each is paired with two loss functions — `MSELoss` (standard squared error, sensitive to outliers) and `SmoothL1Loss` (Huber-like, more robust to outliers) — and two optimizers — `Adam` at lr=1e-3 and `SGD` with momentum at lr=1e-2. That gives 12 total runs.

All other settings are fixed across runs: 80 epochs, batch size 64, L2 weight decay 1e-4, and a `ReduceLROnPlateau` scheduler.


## Key design decisions

**3-way train / val / test split**
The dataset is split 70% train, 15% validation, and 15% test. Validation loss drives learning-rate scheduling and determines which epoch's weights are saved. The test set is only touched once, at the very end. This prevents the common mistake of picking the best model based on test performance, which would be data leakage.

**Model checkpointing instead of retraining**
The best weights (lowest val loss) are saved with `torch.save` during training. Final evaluation loads those weights directly — no redundant retraining, fully reproducible.

**Learning-rate scheduling**
`ReduceLROnPlateau` halves the learning rate after 10 epochs without val-loss improvement. This lets longer runs continue improving without manual tuning.

**Standardised evaluation**
Even though `MSELoss` and `SmoothL1Loss` have different scales during training, every model is evaluated with the same MSE and R2 on the inverse-scaled test set, making cross-experiment comparisons valid.


## Results

After running the notebook, full results are saved to `lab2_results_summary.csv` and displayed as a styled table inside the notebook, sorted by MSE.


## Plots generated

All plots are saved to `lab2_plots/` and model weights to `lab2_weights/`.

`metrics_comparison.png` — side-by-side MSE and R2 bar charts for all 12 runs.

`best_model_diagnostics.png` — prediction vs true values and a residuals plot for the best model.

`all_loss_curves.png` — a grid of all 12 train/val loss curves in one figure.

`{model}_name_loss.png` — individual train and val loss curves, one file per run.

`{model}.pt` — saved model checkpoint for each run. 


## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/lab2-california-housing.git
cd lab2-california-housing

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib tqdm

# Launch the notebook
jupyter notebook Lab2_California_Housing_Improved.ipynb
```

Tested with Python 3.10, PyTorch 2.x, and scikit-learn 1.x.


## File structure

```
.
├── Lab2_California_Housing_Improved.ipynb   # Main notebook
├── lab2_results_summary.csv                 # Auto-generated results table
├── lab2_plots/                              # All figures
│   ├── metrics_comparison.png
│   ├── best_model_diagnostics.png
│   ├── all_loss_curves.png
│   └── {run_name}_loss.png  (x12)
└── lab2_weights/                            # Saved model checkpoints
    └── {run_name}.pt        (x12)
```


## Results and Conclusions

The best-performing configuration was the deep architecture [128, 64, 32] trained with SmoothL1Loss and the Adam optimizer, achieving an MSE of 0.2536 and an R² of 0.8065 on the test set.

Architecture played a decisive role in performance. The deeper network consistently outperformed both shallower alternatives, indicating that additional depth improves the model’s ability to capture complex relationships in the data. Interestingly, the single-layer model [64] underperformed even the smaller two-layer network [16, 8], suggesting that depth is more important than width alone.

Optimizer choice also had a significant impact. Adam outperformed SGD across all configurations, achieving both faster convergence and better final performance. The best SGD-based model showed approximately 13% higher MSE than the best Adam model, highlighting the advantage of adaptive optimization in this setting.

Regarding loss functions, SmoothL1Loss provided a small but consistent improvement over MSELoss, particularly in deeper architectures. This likely reflects its robustness to outliers in housing price data.

Finally, training curves indicate that most performance gains occur within the first 20–30 epochs, after which improvements plateau. This suggests that early stopping could reduce training time without sacrificing model quality.

## Best model

- Architecture: `[128, 64, 32]`
- Loss: SmoothL1Loss
- Optimizer: Adam
- Test MSE: **0.2536**
- Test R²: **0.8065**

## Possible extensions

- Add BatchNorm layers and test whether they stabilise training.
- Try adding residual connections.
- Swap in AdamW or RMSProp as additional optimizers.
- Run a proper hyperparameter search with Optuna or Ray Tune.
