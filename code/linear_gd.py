"""Linear Regression with Gradient Descent.

Contains
  * LinearRegressionGD  - multi-feature linear regression trained with batch
                          gradient descent (NumPy only).
  * the leak-free preprocessing pipeline (entry-error detection, row removal,
    imputation, feature engineering, standardization),
  * the experiments used for model selection, and
  * the code that writes predictions.csv.

Sources: the gradient-descent update follows the class linear-regression demos;
the closed-form check uses numpy.linalg.lstsq.
"""
# imports
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration and constants
SEED = 42
VAL_FRACTION = 0.20
BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results"
FIGURES_DIR = BASE / "figures"

# Target and raw features
TARGET = "Performance Index"
RAW_FEATURES = [
    "Hours Studied",
    "Weekly Study Hours",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Commute Minutes",
]
# Columns that must be present for a training row to be kept. The other raw
# features (Weekly Study Hours, Commute Minutes) are imputed if they are missing.
CORE_FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
SLEEP_SQ = "Sleep Hours Sq"  # engineered: (Sleep Hours - training mean) ** 2

# Data-entry rules: a value for which the rule is True is impossible/implausible.
ENTRY_RULES = {
    "Hours Studied": (lambda s: (s < 0) | (s > 9),
                      "outside 0-9 (the histogram shows the bulk of values in 1-9)"),
    "Weekly Study Hours": (lambda s: (s < 0) | (s > 168),
                           "negative or more than the 168 hours in a week"),
    "Previous Scores": (lambda s: (s < 0) | (s > 100), "outside the 0-100 score range"),
    "Extracurricular Activities": (lambda s: ~s.isin([0, 1]) & s.notna(), "not 0 or 1"),
    "Sleep Hours": (lambda s: (s < 0) | (s > 24), "negative or more than 24 hours"),
    "Sample Question Papers Practiced": (lambda s: s < 0, "negative count"),
    "Commute Minutes": (lambda s: s < 0, "negative duration"),
}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LinearRegressionGD:
    """Linear regression y_hat = X w + b fit by batch gradient descent.

    Cost:      J(w, b) = 1 / (2m) * sum_i (y_hat_i - y_i)^2
    Gradients: dJ/dw = X^T (y_hat - y) / m      dJ/db = mean(y_hat - y)
    Stopping:  |J_previous - J| < tol, or max_iter iterations.
    """
    # Initialize the linear regression model with gradient descent.
    def __init__(self, learning_rate=0.1, max_iter=20000, tol=1e-12):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.n_iter = 0
        self.converged = False
        self.diverged = False

    def fit(self, X, y):
        """Fit the linear regression model using batch gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []
        self.converged = self.diverged = False
        # Perform batch gradient descent to update the weights and bias.
        for i in range(self.max_iter):
            error = X @ self.weights + self.bias - y
            cost = error @ error / (2 * m)
            self.cost_history.append(cost)
            if not np.isfinite(cost):  # learning rate too large
                self.diverged = True
                break
            if i > 0 and abs(self.cost_history[-2] - cost) < self.tol:
                self.converged = True
                break
            self.weights -= self.learning_rate * (X.T @ error) / m
            self.bias -= self.learning_rate * error.mean()
        self.n_iter = len(self.cost_history)
        return self
    # Make predictions using the trained linear regression model.
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return np.asarray(X, dtype=float) @ self.weights + self.bias


def closed_form(X, y):
    """Least-squares solution (bias first) used to verify the gradient descent fit."""
    A = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def load_table(name):
    """Load <name>.csv if present, otherwise <name>.xlsx (both live next to this file)."""
    csv, xlsx = BASE / f"{name}.csv", BASE / f"{name}.xlsx"
    return pd.read_csv(csv) if csv.exists() else pd.read_excel(xlsx)

# Data handling functions for loading, cleaning, and summarizing raw data.
def flag_entry_errors(df):
    """Return {column: boolean mask of data-entry errors} for the columns in df."""
    return {col: rule(df[col]) for col, (rule, _) in ENTRY_RULES.items() if col in df}

# Mask data-entry errors by replacing them with NaN.
def mask_entry_errors(df):
    """Copy of df with every data-entry error replaced by NaN."""
    out = df.copy()
    for col, mask in flag_entry_errors(out).items():
        out.loc[mask, col] = np.nan
    return out

# Summarize the raw data with basic statistics and entry error counts.
def summarize_raw(df):
    """Section 2.1/2.3 numbers: min, max, mean, missing cells and entry errors per column."""
    errors = flag_entry_errors(df)
    return pd.DataFrame({
        "min": df.min(), "max": df.max(), "mean": df.mean().round(2),
        "missing": df.isna().sum(),
        "entry_errors": pd.Series({c: int(m.sum()) for c, m in errors.items()}),
    }).fillna({"entry_errors": 0}).astype({"entry_errors": int})

# Prepare the training data by handling entry errors and dropping incomplete rows.
def prepare_training_rows(raw):
    """Training-side cleaning that needs no statistics (so it cannot leak).

    1. entry errors -> NaN   2. drop rows missing the target or a core feature.
    Returns the cleaned frame and a dict of counts for the report.
    """
    masked = mask_entry_errors(raw)
    keep = masked[CORE_FEATURES + [TARGET]].notna().all(axis=1)
    info = {
        "rows_raw": len(raw),
        "rows_dropped": int((~keep).sum()),
        "rows_kept": int(keep.sum()),
        "cells_missing_after_masking": int(masked.isna().sum().sum()),
    }
    return masked[keep].reset_index(drop=True), info

# Preprocessing class for imputation, feature engineering, and standardization.
class Preprocessor:
    """Imputation + feature engineering + standardization, fit on training data only.

    Order: entry errors -> NaN, impute with training medians, add the Sleep Hours
    squared term, standardize with training mean/std. The same object transforms
    the validation and test features.
    """
    # Fit the preprocessor on the training data, computing medians, means, and standard deviations.
    def fit(self, df):
        masked = mask_entry_errors(df[RAW_FEATURES])
        self.medians = masked.median()
        design = self._design(masked.fillna(self.medians), fitting=True)
        self.mean = design.mean()
        self.std = design.std(ddof=0).replace(0, 1.0)
        return self
    # Transform the data using the fitted preprocessor, applying imputation, feature engineering, and standardization.
    def _design(self, df, fitting=False):
        df = df.copy()
        if fitting:
            self.sleep_center = df["Sleep Hours"].mean()
        df[SLEEP_SQ] = (df["Sleep Hours"] - self.sleep_center) ** 2
        return df
    # Apply the design matrix transformation, including the squared Sleep Hours term.
    def transform(self, df, columns):
        masked = mask_entry_errors(df[RAW_FEATURES]).fillna(self.medians)
        design = self._design(masked)
        return ((design[columns] - self.mean[columns]) / self.std[columns]).to_numpy()

# Split the data into training and validation sets.
def split_train_val(df, val_fraction=VAL_FRACTION, seed=SEED):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_val = int(round(len(df) * val_fraction))
    return df.iloc[idx[n_val:]].reset_index(drop=True), df.iloc[idx[:n_val]].reset_index(drop=True)

# Fit the linear regression model on the training data and evaluate it on the validation set.
def fit_score(train, val, columns, lr, max_iter=20000, tol=1e-12):
    """Fit preprocessing + model on train, return (model, preprocessor, val MSE)."""
    pre = Preprocessor().fit(train)
    model = LinearRegressionGD(lr, max_iter, tol).fit(pre.transform(train, columns), train[TARGET])
    pred = model.predict(pre.transform(val, columns))
    return model, pre, float(np.mean((val[TARGET] - pred) ** 2))


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
# Reporting helpers for saving tables and plotting results.
def to_markdown(df, floatfmt="{:.4f}"):
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.astype(object).iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(rows)

# Save a DataFrame as both CSV and Markdown tables in the results directory.
def save_table(df, stem, floatfmt="{:.4f}"):
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{stem}.csv")
    (RESULTS_DIR / f"{stem}.md").write_text(to_markdown(df, floatfmt) + "\n", encoding="utf-8")

# Plot the training cost curves for different learning rates.
def plot_cost_curves(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for lr, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, label=f"lr = {lr}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost J(w, b)")
    ax.set_title("Training cost vs. iteration"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

# Plot histograms of the raw features with entry-error cutoffs marked.
def plot_histogram(hours_studied, previous_scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)
    # One panel per feature (raw data), with the entry-error cutoff marked.
    panels = [(hours_studied, "Hours Studied", 9, np.arange(0, 100, 1), True),
              (previous_scores, "Previous Scores", 100, np.arange(30, 201, 5), True)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (values, name, cutoff, bins, log_y) in zip(axes, panels):
        ax.hist(values.dropna(), bins=bins)
        ax.axvline(cutoff, color="C3", linestyle="--", label=f"cutoff ({cutoff})")
        if log_y:
            ax.set_yscale("log")
        n_err = int((values > cutoff).sum())
        ax.set_title(f"{name} ({n_err} values above cutoff)")
        ax.set_xlabel(name); ax.set_ylabel("Frequency" + (" (log scale)" if log_y else ""))
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------
# Experiments and final predictions
# --------------------------------------------------------------------------
# Define the feature sets and learning rates for the experiments.
FEATURE_SETS = {
    "A: Previous Scores only": ["Previous Scores"],
    "B: A + Hours Studied": ["Previous Scores", "Hours Studied"],
    "C: 5 core features": CORE_FEATURES,
    "D: C + Sleep Hours Sq": CORE_FEATURES + [SLEEP_SQ],
    "E: D + Weekly Study Hours": CORE_FEATURES + [SLEEP_SQ, "Weekly Study Hours"],
    "F: D + Commute Minutes": CORE_FEATURES + [SLEEP_SQ, "Commute Minutes"],
    "G: all 7 + Sleep Hours Sq": RAW_FEATURES + [SLEEP_SQ],
}
# Main function to run the experiments and generate results.
LEARNING_RATES = [0.001, 0.01, 0.1, 0.5]
# Base learning rate for the experiments.
BASE_LR = 0.1

# Main entry point for running the experiments.
def main():
    # Set the random seed for reproducibility.
    np.random.seed(SEED)
    raw_train, raw_test = load_table("train"), load_table("test_features")

    # --- Section 2: exploration and data handling -------------------------
    summary = summarize_raw(raw_train)
    summary.index.name = "column"
    save_table(summary, "summary_stats", "{:.2f}")
    clean, info = prepare_training_rows(raw_train)
    print("=== Data handling ===")
    print(summary, "\n", info)

    # --- Section 3.2: validation split ------------------------------------
    train, val = split_train_val(clean)
    print(f"\nSplit: {len(train)} train / {len(val)} validation rows (seed {SEED})")

    # --- Section 3.4: feature-set experiments (fixed learning rate) --------
    rows = []
    for name, cols in FEATURE_SETS.items():
        model, _, mse = fit_score(train, val, cols, BASE_LR)
        rows.append({"experiment": name, "n_features": len(cols), "learning_rate": BASE_LR,
                     "iterations": model.n_iter, "val MSE": mse, "val RMSE": np.sqrt(mse)})
    experiments = pd.DataFrame(rows).set_index("experiment")
    best_name = experiments["val RMSE"].idxmin()
    best_cols = FEATURE_SETS[best_name]

    # --- Section 3.3: learning-rate comparison on the best feature set ----
    histories, lr_rows = {}, []
    for lr in LEARNING_RATES:
        model, _, mse = fit_score(train, val, best_cols, lr)
        histories[lr] = model.cost_history
        lr_rows.append({"learning_rate": lr, "iterations": model.n_iter,
                        "converged": model.converged, "diverged": model.diverged,
                        "val MSE": mse, "val RMSE": np.sqrt(mse)})
        rows.append({"experiment": f"{best_name} (lr sweep)", "n_features": len(best_cols),
                     "learning_rate": lr, "iterations": model.n_iter,
                     "val MSE": mse, "val RMSE": np.sqrt(mse)})
    lr_table = pd.DataFrame(lr_rows).set_index("learning_rate")
    plot_cost_curves({lr: h for lr, h in histories.items() if np.isfinite(h[-1])},
                     FIGURES_DIR / "cost_curves.png")
    save_table(pd.DataFrame(rows).set_index("experiment"), "experiments")
    save_table(lr_table, "learning_rates")

    plot_histogram(raw_train["Hours Studied"], raw_train["Previous Scores"], FIGURES_DIR / "histogram.png")

    # Choose the fastest learning rate that converged to (nearly) the best RMSE.
    ok = lr_table[lr_table["converged"] & ~lr_table["diverged"]]
    ok = ok[ok["val RMSE"] <= ok["val RMSE"].min() + 1e-3]
    final_lr = float(ok["iterations"].idxmin())

    # --- Gradient check against the closed-form solution -------------------
    model, pre, val_mse = fit_score(train, val, best_cols, final_lr)
    X_tr = pre.transform(train, best_cols)
    gap = np.max(np.abs(np.r_[model.bias, model.weights] - closed_form(X_tr, train[TARGET])))

    print("\n=== Model selection ===")
    print(experiments.to_string(float_format=lambda v: f"{v:.4f}"))
    print(lr_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest feature set: {best_name} | chosen lr: {final_lr} | "
          f"iterations: {model.n_iter}")
    print(f"Validation MSE: {val_mse:.4f} | RMSE: {np.sqrt(val_mse):.4f}")
    print(f"Max |GD - closed form| over all parameters: {gap:.2e}")

    # --- Final model: refit preprocessing + weights on all cleaned training rows
    pre_full = Preprocessor().fit(clean)
    final = LinearRegressionGD(final_lr).fit(pre_full.transform(clean, best_cols), clean[TARGET])
    predictions = final.predict(pre_full.transform(raw_test, best_cols))
    submission = pd.DataFrame({"ID": raw_test["ID"], TARGET: predictions})
    submission.to_csv(BASE / "predictions.csv", index=False)
    print(f"\nWrote predictions.csv: {submission.shape}, missing = {int(submission.isna().sum().sum())}")

    (RESULTS_DIR / "final_model.md").write_text(
        f"- Feature set: {best_name}\n- Columns: {', '.join(best_cols)}\n"
        f"- Learning rate: {final_lr}, tol: {final.tol}, max_iter: {final.max_iter}\n"
        f"- Iterations (validation fit / final refit): {model.n_iter} / {final.n_iter}\n"
        f"- Validation MSE: {val_mse:.4f}, RMSE: {np.sqrt(val_mse):.4f}\n"
        f"- Max |GD - closed form|: {gap:.2e}\n"
        f"- Split: {len(train)} train / {len(val)} val, seed {SEED}\n"
        f"- Rows: {info}\n", encoding="utf-8")

# run main
if __name__ == "__main__":
    main()
