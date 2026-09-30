"""Part 3:

Five methods are computed for every raw feature, on the same validation split:
  permutation importance, drop-one-feature retraining, single-feature models,
  standardized coefficients (all with LinearRegressionGD) and random-forest
  importance (scikit-learn, allowed in Part 3 only).

The analysis is repeated on several random splits to check stability.

"""
# imports
import numpy as np
import pandas as pd

from linear_gd import (BASE_LR, FIGURES_DIR, RAW_FEATURES, SEED, SLEEP_SQ, TARGET,
                       LinearRegressionGD, Preprocessor, load_table, prepare_training_rows,
                       rmse, save_table, split_train_val)

# Sleep Hours enters the linear model twice (linear + squared term), so the two
# columns are treated as one feature "group" in every method.
GROUPS = {f: [f] for f in RAW_FEATURES}
GROUPS["Sleep Hours"] = ["Sleep Hours", SLEEP_SQ]
ALL_COLUMNS = [c for cols in GROUPS.values() for c in cols]
STABILITY_SEEDS = [1, 2, 3, 4, 5]
N_PERMUTATIONS = 20
LR = 0.5  # fast and converges for standardized features (see learning-rate table)

# Feature importance methods
def fit_linear(train, columns):
    pre = Preprocessor().fit(train)
    model = LinearRegressionGD(LR).fit(pre.transform(train, columns), train[TARGET])
    return pre, model

# Permutation importance
def permutation_importance(train, val, rng):
    pre, model = fit_linear(train, ALL_COLUMNS)
    base = rmse(val[TARGET], model.predict(pre.transform(val, ALL_COLUMNS)))
    scores = {}
    # Compute permutation importance for each raw feature.
    for feat in RAW_FEATURES:
        increases = []
        # Shuffle the feature and measure the increase in RMSE.
        for _ in range(N_PERMUTATIONS):
            shuffled = val.copy()
            shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
            pred = model.predict(pre.transform(shuffled, ALL_COLUMNS))
            increases.append(rmse(val[TARGET], pred) - base)
        scores[feat] = float(np.mean(increases))
    return scores

# Drop-one-feature importance
def drop_one_importance(train, val):
    # Compute drop-one-feature importance for each raw feature.
    def val_rmse(columns):
        pre, model = fit_linear(train, columns)
        return rmse(val[TARGET], model.predict(pre.transform(val, columns)))
    # Compute the full model's RMSE as a baseline.
    full = val_rmse(ALL_COLUMNS)
    return {f: val_rmse([c for c in ALL_COLUMNS if c not in GROUPS[f]]) - full for f in RAW_FEATURES}

# Single-feature importance
def single_feature_importance(train, val):
    # Score = how far below the predict-the-training-mean baseline the RMSE falls.
    baseline = rmse(val[TARGET], np.full(len(val), train[TARGET].mean()))
    out = {}
    # Compute single-feature importance for each feature group.
    for feat, cols in GROUPS.items():
        pre, model = fit_linear(train, cols)
        out[feat] = baseline - rmse(val[TARGET], model.predict(pre.transform(val, cols)))
    return out

# Standardized coefficients importance
def standardized_coefficients(train):
    pre, model = fit_linear(train, ALL_COLUMNS)
    coef = dict(zip(ALL_COLUMNS, np.abs(model.weights)))
    return {f: float(sum(coef[c] for c in cols)) for f, cols in GROUPS.items()}

# Random forest importance
def forest_importance(train, seed):
    from sklearn.ensemble import RandomForestRegressor

    X = train[RAW_FEATURES].fillna(train[RAW_FEATURES].median())
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    rf.fit(X, train[TARGET])
    return dict(zip(RAW_FEATURES, rf.feature_importances_))

# Run all feature importance methods and return a table of results.
def run_all(clean, seed):
    train, val = split_train_val(clean, seed=seed)
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "Permutation (dRMSE)": permutation_importance(train, val, rng),
        "Drop-one (dRMSE)": drop_one_importance(train, val),
        "Single-feature (RMSE gain)": single_feature_importance(train, val),
        "Std. coefficient (|w|)": standardized_coefficients(train),
        "Random forest": forest_importance(train, seed),
    })
    table.index.name = "feature"
    return table

# Plot relationships between top features and the target variable.
def top3(series):
    return list(series.sort_values(ascending=False).index[:3])

# Plot the top 3 features against the target variable and the correlation matrix.
def plot_relationships(clean, top_features):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create the figures directory if it doesn't exist.
    FIGURES_DIR.mkdir(exist_ok=True)
    sample = clean.sample(min(2000, len(clean)), random_state=SEED)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # Plot each of the top 3 features against the target variable.
    for ax, feat in zip(axes, top_features):
        ax.scatter(sample[feat], sample[TARGET], s=6, alpha=0.4)
        means = clean.groupby(feat)[TARGET].mean() if clean[feat].nunique() <= 12 else None
        # Compute the mean of the target variable for each unique value of the feature.
        if means is not None:
            ax.plot(means.index, means.values, color="C3", marker="o", label="mean per value")
            ax.legend()
        ax.set_xlabel(feat); ax.set_ylabel(TARGET)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "top_features_vs_target.png", dpi=200)
    plt.close(fig)

    # Plot the correlation matrix of all raw features and the target variable.
    corr = clean[RAW_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.columns)
    # Add text annotations for each cell in the correlation matrix.
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=200); plt.close(fig)
    corr.index.name = "feature"
    save_table(corr, "correlation_matrix", "{:.3f}")

# Main function to run the feature importance analysis.
def main():
    clean, _ = prepare_training_rows(load_table("train"))

    # Main analysis on the seed used for model selection.
    scores = run_all(clean, SEED)
    ranks = scores.rank(ascending=False)
    ranks.columns = [c.split(" (")[0] + " rank" for c in ranks.columns]
    scores["Mean rank"] = ranks.mean(axis=1)
    save_table(scores.sort_values("Mean rank"), "importance_scores", "{:.4f}")
    save_table(ranks.join(scores["Mean rank"]).sort_values("Mean rank"), "importance_ranks", "{:.1f}")
    overall = list(scores["Mean rank"].sort_values().index[:3])
    print("=== Importance scores (seed %d) ===" % SEED)
    print(scores.sort_values("Mean rank").to_string(float_format=lambda v: f"{v:.4f}"))
    print("Top three (mean rank):", overall)

    # Stability across random splits.
    stability = {}
    for s in STABILITY_SEEDS:
        res = run_all(clean, s)
        row = {m: " > ".join(top3(res[m])) for m in res.columns}
        row["Mean-rank top 3"] = " > ".join(res.rank(ascending=False).mean(axis=1).sort_values().index[:3])
        stability[f"seed {s}"] = row
    stability = pd.DataFrame(stability).T
    stability.index.name = "split"
    save_table(stability, "importance_stability")
    print("\n=== Stability (top three per method) ===")
    print(stability.to_string())

    plot_relationships(clean, overall)

# run main
if __name__ == "__main__":
    main()
