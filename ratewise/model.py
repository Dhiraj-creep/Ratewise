from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import features, groups, validate_frame, safe_csv, NUMERIC

class BoundedRegressor(RegressorMixin, BaseEstimator):
    """Use the same [1,5] prediction rule for CV, evaluation and inference."""
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self
    def predict(self, X):
        return np.clip(self.estimator_.predict(X), 1, 5)

def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None}

def split_indices(frame, seed):
    g = groups(frame)
    traincal, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(frame, groups=g))
    train_local, cal_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1).split(frame.iloc[traincal], groups=g.iloc[traincal]))
    return traincal[train_local], traincal[cal_local], test, g

def pipeline(model, columns):
    numeric = [x for x in columns if x in NUMERIC]
    categorical = [x for x in columns if x not in NUMERIC]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                               ("scale", StandardScaler())]), numeric),
        ("category", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                    max_categories=60, sparse_output=False), categorical)
    ])
    return Pipeline([("preprocess", pre), ("model", BoundedRegressor(model))])

def verify_evidence(frame, evidence_dir):
    missing = []
    for digest in frame.evidence_sha256.unique():
        paths = [Path(evidence_dir) / f"{digest}.{ext}" for ext in ["txt", "html"]]
        if not any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest for p in paths):
            missing.append(digest)
    if missing:
        raise ValueError(f"{len(missing)} evidence files are absent or have changed. Collect via the app or supply the matching evidence directory.")
    return "Evidence file hashes verified; page contents still require a human source audit."

def train(frame, output_dir, mode="existing", seed=42, evidence_dir=None, fixture=False, fast=False):
    if mode not in {"existing", "cold_start"}:
        raise ValueError("Mode must be existing or cold_start.")
    clean, quality = validate_frame(frame)
    if len(clean) < 100 or groups(clean).nunique() < 50:
        raise ValueError("Training requires at least 100 unique products and 50 product families. Plan for about 1,000 products for the final study.")
    if clean.rating.nunique() < 3:
        raise ValueError("At least three distinct observed ratings are needed for a meaningful regression experiment.")
    if not fixture and evidence_dir is None:
        raise ValueError("An evidence directory is required for the Amazon experiment.")
    provenance = "SYNTHETIC SOFTWARE TEST ONLY — not Amazon evidence" if fixture else verify_evidence(clean, evidence_dir)
    X, y = features(clean, mode), clean.rating
    train_i, cal_i, test_i, g = split_indices(clean, seed)
    cv = GroupKFold(n_splits=min(5, g.iloc[train_i].nunique()))
    candidates = {
        "Mean baseline": (DummyRegressor(), {}),
        "Ridge regression": (Ridge(), {"model__estimator__alpha": [1., 10., 100.]}),
        "Random forest": (RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=1),
                          {"model__estimator__min_samples_leaf": [3, 8], "model__estimator__max_features": [.7, 1.]}),
        "Extra trees": (ExtraTreesRegressor(n_estimators=120, random_state=seed, n_jobs=1),
                        {"model__estimator__min_samples_leaf": [3, 8]}),
        "Gradient boosting": (GradientBoostingRegressor(random_state=seed, n_estimators=120, max_depth=2),
                              {"model__estimator__learning_rate": [.03, .08], "model__estimator__min_samples_leaf": [5, 12]})
    }
    if fast:
        candidates = {k: v for k, v in candidates.items() if k in {"Mean baseline", "Ridge regression"}}
    searches, leaderboard, cv_rows = {}, [], []
    for name, (model, grid) in candidates.items():
        search = GridSearchCV(pipeline(model, X.columns), grid, scoring="neg_mean_absolute_error", cv=cv,
                              n_jobs=1, refit=True, error_score="raise")
        search.fit(X.iloc[train_i], y.iloc[train_i], groups=g.iloc[train_i])
        searches[name] = search
        leaderboard.append({"model": name, "CV_MAE": float(-search.best_score_),
                            "CV_SD": float(search.cv_results_["std_test_score"][search.best_index_]),
                            "parameters": json.dumps(search.best_params_)})
        for mean, std, params in zip(search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"], search.cv_results_["params"]):
            cv_rows.append({"model": name, "CV_MAE": -float(mean), "CV_SD": float(std), "parameters": json.dumps(params)})
    ranking = pd.DataFrame(leaderboard).sort_values("CV_MAE", kind="stable")
    winner = ranking.iloc[0]["model"]
    fitted = searches[winner].best_estimator_
    # Family-maximum residuals calibrate against independent family units, not correlated variants.
    residuals = np.abs(y.iloc[cal_i].to_numpy() - fitted.predict(X.iloc[cal_i]))
    family_residuals = pd.DataFrame({"group": g.iloc[cal_i].to_numpy(), "error": residuals}).groupby("group").error.max().to_numpy()
    rank = int(np.ceil((len(family_residuals) + 1) * .9))
    q = float(np.sort(family_residuals)[rank - 1]) if rank <= len(family_residuals) else 4.
    pred = fitted.predict(X.iloc[test_i])
    low, high = np.maximum(1, pred - q), np.minimum(5, pred + q)
    baseline = searches["Mean baseline"].best_estimator_.predict(X.iloc[test_i])
    predictions = clean.iloc[test_i][["asin", "category", "rating"]].copy()
    predictions["prediction"] = pred
    predictions["lower_90"] = low
    predictions["upper_90"] = high
    predictions["residual"] = predictions.rating - pred
    predictions["absolute_error"] = predictions.residual.abs()
    predictions["covered"] = (predictions.rating >= low) & (predictions.rating <= high)
    predictions["product_family"] = g.iloc[test_i].to_numpy()
    importance = permutation_importance(fitted, X.iloc[test_i], y.iloc[test_i], n_repeats=5,
                                       scoring="neg_mean_absolute_error", random_state=seed, n_jobs=1)
    importance_df = pd.DataFrame({"feature": X.columns, "MAE_increase": importance.importances_mean,
                                  "SD": importance.importances_std}).sort_values("MAE_increase", ascending=False)
    slices = []
    for category, part in predictions.groupby("category"):
        slices.append({"category": category, "products": len(part), **metrics(part.rating, part.prediction)})
    split = clean[["asin", "parent_asin"]].copy()
    split["family"] = g
    split["split"] = ""
    for label, indices in [("train", train_i), ("calibration", cal_i), ("test", test_i)]:
        split.loc[indices, "split"] = label
    numeric_ranges = {}
    for key in [k for k in X if k in NUMERIC]:
        values = clean.iloc[train_i][key].dropna()
        numeric_ranges[key] = [float(values.min()), float(values.max())] if len(values) else None
    report = {
        "status": "SYNTHETIC SOFTWARE TEST ONLY" if fixture else "AMAZON OBSERVATIONS; HUMAN SOURCE AUDIT REQUIRED",
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "seed": seed,
        "winner": winner, "selection": "Minimum training-only grouped cross-validation MAE; test data never selects the model.",
        "test_metrics": metrics(y.iloc[test_i], pred), "baseline_test_metrics": metrics(y.iloc[test_i], baseline),
        "test_beats_mean_baseline": bool(mean_absolute_error(y.iloc[test_i], pred) < mean_absolute_error(y.iloc[test_i], baseline)),
        "interval": {"nominal_coverage": .9, "family_calibration_count": len(family_residuals), "radius": q,
                     "test_product_coverage": float(predictions.covered.mean()),
                     "test_family_coverage": float(predictions.groupby("product_family").covered.all().mean()),
                     "mean_width": float(np.mean(high - low))},
        "split_counts": split.split.value_counts().to_dict(),
        "family_counts": split.groupby("split").family.nunique().to_dict(),
        "quality": quality, "provenance": provenance,
        "currency": clean.currency.iloc[0], "features": list(X.columns),
        "dataset_sha256": hashlib.sha256(clean.to_csv(index=False).encode()).hexdigest(),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "limitations": [
            "One-time snapshots estimate current aggregate rating; they do not establish future performance or causality.",
            "Nominal interval coverage assumes exchangeable product families; marketplace drift and selection bias can invalidate it.",
            "Unidentified variants may leak related products across splits; audit parent ASINs and near-duplicate titles.",
            "Missing written-review counts are imputed, never treated as observed zero. Report their missingness.",
            "Cold-start is a feature-ablation experiment on rated products, not validated performance on genuinely new products.",
            "Evidence hashes establish file integrity, not source truth. Audit against Amazon pages.",
            "Test-based explanations and subgroup metrics are descriptive; further tuning needs a fresh test set."
        ]}
    artifact = {"pipeline": fitted, "report": report, "radius": q, "mode": mode,
                "numeric_ranges": numeric_ranges,
                "categories": {k: sorted(clean.iloc[train_i][k].unique().tolist()) for k in ["brand", "category"]}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "model.joblib")
    for name, table in [("model_comparison", ranking), ("cv_search", pd.DataFrame(cv_rows)),
                        ("test_predictions", predictions), ("feature_importance", importance_df),
                        ("category_metrics", pd.DataFrame(slices)), ("split_manifest", split)]:
        (out / f"{name}.csv").write_bytes(safe_csv(table))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "model_card.md").write_text(model_card(report), encoding="utf-8")
    return artifact, report

def predict(artifact, frame):
    if "currency" not in frame or not frame.currency.eq(artifact["report"]["currency"]).all():
        raise ValueError("Every prediction row must use the model's training currency.")
    if len(frame) == 0:
        raise ValueError("Supply at least one product.")
    X = features(frame, artifact["mode"])
    pred = artifact["pipeline"].predict(X)
    warnings = [[] for _ in range(len(frame))]
    for k, known in artifact["categories"].items():
        for i, val in enumerate(X[k]):
            if val not in known:
                warnings[i].append(f"Unseen {k}")
    for k, bounds in artifact["numeric_ranges"].items():
        if not bounds:
            continue
        vals = frame[k].map(number) if k in frame else pd.Series([np.nan] * len(frame))
        for i, val in enumerate(vals):
            if np.isnan(val):
                warnings[i].append(f"Missing {k}; imputed")
            elif val < bounds[0] or val > bounds[1]:
                warnings[i].append(f"{k} outside training range")
    q = artifact["radius"]
    return pd.DataFrame({"prediction": pred, "lower_90": np.maximum(1, pred - q),
                         "upper_90": np.minimum(5, pred + q),
                         "warnings": ["; ".join(x) for x in warnings]})

from .data import number

def model_card(r):
    return f'''# RateWise model card

Status: **{r['status']}**

Mode: {r['mode']}. Winner: {r['winner']}. Currency: {r['currency']}.

## Evaluation
Selection: {r['selection']}

Held-out MAE: {r['test_metrics']['MAE']:.4f}; RMSE: {r['test_metrics']['RMSE']:.4f}; R²: {r['test_metrics']['R2']}.
Mean-baseline MAE: {r['baseline_test_metrics']['MAE']:.4f}.
Beats mean baseline on held-out MAE: {r['test_beats_mean_baseline']}.

Nominal 90% interval; observed product coverage: {r['interval']['test_product_coverage']:.1%}.
Observed family coverage: {r['interval']['test_family_coverage']:.1%}.
Mean interval width: {r['interval']['mean_width']:.3f} rating points.

## Provenance
{r['provenance']}

Dataset SHA-256: {r['dataset_sha256']}

## Intended use
Academic analysis of existing rated Amazon products within the sampled marketplace and categories.
Predictions are estimates, not Amazon endorsements or guarantees of quality.

## Limitations
''' + "\n".join("- " + x for x in r["limitations"]) + "\n"
