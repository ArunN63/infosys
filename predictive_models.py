"""
predictive_models.py

Implements:
- data loading
- cleaning / EDA prints
- time-series feature engineering (lags, rolling)
- create targets (price horizon, promo horizon)
- time-aware splits (TimeSeriesSplit)
- LightGBM train functions for regression & classification
- sentiment analysis using Hugging Face transformers (local model)
- save/load models with joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
import lightgbm as lgb
from typing import List, Tuple, Dict, Any

# ---- Optional: HF sentiment ----
from transformers import pipeline

# ---- Initialize sentiment pipeline once (cached) ----
SENTIMENT_PIPELINE = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# ----------------------------
# 1) Data loading
# ----------------------------
def load_data(competitor_path: str, reviews_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_comp = pd.read_csv(competitor_path, parse_dates=["date"])
    df_rev = pd.read_csv(reviews_path, parse_dates=["date"])
    return df_comp, df_rev

# ----------------------------
# 2) Basic cleaning / EDA
# ----------------------------
def basic_eda(df_comp: pd.DataFrame, df_rev: pd.DataFrame):
    print("Competitor data shape:", df_comp.shape)
    print("Reviews data shape:", df_rev.shape)
    print("\nCompetitor head:\n", df_comp.head(3))
    print("\nReviews head:\n", df_rev.head(3))
    print("\nMissing values (competitor):\n", df_comp.isna().sum())
    print("\nMissing values (reviews):\n", df_rev.isna().sum())

# ----------------------------
# 3) Feature engineering
# ----------------------------
def make_time_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["competitor_id", "product_id"],
    value_col: str = "price",
    lags: List[int] = [1, 7, 14],
    roll_windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])

    # Lag features
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_cols)[value_col].shift(lag)

    # Rolling mean features
    for w in roll_windows:
        df[f"{value_col}_roll_mean_{w}"] = (
            df.groupby(group_cols)[value_col]
              .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )

    # Promo history (if exists)
    if "promo_flag" in df.columns:
        df["promo_last_7d"] = (
            df.groupby(group_cols)["promo_flag"]
              .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).max())
        )

    return df

# ----------------------------
# 4) Create targets
# ----------------------------
def create_targets(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = df.copy().sort_values(["competitor_id", "product_id", "date"])
    df["target_price"] = df.groupby(["competitor_id", "product_id"])["price"].shift(-horizon)
    if "promo_flag" in df.columns:
        df["target_promo"] = df.groupby(["competitor_id", "product_id"])["promo_flag"].shift(-horizon)
    return df

# ----------------------------
# 5) Time-aware split
# ----------------------------
def time_series_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X_idx = np.arange(len(df))
    return list(tscv.split(X_idx))

# ----------------------------
# 6a) Train LightGBM regressor
# ----------------------------
def train_lgb_reg(X_train, y_train, X_val, y_val, save_path: str = None):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"[REG] Validation RMSE: {rmse:.4f}")
    if save_path:
        joblib.dump(model, save_path)
        print(f"Saved regressor to {save_path}")
    return model

# ----------------------------
# 6b) Train LightGBM classifier
# ----------------------------
def train_lgb_clf(X_train, y_train, X_val, y_val, save_path: str = None):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(X_val)
    # Robust AUC calculation
    try:
        auc = roc_auc_score(y_val, preds)
    except ValueError:
        auc = float("nan")
    print(f"[CLF] Validation AUC: {auc:.4f}")
    if save_path:
        joblib.dump(model, save_path)
        print(f"Saved classifier to {save_path}")
    return model

# ----------------------------
# 7) Sentiment analysis
# ----------------------------
def sentiment_analysis_local(reviews: List[str]) -> List[Dict[str, Any]]:
    if not reviews:
        return []
    results = SENTIMENT_PIPELINE(reviews, truncation=True)
    out = []
    for r, res in zip(reviews, results):
        out.append({"review": r, "sentiment": res["label"].lower(), "score": float(res["score"])})
    return out

# ----------------------------
# 8) Orchestration
# ----------------------------
def prepare_and_train(
    competitor_csv: str,
    review_csv: str,
    horizon: int = 7,
    test_ratio: float = 0.2,
    save_dir: str = "models"
):
    os.makedirs(save_dir, exist_ok=True)

    df_comp, df_rev = load_data(competitor_csv, review_csv)
    basic_eda(df_comp, df_rev)

    required_cols = ["date", "competitor_id", "product_id", "price"]
    for c in required_cols:
        if c not in df_comp.columns:
            raise ValueError(f"competitor CSV missing required column: {c}")

    df_comp = df_comp.sort_values(["competitor_id", "product_id", "date"])
    df_comp["price"] = pd.to_numeric(df_comp["price"], errors="coerce")
    df_comp["promo_flag"] = df_comp.get("promo_flag", 0).fillna(0).astype(int)

    df_feats = make_time_features(df_comp, value_col="price")
    df_targets = create_targets(df_feats, horizon=horizon)
    df_targets = df_targets.dropna(subset=["target_price"])

    n_total = len(df_targets)
    n_train = int((1 - test_ratio) * n_total)
    train_df = df_targets.iloc[:n_train].copy()
    val_df = df_targets.iloc[n_train:].copy()

    feature_cols = [c for c in train_df.columns if ("lag" in c) or ("roll_mean" in c) or ("promo" in c)]
    if not feature_cols:
        feature_cols = ["price"]

    print("Feature columns used:", feature_cols)

    X_train = train_df[feature_cols].fillna(0)
    y_train_reg = train_df["target_price"]
    X_val = val_df[feature_cols].fillna(0)
    y_val_reg = val_df["target_price"]

    reg_model = train_lgb_reg(
        X_train, y_train_reg, X_val, y_val_reg,
        save_path=os.path.join(save_dir, "lgb_price_model.pkl")
    )

    if "target_promo" in df_targets.columns:
        df_targets_clf = df_targets.dropna(subset=["target_promo"])
        n_total = len(df_targets_clf)
        n_train = int((1 - test_ratio) * n_total)
        train_df_c = df_targets_clf.iloc[:n_train].copy()
        val_df_c = df_targets_clf.iloc[n_train:].copy()

        X_train_c = train_df_c[feature_cols].fillna(0)
        y_train_c = train_df_c["target_promo"].astype(int)
        X_val_c = val_df_c[feature_cols].fillna(0)
        y_val_c = val_df_c["target_promo"].astype(int)

        clf_model = train_lgb_clf(
            X_train_c, y_train_c, X_val_c, y_val_c,
            save_path=os.path.join(save_dir, "lgb_promo_model.pkl")
        )

    sample_reviews = df_rev["review_text"].dropna().astype(str).tolist()[:200]
    sentiments = sentiment_analysis_local(sample_reviews)
    print("Sample sentiment output (first 5):", sentiments[:5])

    return {"reg_model": reg_model, "sentiments_sample": sentiments}

# ----------------------------
# 9) Main
# ----------------------------
if __name__ == "__main__":
    competitor_csv = "C:/Users/arunn/OneDrive/Documents/infosys/Sample/data/competitor_history.csv"
    reviews_csv = "C:/Users/arunn/OneDrive/Documents/infosys/Sample/data/review.csv"

    if not os.path.exists(competitor_csv) or not os.path.exists(reviews_csv):
        print("Missing CSVs. Place competitor_history.csv and review.csv in same folder or update paths.")
    else:
        res = prepare_and_train(competitor_csv, reviews_csv, horizon=7)
        print("Done.")
