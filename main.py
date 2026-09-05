"""
main.py

Script chạy toàn bộ pipeline chính của project (không bao gồm phần
EDA thuần khám phá và visualize chi tiết — các phần đó vẫn nên xem
trực tiếp trong notebook để có biểu đồ/phân tích trực quan).

Thứ tự: load & sample data -> train GB (tuned) -> pretrained zero-shot
-> ensemble -> gain analysis -> lưu toàn bộ kết quả ra outputs/.

Chạy: python main.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_and_preprocess_m4_monthly
from length_grouping import filter_and_group_series
from features import create_time_features_optimized, make_train_test_split
from models_gb import (
    optimize_lightgbm_params, optimize_xgboost_params, train_with_best_params,
    predict_lightgbm, predict_xgboost
)
from models_pretrained import load_pretrained_model, build_series_dict, predict_batch
from ensemble import simple_average, tune_weight, weighted_average
from evaluation import calculate_metrics, smape_per_series
from stats_analysis import compute_gain, attach_metadata, kruskal_test, mannwhitney_test


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUTS_METRICS = "outputs/metrics"
OUTPUTS_MODELS = "outputs/models"

TEST_HORIZON = 18
MIN_LENGTH = 36
MAX_LENGTH = 400
SAMPLE_SIZE = 500
OPTUNA_TRIALS = 30  # giảm nếu cần chạy nhanh; 50 dùng khi có nhiều thời gian


def step_1_prepare_data():
    print("\n=== BƯỚC 1: Load & Sample dữ liệu ===")
    train_path = f"{RAW_DIR}/Train/Monthly-train.csv"
    test_path = f"{RAW_DIR}/Test/Monthly-test.csv"
    info_path = f"{RAW_DIR}/M4-info.csv"

    metadata_path = f"{PROCESSED_DIR}/sampled_metadata.csv"
    if os.path.exists(metadata_path):
        print(f"Đã có sẵn {metadata_path}, dùng lại.")
        sampled_metadata = pd.read_csv(metadata_path)
    else:
        train_full, _ = load_and_preprocess_m4_monthly(train_path, test_path)
        info_df = pd.read_csv(info_path)
        sampled_metadata = filter_and_group_series(
            train_full, info_df,
            min_length=MIN_LENGTH, max_length=MAX_LENGTH, sample_size=SAMPLE_SIZE
        )
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        sampled_metadata.to_csv(metadata_path, index=False)

    selected_ids = sampled_metadata["unique_id"].tolist()
    train_df, test_df = load_and_preprocess_m4_monthly(
        train_path, test_path, series_ids=selected_ids
    )
    train_df = train_df.merge(
        sampled_metadata[["unique_id", "category", "length_group"]],
        on="unique_id", how="left"
    )
    assert train_df["length_group"].isna().sum() == 0

    return train_df, test_df, sampled_metadata


def step_2_train_gb(train_df):
    print("\n=== BƯỚC 2: Train GB baseline (Optuna tuned) ===")
    train_df = create_time_features_optimized(train_df)
    train_split, val_split = make_train_test_split(train_df, test_horizon=TEST_HORIZON)

    feature_cols = [c for c in train_split.columns
                    if c not in ["unique_id", "ds", "y", "category", "length_group"]]

    X_train = train_split[feature_cols].ffill()
    y_train = train_split["y"]
    X_val = val_split[feature_cols].ffill()
    y_val = val_split["y"]

    study_lgb = optimize_lightgbm_params(X_train, y_train, X_val, y_val,
                                          n_trials=OPTUNA_TRIALS, timeout=1800)
    study_xgb = optimize_xgboost_params(X_train, y_train, X_val, y_val,
                                         n_trials=OPTUNA_TRIALS, timeout=1800)

    lgb_model = train_with_best_params("lightgbm", study_lgb, X_train, y_train, X_val, y_val)
    xgb_model = train_with_best_params("xgboost", study_xgb, X_train, y_train, X_val, y_val)

    lgb_pred = predict_lightgbm(lgb_model, X_val)
    xgb_pred = predict_xgboost(xgb_model, X_val)

    print("LightGBM metrics:", calculate_metrics(y_val, lgb_pred))
    print("XGBoost metrics:", calculate_metrics(y_val, xgb_pred))

    os.makedirs(OUTPUTS_MODELS, exist_ok=True)
    joblib.dump(lgb_model, f"{OUTPUTS_MODELS}/lgb_model.pkl")
    joblib.dump(xgb_model, f"{OUTPUTS_MODELS}/xgb_model.pkl")
    with open(f"{OUTPUTS_MODELS}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    results_df = val_split[["unique_id", "length_group", "category", "ds", "y"]].copy()
    results_df["pred_lgb"] = lgb_pred
    results_df["pred_xgb"] = xgb_pred

    os.makedirs(OUTPUTS_METRICS, exist_ok=True)
    results_df.to_csv(f"{OUTPUTS_METRICS}/gb_baseline_predictions.csv", index=False)

    return train_split, val_split, results_df


def step_3_pretrained(train_split, val_split, sampled_metadata):
    print("\n=== BƯỚC 3: Pretrained model (Chronos zero-shot) ===")
    series_dict = build_series_dict(train_split)
    pipeline = load_pretrained_model("amazon/chronos-t5-small", device="cpu")

    predictions_long = predict_batch(pipeline, series_dict, horizon=TEST_HORIZON,
                                      num_samples=20, show_progress=True)

    val_split_sorted = val_split.sort_values(["unique_id", "ds"]).copy()
    val_split_sorted["step"] = val_split_sorted.groupby("unique_id").cumcount() + 1

    predictions_merged = predictions_long.merge(
        val_split_sorted[["unique_id", "step", "ds", "y"]],
        on=["unique_id", "step"], how="left"
    )
    predictions_merged = predictions_merged.merge(
        sampled_metadata[["unique_id", "category", "length_group"]],
        on="unique_id", how="left"
    )

    results_df = predictions_merged[
        ["unique_id", "length_group", "category", "ds", "y", "pred_pretrained"]
    ].copy()
    results_df.to_csv(f"{OUTPUTS_METRICS}/pretrained_predictions.csv", index=False)

    return results_df


def step_4_ensemble(gb_pred, pretrained_pred):
    print("\n=== BƯỚC 4: Ensemble ===")
    merged = gb_pred.merge(
        pretrained_pred[["unique_id", "ds", "pred_pretrained"]],
        on=["unique_id", "ds"], how="inner"
    )
    assert len(merged) > 0, "Merge rỗng — kiểm tra lại khoảng thời gian giữa 2 nguồn dự báo"

    metrics_lgb = calculate_metrics(merged["y"], merged["pred_lgb"])
    metrics_xgb = calculate_metrics(merged["y"], merged["pred_xgb"])
    best_gb_col = "pred_lgb" if metrics_lgb["RMSE"] < metrics_xgb["RMSE"] else "pred_xgb"
    merged["pred_gb"] = merged[best_gb_col]

    merged["pred_ensemble_avg"] = simple_average(merged["pred_gb"], merged["pred_pretrained"])

    unique_ids = merged["unique_id"].unique()
    tune_ids, _ = train_test_split(unique_ids, test_size=0.3, random_state=42)
    tune_set = merged[merged["unique_id"].isin(tune_ids)]
    best_w, best_rmse = tune_weight(tune_set)
    print(f"Trọng số tối ưu: w={best_w:.2f} (RMSE tune set: {best_rmse:.2f})")

    merged["pred_ensemble_weighted"] = weighted_average(
        merged["pred_gb"], merged["pred_pretrained"], best_w
    )

    final_results = merged[[
        "unique_id", "length_group", "category", "ds", "y",
        "pred_gb", "pred_pretrained", "pred_ensemble_avg", "pred_ensemble_weighted"
    ]].copy()
    final_results.to_csv(f"{OUTPUTS_METRICS}/ensemble_predictions.csv", index=False)

    return final_results


def step_5_gain_analysis(ensemble_df, sampled_metadata):
    print("\n=== BƯỚC 5: Gain Analysis ===")
    gain_df = compute_gain(
        ensemble_df, y_col="y", baseline_col="pred_gb",
        ensemble_col="pred_ensemble_weighted"
    )
    gain_df = attach_metadata(gain_df, sampled_metadata)
    gain_df.to_csv(f"{OUTPUTS_METRICS}/gain_analysis_results.csv", index=False)

    print("\nThống kê Gain theo length_group:")
    print(gain_df.groupby("length_group", observed=True)["gain"]
          .agg(["mean", "median", "std", "count"]).round(3))

    kw_result = kruskal_test(gain_df)
    print("\nKruskal-Wallis test:", kw_result)

    mw_result = mannwhitney_test(
        gain_df, group_col="length_group", value_col="gain",
        group_a="cold_start", group_b="long_history"
    )
    print("Mann-Whitney (cold_start vs long_history):", mw_result)

    return gain_df


def main():
    train_df, test_df, sampled_metadata = step_1_prepare_data()
    train_split, val_split, gb_pred = step_2_train_gb(train_df)
    pretrained_pred = step_3_pretrained(train_split, val_split, sampled_metadata)
    ensemble_df = step_4_ensemble(gb_pred, pretrained_pred)
    gain_df = step_5_gain_analysis(ensemble_df, sampled_metadata)

    print("\n=== HOÀN THÀNH TOÀN BỘ PIPELINE ===")
    print(f"Kết quả đã lưu tại: {OUTPUTS_METRICS}/")


if __name__ == "__main__":
    main()