"""
ensemble.py

Các phương pháp ensemble đơn giản để kết hợp dự báo từ GB baseline
và pretrained model — simple average và weighted average (tune trọng
số trên tập validation riêng, tránh leakage).
"""

import numpy as np
import pandas as pd
from evaluation import calculate_metrics


def simple_average(pred_gb: pd.Series, pred_pretrained: pd.Series) -> pd.Series:
    """Ensemble bằng trung bình cộng đơn giản giữa 2 nguồn dự báo."""
    return (pred_gb + pred_pretrained) / 2


def tune_weight(tune_set: pd.DataFrame,
                 y_col: str = "y",
                 gb_col: str = "pred_gb",
                 pretrained_col: str = "pred_pretrained",
                 weight_range=np.arange(0, 1.05, 0.05)) -> tuple:
    """
    Grid search trọng số w tối ưu cho weighted average, dựa trên RMSE
    trên tune_set (KHÔNG phải tập dùng để đánh giá cuối cùng).

    Returns
    -------
    (best_w, best_rmse)
    """
    best_w, best_rmse = None, np.inf
    for w in weight_range:
        pred = w * tune_set[gb_col] + (1 - w) * tune_set[pretrained_col]
        rmse = calculate_metrics(tune_set[y_col], pred)["RMSE"]
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return best_w, best_rmse


def weighted_average(pred_gb: pd.Series, pred_pretrained: pd.Series,
                      w: float) -> pd.Series:
    """Ensemble bằng trung bình có trọng số w (cho GB) đã tune trước đó."""
    return w * pred_gb + (1 - w) * pred_pretrained