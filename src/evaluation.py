"""
evaluation.py

Các hàm tính chỉ số đánh giá mô hình forecasting: MAE, RMSE, MAPE,
sMAPE, MASE — cả dạng tổng hợp và dạng theo từng series (cần cho
05_gain_analysis.ipynb).
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred, y_train=None, seasonality=12):
    """
    Tính các chỉ số đánh giá tổng hợp trên toàn bộ tập dữ liệu.

    Parameters
    ----------
    y_true, y_pred : array-like, cùng độ dài
    y_train : array-like, optional
        Lịch sử train (để tính MASE). Nếu None, bỏ qua MASE.
    seasonality : int
        Chu kỳ mùa vụ dùng cho naive seasonal baseline của MASE
        (Monthly = 12).

    Returns
    -------
    dict: MAE, RMSE, MAPE, sMAPE, (MASE nếu có y_train)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: tránh chia cho 0 bằng cách bỏ qua các điểm y_true = 0
    mask = y_true != 0
    mape = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.any() else np.nan)

    smape_val = smape(y_true, y_pred)

    result = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape_val}

    if y_train is not None:
        result['MASE'] = mase(y_true, y_pred, y_train, seasonality=seasonality)

    return result


def smape(y_true, y_pred):
    """
    Symmetric MAPE — metric chính thức của M4 Competition.
    Không nhạy với scale giữa các series khác nhau, ổn định hơn MAPE
    (mẫu số là (|y_true| + |y_pred|)/2, tránh chia cho 0 tuyệt đối).

    Trả về giá trị theo thang % (0-200).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)

    # Nếu cả y_true và y_pred đều = 0, coi sai số = 0 (tránh chia 0/0)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator == 0, 0, diff / denominator)

    return np.mean(result) * 100


def mase(y_true, y_pred, y_train, seasonality=12):
    """
    Mean Absolute Scaled Error — metric chính thức thứ 2 của M4.
    So sánh sai số của model với sai số của naive seasonal forecast
    (dự báo bằng giá trị cùng kỳ trước đó `seasonality` bước).

    Parameters
    ----------
    y_train : array-like
        Lịch sử TRAIN của series (dùng để tính mẫu số — sai số của
        naive seasonal baseline trong quá khứ).
    seasonality : int
        Monthly = 12, Quarterly = 4, Yearly = 1...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonality:
        return np.nan  # không đủ dữ liệu để tính naive seasonal error

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / scale


def smape_per_series(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                      group_col: str = "unique_id") -> pd.Series:
    """
    Tính sMAPE RIÊNG cho từng series — cần cho việc tính Ensemble Gain
    theo từng series ở 05_gain_analysis.ipynb.

    Returns
    -------
    pd.Series, index = unique_id, value = sMAPE của series đó
    """
    def _smape_group(g):
        return smape(g[y_true_col], g[y_pred_col])

    return df.groupby(group_col).apply(_smape_group, include_groups=False)


def mase_per_series(df: pd.DataFrame, train_dict: dict,
                     y_true_col: str, y_pred_col: str,
                     group_col: str = "unique_id",
                     seasonality: int = 12) -> pd.Series:
    """
    Tính MASE riêng cho từng series.

    Parameters
    ----------
    train_dict : dict {unique_id: mảng lịch sử train của series đó}
        (lấy từ build_series_dict() trong models_pretrained.py, hoặc
        tự tạo tương tự từ train_split)
    """
    results = {}
    for uid, g in df.groupby(group_col):
        y_train = train_dict.get(uid)
        if y_train is None:
            results[uid] = np.nan
            continue
        results[uid] = mase(g[y_true_col], g[y_pred_col], y_train, seasonality)

    return pd.Series(results)


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Vẽ scatter plot Actual vs Predicted, kèm đường y=x tham chiếu."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.tight_layout()
    plt.show()