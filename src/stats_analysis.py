"""
stats_analysis.py

Tính Ensemble Gain theo từng series và kiểm định thống kê so sánh
Gain giữa các nhóm length_group (cold_start/short/medium/long_history).
"""

import numpy as np
import pandas as pd
from scipy import stats
from evaluation import smape_per_series


def compute_gain(df: pd.DataFrame,
                  y_col: str = "y",
                  baseline_col: str = "pred_gb",
                  ensemble_col: str = "pred_ensemble_weighted",
                  group_col: str = "unique_id") -> pd.DataFrame:
    """
    Tính Ensemble Gain cho từng series:
        Gain = sMAPE(baseline) - sMAPE(ensemble)

    Gain > 0: ensemble tốt hơn baseline (GB) — điều ta kỳ vọng thấy
              rõ hơn ở nhóm cold_start theo giả thuyết nghiên cứu.
    Gain < 0: ensemble tệ hơn baseline.

    Returns
    -------
    DataFrame: unique_id, smape_baseline, smape_ensemble, gain
    """
    smape_baseline = smape_per_series(df, y_col, baseline_col, group_col)
    smape_ensemble = smape_per_series(df, y_col, ensemble_col, group_col)

    gain_df = pd.DataFrame({
        "smape_baseline": smape_baseline,
        "smape_ensemble": smape_ensemble,
    })
    gain_df["gain"] = gain_df["smape_baseline"] - gain_df["smape_ensemble"]
    gain_df = gain_df.reset_index().rename(columns={"index": group_col})

    return gain_df


def attach_metadata(gain_df: pd.DataFrame, metadata_df: pd.DataFrame,
                     group_col: str = "unique_id") -> pd.DataFrame:
    """Gắn length_group, category vào bảng Gain đã tính."""
    meta_cols = [group_col, "length_group", "category"]
    return gain_df.merge(
        metadata_df[meta_cols].drop_duplicates(subset=group_col),
        on=group_col, how="left"
    )


def kruskal_test(df: pd.DataFrame, group_col: str = "length_group",
                  value_col: str = "gain") -> dict:
    """
    Kruskal-Wallis test — so sánh phân phối Gain giữa NHIỀU nhóm
    length_group cùng lúc (không giả định phân phối chuẩn).

    H0: Gain có cùng phân phối ở mọi length_group.
    """
    groups = [g[value_col].dropna().values
              for _, g in df.groupby(group_col, observed=True)]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {"statistic": np.nan, "p_value": np.nan,
                "note": "Không đủ nhóm để kiểm định"}

    stat, p_value = stats.kruskal(*groups)
    return {"statistic": stat, "p_value": p_value}


def mannwhitney_test(df: pd.DataFrame, group_col: str, value_col: str,
                      group_a: str, group_b: str) -> dict:
    """
    Mann-Whitney U test — so sánh Gain giữa 2 nhóm CỤ THỂ.
    Dùng để kiểm định trực tiếp giả thuyết chính:
        cold_start vs long_history

    H0: Gain ở 2 nhóm có cùng phân phối.
    H1 (một phía, 'greater'): Gain ở group_a LỚN HƠN group_b — dùng
        khi giả thuyết có hướng rõ ràng (VD: cold_start có Gain cao
        hơn long_history).
    """
    values_a = df[df[group_col] == group_a][value_col].dropna()
    values_b = df[df[group_col] == group_b][value_col].dropna()

    if len(values_a) == 0 or len(values_b) == 0:
        return {"statistic": np.nan, "p_value": np.nan,
                "note": f"Thiếu dữ liệu ở {group_a} hoặc {group_b}"}

    stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative='greater')

    return {
        "statistic": stat,
        "p_value": p_value,
        "median_" + group_a: values_a.median(),
        "median_" + group_b: values_b.median(),
        "n_" + group_a: len(values_a),
        "n_" + group_b: len(values_b),
    }


def summarize_gain_by_group(df: pd.DataFrame, group_col: str = "length_group",
                             value_col: str = "gain") -> pd.DataFrame:
    """Bảng tóm tắt Gain (mean/median/std/count) theo từng length_group."""
    return df.groupby(group_col, observed=True)[value_col].agg(
        ['mean', 'median', 'std', 'count']
    ).round(3)