"""
length_grouping.py

Tính độ dài lịch sử từng series, merge với category, lọc theo ngưỡng,
chia nhóm theo quartile (cold_start/short/medium/long_history), và
thực hiện stratified sampling theo (length_group x category).

Input: DataFrame dạng long (unique_id, ds, y) — output từ data_loader.py
Output: sampled_metadata.csv (unique_id, length, category, length_group)
"""

import pandas as pd
import numpy as np


def compute_series_lengths(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Tính độ dài thực tế (số điểm dữ liệu) của từng series.

    Parameters
    ----------
    df_long : DataFrame dạng long, có cột 'unique_id' và 'y'
        (NaN padding đã được loại bỏ từ bước data_loader.py)

    Returns
    -------
    DataFrame với 2 cột: unique_id, length
    """
    lengths_df = (
        df_long.groupby("unique_id")
        .size()
        .reset_index(name="length")
    )
    return lengths_df


def merge_category(lengths_df: pd.DataFrame, info_df: pd.DataFrame,
                    frequency: str = "Monthly") -> pd.DataFrame:
    """
    Merge bảng độ dài với M4-info.csv để lấy category.

    Parameters
    ----------
    lengths_df : output từ compute_series_lengths()
    info_df : DataFrame đọc từ M4-info.csv (cột M4id, category, SP)
    frequency : lọc đúng nhóm tần suất đang xử lý (Monthly, Daily...)

    Returns
    -------
    DataFrame: unique_id, length, category
    """
    info_filtered = info_df[info_df["SP"] == frequency][["M4id", "category"]]

    merged = lengths_df.merge(
        info_filtered, left_on="unique_id", right_on="M4id", how="left"
    )
    merged = merged.drop(columns=["M4id"])

    n_missing = merged["category"].isna().sum()
    if n_missing > 0:
        print(f"⚠️ Cảnh báo: {n_missing} series không tìm thấy category sau merge")

    return merged


def filter_by_length(merged_df: pd.DataFrame,
                      min_length: int = 36,
                      max_length: int = 400) -> pd.DataFrame:
    """
    Loại series quá ngắn (không đủ để tạo lag/rolling ổn định) và
    quá dài (tránh lệch runtime tổng thể).

    min_length=36: tương đương 3 chu kỳ mùa vụ (12 tháng x 3) cho Monthly.
    max_length=400: ngưỡng tùy chỉnh theo tài nguyên tính toán của team.
    """
    before = len(merged_df)
    filtered = merged_df[
        (merged_df["length"] >= min_length) &
        (merged_df["length"] <= max_length)
    ].copy()
    after = len(filtered)

    print(f"Lọc theo length: {before} -> {after} series "
          f"(loại {before - after} series ngoài [{min_length}, {max_length}])")

    return filtered


def assign_length_group(filtered_df: pd.DataFrame,
                         n_groups: int = 4,
                         labels: list = None) -> pd.DataFrame:
    """
    Chia series thành các nhóm theo quartile độ dài lịch sử.

    Dùng qcut (dựa trên phân phối thực tế), không hardcode ngưỡng cố định.
    """
    if labels is None:
        labels = ["cold_start", "short", "medium", "long_history"][:n_groups]

    df = filtered_df.copy()
    df["length_group"] = pd.qcut(
        df["length"], q=n_groups, labels=labels
    )

    print("\nPhân bố length_group:")
    print(df["length_group"].value_counts().sort_index())

    print("\nRanh giới quartile:")
    print(df["length"].quantile(np.linspace(0, 1, n_groups + 1)).round(1))

    return df


def check_confound(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kiểm tra category có bị lệch theo length_group không (confound check).
    Trả về bảng crosstab để xem xét trước khi sampling.
    """
    crosstab = pd.crosstab(grouped_df["length_group"], grouped_df["category"])
    print("\nCrosstab length_group x category:")
    print(crosstab)
    return crosstab


def stratified_sample(grouped_df: pd.DataFrame,
                       sample_size: int = 500,
                       group_cols: list = None,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Lấy mẫu cuối cùng, đảm bảo mỗi tổ hợp (length_group x category)
    đều có đại diện tương ứng theo tỷ lệ trong dữ liệu đã lọc.
    """
    if group_cols is None:
        group_cols = ["length_group", "category"]

    combos = grouped_df[group_cols].drop_duplicates()
    n_groups = len(combos)
    if n_groups == 0:
        raise ValueError("Không có nhóm nào để sample.")

    per_group = max(1, sample_size // n_groups)

    sampled_parts = []
    for _, combo in combos.iterrows():
        mask = pd.Series(True, index=grouped_df.index)
        for col in group_cols:
            mask &= (grouped_df[col] == combo[col])
        subset = grouped_df[mask]
        n = min(len(subset), per_group)
        if n > 0:
            sampled_parts.append(subset.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    print(f"\nĐã sample: {len(sampled)} / mục tiêu {sample_size} series")
    print("Phân bố length_group sau sampling:")
    print(sampled["length_group"].value_counts().sort_index())

    return sampled


def filter_and_group_series(df_long: pd.DataFrame,
                             info_df: pd.DataFrame,
                             frequency: str = "Monthly",
                             min_length: int = 36,
                             max_length: int = 400,
                             n_groups: int = 4,
                             sample_size: int = 500,
                             random_state: int = 42) -> pd.DataFrame:
    """
    Hàm điều phối chính — chạy toàn bộ pipeline từ đầu đến cuối.

    Returns
    -------
    DataFrame: unique_id, length, category, length_group
    (đây là nội dung cần lưu ra sampled_metadata.csv)
    """
    print("--- Bước 1: Tính độ dài series ---")
    lengths_df = compute_series_lengths(df_long)

    print("\n--- Bước 2: Merge category ---")
    merged = merge_category(lengths_df, info_df, frequency=frequency)

    print("\n--- Bước 3: Lọc theo ngưỡng độ dài ---")
    filtered = filter_by_length(merged, min_length=min_length, max_length=max_length)

    print("\n--- Bước 4: Chia length_group theo quartile ---")
    grouped = assign_length_group(filtered, n_groups=n_groups)

    print("\n--- Bước 5: Kiểm tra confound category x length_group ---")
    check_confound(grouped)

    print("\n--- Bước 6: Stratified sampling ---")
    sampled = stratified_sample(
        grouped, sample_size=sample_size, random_state=random_state
    )

    return sampled


if __name__ == "__main__":
    # Ví dụ chạy độc lập để test
    import sys
    sys.path.append(".")
    from data_loader import load_and_preprocess_m4_monthly

    train_df, _ = load_and_preprocess_m4_monthly(
        "data/raw/Train/Monthly-train.csv",
        "data/raw/Test/Monthly-test.csv"
    )
    info_df = pd.read_csv("data/raw/M4-info.csv")

    sampled_metadata = filter_and_group_series(train_df, info_df)
    sampled_metadata.to_csv("data/processed/sampled_metadata.csv", index=False)
    print(f"\nĐã lưu sampled_metadata.csv với {len(sampled_metadata)} series")