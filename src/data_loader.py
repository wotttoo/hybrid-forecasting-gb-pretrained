import pandas as pd
import numpy as np

def read_and_melt_m4(file_path):
    """
    Đọc dữ liệu M4 từ CSV và chuyển từ Wide sang Long format.
    """
    df_wide = pd.read_csv(file_path)

    # Đổi tên cột đầu tiên thành 'unique_id'
    df_wide = df_wide.rename(columns={df_wide.columns[0]: 'unique_id'})

    # Chuyển sang Long format
    df_long = pd.melt(
        df_wide,
        id_vars=['unique_id'],
        var_name='time_step',
        value_name='y'
    ).dropna(subset=['y'])

    # Trích xuất số từ V1, V2... và sắp xếp
    df_long['time_step'] = df_long['time_step'].str.replace('V', '').astype(int)
    df_long = df_long.sort_values(by=['unique_id', 'time_step']).reset_index(drop=True)

    return df_long



def generate_monthly_dates(train_long, test_long, start_year=1900):
    """
    Tạo cột Datetime (ds) cho tập Train và Test bằng Vectorization (rất nhanh).
    Đảm bảo ngày chuẩn YYYY-MM-01.
    """
    # 1. Tính toán cho tập Train
    train_steps = train_long.groupby('unique_id').cumcount()
    train_years = start_year + (train_steps // 12)
    train_months = (train_steps % 12) + 1

    train_long['ds'] = pd.to_datetime({
        'year': train_years,
        'month': train_months,
        'day': 1
    })

    # 2. Tính toán cho tập Test
    # Lấy độ dài của mỗi chuỗi trong tập Train
    train_lengths = train_long.groupby('unique_id', sort=False).size().rename('train_len')
    test_long = test_long.merge(train_lengths, on='unique_id', how='left')

    # Bước của tập Test nối tiếp ngay sau tập Train
    test_steps = test_long['train_len'] + test_long.groupby('unique_id').cumcount()
    test_years = start_year + (test_steps // 12)
    test_months = (test_steps % 12) + 1

    test_long['ds'] = pd.to_datetime({
        'year': test_years,
        'month': test_months,
        'day': 1
    })

    # Chỉ giữ lại các cột cần thiết
    cols = ['unique_id', 'ds', 'y']
    return train_long[cols], test_long[cols]


def load_and_preprocess_m4_monthly(train_path, test_path):
    """
    Pipeline chính: Điều phối các hàm con để xử lý dữ liệu.
    """
    print("--- Bước 1 & 2: Đọc và Melt dữ liệu ---")
    train_long = read_and_melt_m4(train_path)
    test_long = read_and_melt_m4(test_path)

    print("--- Bước 3: Tái tạo mốc thời gian ---")
    train_processed, test_processed = generate_monthly_dates(train_long, test_long)

    print(f"Hoàn thành! Kích thước Train: {train_processed.shape}, Test: {test_processed.shape}")
    return train_processed, test_processed

# Gọi hàm
train_df, test_df = load_and_preprocess_m4_monthly(
    "data/raw/Train/Monthly-train.csv", 
    "data/raw/Test/Monthly-test.csv"
)
