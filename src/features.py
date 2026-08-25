import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Tạo các đặc trưng Lag, Rolling Statistics và Date Features từ tập dữ liệu sạch.
    Đầu vào: DataFrame chứa các cột ['unique_id', 'ds', 'y']
    """
    # Đảm bảo dữ liệu sắp xếp đúng thứ tự thời gian của từng chuỗi
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    # --- 1. Date/Calendar Features (Đặc trưng lịch) ---
    # Vì là chuỗi Monthly (hàng tháng), các đặc trưng liên quan đến năm, tháng, quý là phù hợp nhất
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    
    # --- 2. Lag Features (Đặc trưng trễ) ---
    # Cho dữ liệu chu kỳ tháng, các lag phổ biến là 1, 2, 3 (ngắn hạn) và 6, 12 (chu kỳ mùa vụ năm)
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('unique_id')['y'].shift(lag)
        
    # --- 3. Rolling Statistics (Thống kê trượt) ---
    # ĐỂ TRÁNH RÒ RỈ DỮ LIỆU (DATA LEAKAGE):
    # Ta phải shift(1) trước khi tính rolling. Điều này đảm bảo giá trị thống kê trượt 
    # tại thời điểm t không bao giờ chứa thông tin của chính mục tiêu y_t mà mô hình cần dự báo.
    windows = [3, 6, 12]
    for w in windows:
        # Tính trên giá trị đã lag 1 (tức là thông tin quá khứ gần nhất có sẵn tại thời điểm t)
        df[f'rolling_mean_{w}'] = df.groupby('unique_id')['y'].shift(1).rolling(window=w).mean()
        df[f'rolling_std_{w}'] = df.groupby('unique_id')['y'].shift(1).rolling(window=w).std()
        
    return df

def make_train_test_split(df, test_horizon=18):
    """
    Chia tách dữ liệu thành tập Train và Test theo thời gian (Time-series Split).
    Lấy test_horizon dòng cuối cùng của mỗi unique_id làm tập Test, phần trước đó làm Train.
    """
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    train_list = []
    test_list = []
    
    for uid, group in df.groupby('unique_id'):
        # Tập test: n dòng cuối cùng của mỗi group
        test_list.append(group.tail(test_horizon))
        # Tập train: tất cả ngoại trừ n dòng cuối
        train_list.append(group.iloc[:-test_horizon])
        
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    return train_df, test_df

def get_rolling_origin_splits(df, test_horizon=18, n_splits=3, step_size=6):
    """
    Tạo các tập cắt Rolling-Origin Cross Validation (Time-series Split tịnh tiến) cho huấn luyện.
    Trả về danh sách các cặp (train_df, val_df) cho mỗi split.
    
    Parameters:
    - df: DataFrame dữ liệu đã được tạo features
    - test_horizon: độ dài tập validation (ví dụ: M4 Monthly là 18)
    - n_splits: số lượng tập cắt CV muốn thử nghiệm (mặc định: 3)
    - step_size: số bước dịch chuyển tịnh tiến lùi về quá khứ cho mỗi split
    """
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    splits = []
    
    for i in range(n_splits):
        val_end_offset = i * step_size
        val_start_offset = val_end_offset + test_horizon
        
        val_list = []
        train_list = []
        
        for uid, group in df.groupby('unique_id'):
            total_len = len(group)
            if total_len <= val_start_offset:
                # Nếu chuỗi quá ngắn không đủ chia, bỏ qua để tránh lỗi index
                continue
                
            # Cắt tập validation
            if val_end_offset == 0:
                val_group = group.iloc[-val_start_offset:]
            else:
                val_group = group.iloc[-val_start_offset : -val_end_offset]
                
            # Tập train là tất cả phần trước tập validation
            train_group = group.iloc[:-val_start_offset]
            
            val_list.append(val_group)
            train_list.append(train_group)
            
        if len(val_list) > 0 and len(train_list) > 0:
            val_df = pd.concat(val_list).reset_index(drop=True)
            train_df = pd.concat(train_list).reset_index(drop=True)
            splits.append((train_df, val_df))
            
    return splits
