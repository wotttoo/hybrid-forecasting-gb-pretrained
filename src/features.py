import pandas as pd
import numpy as np

def create_time_features_optimized(df):
    """
    Tạo các đặc trưng cho dữ liệu lớn. Tối ưu hóa Groupby và bộ nhớ.
    """
    # 1. Sắp xếp (In-place sort nếu có thể để tiết kiệm RAM, nhưng gán lại như bạn cũng ổn)
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # --- 1. Date/Calendar Features ---
    # Downcast các biến thời gian về int8 hoặc int16 để tiết kiệm RAM
    df['year'] = df['ds'].dt.year.astype(np.int16)
    df['month'] = df['ds'].dt.month.astype(np.int8)
    df['quarter'] = df['ds'].dt.quarter.astype(np.int8)
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(np.int8)

    # --- TỐI ƯU HÓA: Tạo GroupBy object MỘT LẦN ---
    # Thao tác này giúp Pandas không phải chia nhóm lại từ đầu
    grouped_y = df.groupby('unique_id')['y']

    # --- 2. Lag Features ---
    lags = [1, 2, 3, 6, 12]
    for lag in lags:
        # Cast về float32 để giảm một nửa dung lượng RAM so với float64
        df[f'lag_{lag}'] = grouped_y.shift(lag).astype(np.float32)

    # --- 3. Rolling Statistics ---
    windows = [3, 6, 12]
    
    # TỐI ƯU HÓA: Tái sử dụng cột 'lag_1' thay vì shift(1) lại từ đầu
    grouped_lag1 = df.groupby('unique_id')['lag_1']
    
    for w in windows:
        # Gọi .rolling() trên groupby object sẽ sinh ra MultiIndex (unique_id, index_cũ).
        # Ta dùng .reset_index(level=0, drop=True) để khớp lại đúng với index của df ban đầu
        rolling_obj = grouped_lag1.rolling(window=w)
        
        df[f'rolling_mean_{w}'] = rolling_obj.mean().reset_index(level=0, drop=True).astype(np.float32)
        df[f'rolling_std_{w}']  = rolling_obj.std().reset_index(level=0, drop=True).astype(np.float32)

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
