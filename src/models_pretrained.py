"""
models_pretrained.py

Wrapper chạy inference zero-shot cho pretrained time-series model
(Chronos - amazon/chronos-t5-small), song song với models_gb.py để
dễ dùng thống nhất trong pipeline.

Zero-shot nghĩa là: KHÔNG train lại model trên dữ liệu M4, chỉ dùng
model đã được Amazon pretrain sẵn trên hàng triệu series khác, đưa
thẳng lịch sử của series cần dự báo vào để lấy kết quả.
"""

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from tqdm import tqdm


def load_pretrained_model(model_name: str = "amazon/chronos-t5-small",
                           device: str = "cpu"):
    """
    Load model Chronos đã pretrain sẵn từ HuggingFace Hub.

    Parameters
    ----------
    model_name : tên model trên HuggingFace.
        - "amazon/chronos-t5-tiny"  : nhanh nhất, độ chính xác thấp hơn
        - "amazon/chronos-t5-small" : cân bằng tốc độ/độ chính xác (khuyến nghị)
        - "amazon/chronos-t5-base"  : chính xác hơn nhưng chậm hơn nhiều
    device : "cpu" hoặc "cuda" (nếu có GPU)

    Returns
    -------
    ChronosPipeline object, dùng để gọi .predict() sau này
    """
    print(f"Đang tải model {model_name} lên {device}...")

    # torch_dtype=torch.float32 an toàn cho CPU (bfloat16 có thể
    # không tương thích tốt trên 1 số CPU đời cũ)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    print("Tải model thành công.")
    return pipeline


def predict_series(pipeline, history_values: np.ndarray,
                    horizon: int = 18, num_samples: int = 20) -> np.ndarray:
    """
    Dự báo h bước tiếp theo cho MỘT series, dùng zero-shot inference.

    Parameters
    ----------
    pipeline : ChronosPipeline đã load từ load_pretrained_model()
    history_values : mảng 1 chiều, giá trị lịch sử thật của series
        (đã dropna, không còn NaN padding)
    horizon : số bước cần dự báo (M4 Monthly = 18)
    num_samples : số lượng sample Chronos sinh ra cho mỗi bước dự báo
        (Chronos là probabilistic model — sinh nhiều kịch bản dự báo
        khác nhau, sau đó ta lấy median làm điểm dự báo cuối cùng)

    Returns
    -------
    Mảng 1 chiều, độ dài = horizon — dự báo median cho h bước tới
    """
    context = torch.tensor(history_values, dtype=torch.float32)

    # pipeline.predict() trả về tensor shape: [1, num_samples, horizon]
    # (chiều đầu là batch=1 vì ta chỉ đưa vào 1 series)
    forecast = pipeline.predict(
        context=context,
        prediction_length=horizon,
        num_samples=num_samples,
    )

    # Lấy median (quantile 0.5) trên chiều num_samples
    # -> ra 1 điểm dự báo duy nhất cho mỗi bước, thay vì cả phân phối
    forecast_np = forecast[0].numpy()          # shape: [num_samples, horizon]
    median_forecast = np.median(forecast_np, axis=0)   # shape: [horizon]

    return median_forecast


def predict_batch(pipeline,
                   series_dict: dict,
                   horizon: int = 18,
                   num_samples: int = 20,
                   show_progress: bool = True) -> pd.DataFrame:
    """
    Chạy predict_series() cho NHIỀU series liên tiếp, có progress bar.

    Parameters
    ----------
    series_dict : dict dạng {unique_id: mảng giá trị lịch sử}
    horizon, num_samples : xem predict_series()
    show_progress : hiện thanh tiến trình (tqdm) — nên bật vì bước
        này khá tốn thời gian với nhiều series

    Returns
    -------
    DataFrame dạng long: unique_id, step (1..horizon), pred_pretrained

    Lưu ý: đây là vòng lặp tuần tự (predict từng series một), KHÔNG
    phải batch inference thật sự của Chronos. Cách này đơn giản, dễ
    hiểu, đủ dùng cho quy mô ~500 series của baseline. Nếu sau này
    cần chạy nhanh hơn trên nhiều series, có thể tìm hiểu batching
    nhiều context cùng lúc trong 1 lần gọi pipeline.predict().
    """
    results = []
    iterator = series_dict.items()
    if show_progress:
        iterator = tqdm(iterator, total=len(series_dict), desc="Chronos zero-shot")

    for uid, history in iterator:
        try:
            pred = predict_series(pipeline, history, horizon=horizon,
                                   num_samples=num_samples)
            for step, value in enumerate(pred, start=1):
                results.append({
                    "unique_id": uid,
                    "step": step,
                    "pred_pretrained": value
                })
        except Exception as e:
            # Không để 1 series lỗi làm chết cả vòng lặp — log lại
            # để kiểm tra sau, các series khác vẫn tiếp tục chạy
            print(f"Lỗi khi dự báo series {uid}: {e}")
            continue

    return pd.DataFrame(results)


def build_series_dict(train_df: pd.DataFrame) -> dict:
    """
    Chuyển DataFrame dạng long (unique_id, ds, y) thành dict
    {unique_id: mảng giá trị y theo đúng thứ tự thời gian} — định
    dạng input mà predict_batch() cần.

    Parameters
    ----------
    train_df : DataFrame long-format, đã sort theo (unique_id, ds)
    """
    series_dict = {}
    for uid, group in train_df.groupby("unique_id"):
        # Đảm bảo đúng thứ tự thời gian trước khi lấy giá trị
        sorted_group = group.sort_values("ds")
        series_dict[uid] = sorted_group["y"].values

    return series_dict