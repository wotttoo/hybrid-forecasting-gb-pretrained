
# Hybrid Forecasting: Gradient Boosting + Pretrained Time-Series Model

> Nghiên cứu thực nghiệm về việc kết hợp (ensemble) dự báo từ mô hình Gradient Boosting (XGBoost/LightGBM) với một mô hình time-series pretrained (Chronos), nhằm trả lời câu hỏi: **lợi ích của việc kết hợp mô hình có phụ thuộc vào độ dài lịch sử dữ liệu hay không?**

---

## 📌 Tóm tắt đề tài

|                                   |                                                                             |
| --------------------------------- | --------------------------------------------------------------------------- |
| **Hướng nghiên cứu**    | So sánh phương pháp / kiểm định giả thuyết                         |
| **Dataset**                 | M4 Competition — nhóm Monthly                                             |
| **Baseline models**         | LightGBM, XGBoost (tuned bằng Optuna)                                      |
| **Pretrained model**        | Chronos (`amazon/chronos-t5-small`), zero-shot                            |
| **Phương pháp Ensemble** | Simple Average, Weighted Average                                            |
| **Công cụ phân tích**   | SHAP, kiểm định thống kê phi tham số (Kruskal-Wallis, Mann-Whitney U) |

---

## 🎯 Câu hỏi nghiên cứu

> **Lợi ích của việc kết hợp GB + pretrained time-series model có phụ thuộc vào độ dài lịch sử dữ liệu (cold-start vs. long-history) hay không?**

**Giả thuyết ban đầu (H1):** Ensemble sẽ có lợi rõ rệt hơn ở các series **cold-start** (ít dữ liệu lịch sử), vì pretrained model — vốn được huấn luyện sẵn trên hàng triệu chuỗi khác — có thể bù đắp cho việc Gradient Boosting không đủ dữ liệu để học lag/seasonal features một cách ổn định.

### 🔍 Kết quả chính (spoiler)

Giả thuyết ban đầu **không được ủng hộ**. Kết quả thực nghiệm cho thấy điều ngược lại: nhóm **long_history** mới là nơi Ensemble mang lại lợi ích rõ rệt nhất, trong khi ở **cold_start**, Ensemble Gain trung bình lại âm nhẹ. Phân tích sâu hơn (so sánh sMAPE riêng của từng model, kết hợp SHAP) cho thấy nguyên nhân: **GB (huấn luyện pooled trên nhiều series) vượt trội hơn Chronos ở hầu hết mọi nhóm độ dài**, khiến việc ensemble với một model yếu hơn không mang lại cải thiện ở cold-start. Chi tiết đầy đủ xem tại [Kết quả &amp; Kết luận](#-kết-quả--kết-luận).

---

## 🗂️ Cấu trúc project

```
hybrid-forecasting-gb-pretrained/
│
├── data/
│   ├── raw/                    # Dữ liệu gốc M4 (Train/Test theo từng tần suất + M4-info.csv)
│   └── processed/
│       └── sampled_metadata.csv   # Danh sách series đã qua stratified sampling
│
├── notebooks/
│   ├── 01_eda.ipynb             # Khám phá dữ liệu, tạo sampled_metadata.csv
│   ├── 02_baseline_gb.ipynb     # Train LightGBM/XGBoost (tuned bằng Optuna)
│   ├── 03_pretrained.ipynb      # Chronos zero-shot inference
│   ├── 04_ensemble.ipynb        # Simple average & Weighted average
│   ├── 05_gain_analysis.ipynb   # Phân tích giả thuyết chính + kiểm định thống kê
│   └── 06_shap_case_study.ipynb # SHAP explainability + case study định tính
│
├── src/
│   ├── data_loader.py           # Đọc & parse dữ liệu M4 (wide → long format)
│   ├── length_grouping.py       # Lọc, chia length_group, stratified sampling
│   ├── features.py              # Lag/rolling/date features, train-test split
│   ├── models_gb.py             # Train + tune (Optuna) LightGBM/XGBoost
│   ├── models_pretrained.py     # Wrapper inference Chronos zero-shot
│   ├── ensemble.py              # Simple average / weighted average
│   ├── evaluation.py            # MAE, RMSE, MAPE, sMAPE, MASE
│   └── stats_analysis.py        # Tính Gain + kiểm định thống kê
│
├── outputs/
│   ├── models/                  # Model đã train (.pkl) + feature_cols.json
│   ├── metrics/                 # Toàn bộ predictions & kết quả (.csv)
│   └── figures/                 # Biểu đồ xuất ra từ notebook
│
├── main.py                      # Script chạy toàn bộ pipeline từ đầu đến cuối
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Sử dụng bộ dữ liệu **[M4 Competition](https://github.com/Mcompetitions/M4-methods)** — bộ benchmark forecasting phổ biến với 100,000 series thuộc 6 nhóm tần suất. Đề tài tập trung vào nhóm **Monthly** (horizon = 18 bước).

### Cấu trúc dữ liệu gốc

- `Monthly-train.csv`: mỗi hàng là 1 series, cột `V1` là ID, các cột `V2, V3...` là giá trị lịch sử (có NaN padding do độ dài series khác nhau).
- `Monthly-test.csv`: 18 giá trị thật tương ứng, dùng làm ground truth.
- `M4-info.csv`: metadata — category (Micro/Macro/Finance/Industry/...), tần suất.

### Quy trình chọn mẫu (Stratified Sampling)

1. Tính độ dài thực tế từng series (bỏ NaN padding).
2. Merge với category từ `M4-info.csv`.
3. Lọc series quá ngắn (`< 36` điểm — dưới 3 chu kỳ mùa vụ) và quá dài (`> 400` điểm).
4. Chia thành 4 nhóm theo **quartile độ dài lịch sử**: `cold_start`, `short`, `medium`, `long_history`.
5. Lấy mẫu ~500 series theo cách **stratified** trên `(length_group × category)`, đảm bảo mỗi nhóm đều có đại diện cân đối.

---

## 🧪 Thiết kế thí nghiệm

```
                 ┌──────────────────────┐
                 │   M4 Monthly (raw)   │
                 └──────────┬───────────┘
                            ▼
              Lọc + chia length_group + sample
                            ▼
        ┌───────────────────┴───────────────────┐
        ▼                                        ▼
┌───────────────────┐                  ┌──────────────────────┐
│   GB Baseline      │                  │  Pretrained (Chronos) │
│ (LightGBM/XGBoost)  │                  │     Zero-shot         │
│  + Optuna tuning    │                  └──────────┬───────────┘
└─────────┬──────────┘                              │
          └───────────────────┬──────────────────────┘
                               ▼
                    Ensemble (Avg / Weighted)
                               ▼
                  Đánh giá theo length_group
                    (sMAPE, Ensemble Gain)
                               ▼
              Kiểm định thống kê (Kruskal-Wallis,
                    Mann-Whitney U test)
                               ▼
                SHAP + Case Study (giải thích cơ chế)
```

### Baseline: Gradient Boosting

- **Feature engineering**: lag (1, 2, 3, 6, 12), rolling mean/std (3, 6, 12 — có `shift(1)` để tránh leakage), date features giả lập theo chu kỳ (month, quarter, year).
- **Train/Val split**: theo thời gian (18 tháng cuối làm validation), không random split.
- **Hyperparameter tuning**: Optuna (TPE sampler + Median pruner), tối ưu RMSE trên validation set.
- Chọn model tốt hơn giữa LightGBM/XGBoost làm đại diện GB cho bước ensemble.

### Pretrained: Chronos (zero-shot)

- Model: `amazon/chronos-t5-small` — chạy CPU-friendly, không cần fine-tune.
- Dự báo probabilistic (nhiều sample), lấy **median** làm điểm dự báo.
- Dự báo trên **cùng khung thời gian** (18 tháng cuối) với GB validation set, đảm bảo so sánh công bằng.

### Ensemble

- **Simple Average**: `(pred_GB + pred_Pretrained) / 2`
- **Weighted Average**: grid search trọng số `w` trên tập *tune* riêng (70% series), đánh giá cuối trên tập *eval* độc lập (30% series) để tránh leakage.

### Đánh giá

- **Metric chính**: sMAPE (Symmetric MAPE) — chuẩn của M4 Competition, không nhạy scale giữa các series.
- **Ensemble Gain** (biến trung tâm của nghiên cứu):
  ```
  Gain = sMAPE(GB) - sMAPE(Ensemble)
  ```

  Gain > 0 nghĩa là ensemble cải thiện so với GB đơn lẻ.
- **Kiểm định thống kê**: Kruskal-Wallis test (so sánh 4 nhóm), Mann-Whitney U test (so sánh trực tiếp cold_start vs long_history).

---

## 📈 Kết quả & Kết luận

### Ensemble Gain theo length_group

| length_group           |        Mean Gain | Median Gain | % series Ensemble thắng |
| ---------------------- | ---------------: | ----------: | -----------------------: |
| cold_start             |          −0.058 |      +0.287 |                    60.6% |
| short                  |           +0.258 |          — |                    62.5% |
| medium                 |           +0.135 |          — |                    63.3% |
| **long_history** | **+0.539** |      +0.268 |          **79.0%** |

- **Kruskal-Wallis test** (4 nhóm): p = 0.261 → không có ý nghĩa thống kê
- **Mann-Whitney U test** (cold_start vs long_history, one-sided): p = 0.825 → không đủ bằng chứng thống kê

### Nguyên nhân — so sánh sMAPE riêng biệt GB vs Pretrained

| length_group | sMAPE (GB) | sMAPE (Pretrained) |
| ------------ | ---------: | -----------------: |
| cold_start   |     ~16.2% |             ~22.4% |
| short        |     ~10.8% |             ~13.5% |
| medium       |      ~5.5% |              ~7.7% |
| long_history |      ~6.2% |              ~5.5% |

**GB vượt trội hơn Pretrained ở mọi nhóm, ngoại trừ long_history** — nơi 2 model gần tương đương. Đây là nguyên nhân gốc rễ khiến ensemble không phát huy tác dụng ở cold_start: ensemble chỉ có lợi khi 2 model đủ mạnh và bổ sung cho nhau; trộn thêm một model yếu hơn (Chronos ở cold_start) vào GB không cải thiện kết quả.

### SHAP — xác nhận cơ chế

Phân tích SHAP cho thấy GB dựa vào **gần như cùng một bộ feature quan trọng nhất** (`lag_1`, `rolling_mean_3`, `lag_2`, `lag_3`, `lag_12`...) ở cả nhóm cold_start lẫn long_history, với thứ hạng gần như không đổi. Điều này cho thấy nhờ cơ chế **pooled training** (nhiều series cùng học chung 1 model), GB không cần mỗi series tự có đủ lịch sử riêng để học ra chiến lược dự báo hiệu quả.

### Kết luận

> Lợi ích của ensemble **không phụ thuộc trực tiếp vào độ dài lịch sử dữ liệu** như giả thuyết ban đầu, mà phụ thuộc vào **sức mạnh tương đối giữa hai model** tại từng nhóm/series cụ thể. Cơ chế "pretrained bù đắp cho GB thiếu dữ liệu" chỉ đúng khi pretrained model thực sự đủ mạnh — điều không xảy ra với Chronos-t5-small (bản nhỏ nhất) trên dataset M4 Monthly trong thiết lập thí nghiệm này.

### Giới hạn của nghiên cứu

- Chỉ thực hiện trên 1 tần suất (Monthly), chưa kiểm chứng trên Daily/Weekly/Yearly.
- Chỉ dùng 1 pretrained model (Chronos bản "small") — chưa thử các model/bản lớn hơn (TimesFM, Lag-Llama, hay Chronos bản base/large).
- Cỡ mẫu mỗi length_group (~100–150 series) có thể chưa đủ lớn để phát hiện khác biệt nhỏ có ý nghĩa thống kê.
- Trọng số weighted ensemble được tune chung cho toàn bộ tập, chưa tối ưu riêng theo từng length_group.

---

## ⚙️ Cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd hybrid-forecasting-gb-pretrained
```

### 2. Tạo virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=hybrid-forecast --display-name "Python (hybrid-forecast)"
```

> ⚠️ Package `chronos-forecasting` cần pin đúng version `1.5.3` — các bản mới hơn (2.x, phát hành cùng Chronos-2) đã đổi API và không còn class `ChronosPipeline` dùng trong project này.

### 4. Tải dữ liệu M4

Tải `Monthly-train.csv`, `Monthly-test.csv`, `M4-info.csv` từ [Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods) và đặt vào:

```
data/raw/Train/Monthly-train.csv
data/raw/Test/Monthly-test.csv
data/raw/M4-info.csv
```

---

## ▶️ Cách chạy

### Chạy từng bước qua notebook (khuyến nghị — có biểu đồ trực quan)

Chạy tuần tự theo đúng thứ tự đánh số:

```
01_eda.ipynb  →  02_baseline_gb.ipynb  →  03_pretrained.ipynb
     →  04_ensemble.ipynb  →  05_gain_analysis.ipynb  →  06_shap_case_study.ipynb
```

### Chạy nhanh toàn bộ pipeline qua script

```bash
python main.py
```

> `main.py` chạy phần lõi của pipeline (data → GB → pretrained → ensemble → gain analysis) và lưu toàn bộ kết quả vào `outputs/`. Phần EDA trực quan và SHAP case study vẫn nên xem qua notebook.

---

## 📦 Output chính

| File                                                | Nội dung                                            |
| --------------------------------------------------- | ---------------------------------------------------- |
| `outputs/models/lgb_model.pkl`, `xgb_model.pkl` | Model GB đã train (đã tune bằng Optuna)         |
| `outputs/metrics/gb_baseline_predictions.csv`     | Dự báo GB trên validation set                     |
| `outputs/metrics/pretrained_predictions.csv`      | Dự báo Chronos zero-shot                           |
| `outputs/metrics/ensemble_predictions.csv`        | Dự báo GB / Pretrained / Ensemble (avg & weighted) |
| `outputs/metrics/gain_analysis_results.csv`       | Ensemble Gain theo từng series, kèm length_group   |
| `outputs/figures/`                                | Biểu đồ SHAP, case study, boxplot Gain            |

---

## 🛠️ Kiến thức & công cụ sử dụng

- **Decision Tree, Random Forest, Boosting**: nền tảng cho GB baseline (LightGBM, XGBoost).
- **Time-series feature engineering**: lag, rolling statistics, trend/seasonality (phân tích qua ACF/PACF ở bước EDA).
- **Cross-validation cho time-series**: time-based split, tránh leakage.
- **Optuna**: tối ưu hyperparameter (TPE sampler, Median pruner).
- **SHAP**: giải thích feature importance, so sánh cơ chế dự báo giữa các nhóm dữ liệu.
- **Kiểm định thống kê phi tham số**: Kruskal-Wallis, Mann-Whitney U test (phù hợp với dữ liệu không giả định phân phối chuẩn).

---

## 👥 Nhóm thực hiện

*(Điền tên thành viên và phân công công việc)*

| Thành viên | Vai trò                             |
| ------------ | ------------------------------------ |
| ...          | Data preprocessing & length_grouping |
| ...          | GB baseline & Optuna tuning          |
| ...          | Pretrained model & Ensemble          |
| ...          | Gain analysis, SHAP & báo cáo      |

---

## 📚 Tài liệu tham khảo

- Makridakis, S. et al. (2020). *The M4 Competition: 100,000 time series and 61 forecasting methods.* International Journal of Forecasting.
- Ansari, A. F. et al. (2024). *Chronos: Learning the Language of Time Series.*
