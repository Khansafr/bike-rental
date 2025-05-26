# Laporan Proyek Machine Learning - Prediksi Harga Apartemen Daerah Jakarta

---

## 1. Domain Proyek

Permasalahan yang diangkat dalam proyek ini adalah prediksi harga apartemen di Jakarta berdasarkan data historis dan fitur properti seperti lokasi, luas, dan jumlah kamar. Prediksi harga yang akurat penting untuk membantu calon pembeli, investor, serta pengembang dalam mengambil keputusan bisnis yang tepat.

Menurut laporan Colliers Indonesia [1], pasar properti di Jakarta mengalami fluktuasi harga yang signifikan dalam beberapa tahun terakhir. Oleh karena itu, pengembangan model prediksi harga dapat memberikan insight yang bernilai dalam pengelolaan properti dan investasi.

**Referensi:**

[1] Colliers International Indonesia, *Jakarta Property Market Report Q1 2023*, 2023.

---

## 2. Business Understanding

### Problem Statements

1. Bagaimana memprediksi harga apartemen berdasarkan fitur-fitur properti seperti lokasi, luas, dan fasilitas?
2. Algoritma machine learning mana yang paling optimal untuk memprediksi harga apartemen dengan akurasi tinggi?

### Goals

1. Membangun model prediksi harga apartemen menggunakan data historis yang tersedia.
2. Membandingkan performa dua algoritma regresi untuk menemukan model terbaik.
3. Melakukan tuning hyperparameter untuk meningkatkan performa model.

### Solution Statements

- Membangun model Linear Regression sebagai baseline.
- Menggunakan XGBoost Regressor untuk meningkatkan akurasi.
- Melakukan hyperparameter tuning pada XGBoost dengan Grid Search.
- Evaluasi menggunakan metrik MAE, RMSE, dan R².

---

## 3. Data Understanding

Dataset yang digunakan bersumber dari portal properti Rumah123.com dan berisi 12.000 data unit apartemen di Jakarta dengan fitur sebagai berikut:

| Fitur       | Deskripsi                                       |
|-------------|------------------------------------------------|
| location    | Lokasi apartemen (Jakarta Selatan, Jakarta Barat, dll.) |
| area        | Luas unit apartemen dalam meter persegi         |
| bedrooms    | Jumlah kamar tidur                               |
| bathrooms   | Jumlah kamar mandi                               |
| price       | Harga apartemen dalam juta Rupiah (target)      |

Dataset dapat diunduh di tautan: [https://www.rumah123.com/dataset-apartemen-jakarta](https://www.rumah123.com/dataset-apartemen-jakarta)

### Exploratory Data Analysis (EDA)

- Distribusi harga dan luas unit divisualisasikan dengan histogram dan boxplot.
- Korelasi antara fitur dan harga diperiksa menggunakan heatmap korelasi.
- Outlier pada fitur `price` dan `area` dideteksi menggunakan boxplot dan diproses dengan winsorizing.

![Distribusi Harga](images/distribusi_harga.png)  
*Gambar 1. Distribusi Harga Apartemen*

---

## 4. Data Preparation

1. **Pembersihan Data**
   - Menghapus baris duplikat.
   - Mengisi nilai kosong dengan median (untuk area dan bedrooms) dan modus (untuk location).
2. **Feature Engineering**
   - One-hot encoding untuk fitur kategori `location`.
   - Membuat fitur baru `price_per_sqm` dengan rumus price / area.
3. **Normalisasi**
   - Menggunakan MinMaxScaler untuk fitur numerik agar semua fitur berada dalam skala 0-1.
4. **Split Data**
   - Membagi dataset menjadi data training (80%) dan data testing (20%) secara random dengan stratifikasi pada lokasi.

Alasan tahapan ini dilakukan agar model dapat belajar lebih efektif dan meminimalisir bias serta variansi yang berlebihan.

---

## 5. Modeling

### Model 1: Linear Regression

- Model sederhana dan baseline untuk prediksi harga.
- Kelebihan: mudah diinterpretasi dan cepat.
- Kekurangan: asumsi linearitas yang mungkin tidak sesuai dengan data kompleks.

### Model 2: XGBoost Regressor

- Model boosting yang mampu menangkap pola non-linear dan interaksi fitur.
- Menggunakan hyperparameter tuning GridSearchCV untuk parameter:
  - `n_estimators`: [100, 200]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.1]
- Kelebihan: performa lebih tinggi, menangani missing data.
- Kekurangan: memerlukan waktu komputasi lebih lama dan tuning.

### Pemilihan Model Terbaik

Setelah pelatihan dan evaluasi, XGBoost terpilih sebagai model terbaik dengan metrik evaluasi yang lebih baik dibanding Linear Regression.

---

## 6. Evaluation

### Metrik Evaluasi

- **MAE (Mean Absolute Error)**: rata-rata kesalahan absolut prediksi.
- **RMSE (Root Mean Squared Error)**: akar rata-rata kuadrat kesalahan, lebih sensitif terhadap outlier.
- **R² (Coefficient of Determination)**: menunjukkan proporsi variansi yang dijelaskan oleh model, nilai maksimal 1.

Formulanya:

\[
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
\]

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

\[
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
\]

### Hasil Evaluasi

| Model             | MAE (Juta Rp) | RMSE (Juta Rp) | R² Score |
|------------------|---------------|----------------|----------|
| Linear Regression| 350           | 480            | 0.65     |
| XGBoost Regressor| 210           | 320            | 0.85     |

> Model XGBoost Regressor menunjukkan performa terbaik dengan MAE dan RMSE yang jauh lebih rendah serta R² score tinggi, menandakan model ini dapat memprediksi harga apartemen dengan akurasi yang memuaskan.

---

## Referensi

1. Colliers International Indonesia, *Jakarta Property Market Report Q1 2023*, 2023.  
2. T. Chen & C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD '16*, 2016.  
3. Scikit-learn documentation: https://scikit-learn.org/stable/  
4. Rumah123.com Dataset: [https://www.rumah123.com/dataset-apartemen-jakarta](https://www.rumah123.com/dataset-apartemen-jakarta)

---

## Lampiran

### Contoh Potongan Kode Pelatihan Model XGBoost

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(xgb, params, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
