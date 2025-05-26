# Prediksi Harga Apartemen di Jakarta Menggunakan Machine Learning

## 1. Domain Proyek

### Latar Belakang
Pasar properti di Jakarta, sebagai ibu kota Indonesia, mengalami pertumbuhan yang dinamis seiring meningkatnya kebutuhan hunian di kawasan urban. Salah satu indikator utama dalam pengambilan keputusan investasi dan pembelian properti adalah harga. Namun, harga apartemen sangat bervariasi tergantung pada lokasi, fasilitas, ukuran unit, hingga status hukum lahan. Oleh karena itu, dibutuhkan sebuah pendekatan berbasis data yang dapat memberikan estimasi harga secara objektif dan terukur.

Prediksi harga apartemen berbasis Machine Learning (ML) telah menjadi fokus penelitian dalam beberapa tahun terakhir. Studi oleh S. Prasetyo dkk. (2023) menunjukkan bahwa model regresi berbasis Random Forest mampu menghasilkan prediksi harga properti dengan akurasi tinggi dibandingkan regresi linier konvensional. Selain itu, menurut Ahmed et al. (2022), penggunaan model XGBoost memberikan keunggulan dalam menangani data yang kompleks dan tidak linier.

**Referensi:**
- Prasetyo, S., et al. (2023). *Predicting House Prices Using Random Forest Regression*. Journal of Data Science, 11(2), 89–102.
- Ahmed, M., et al. (2022). *A Comparative Study of XGBoost and Random Forest for Real Estate Price Prediction*. Journal of AI Research, 45(4), 210–223.

---

## 2. Business Understanding

### Problem Statement
Bagaimana memprediksi harga apartemen di wilayah Jakarta berdasarkan fitur-fitur properti seperti lokasi, luas bangunan, jumlah kamar tidur, dan status hak guna?

### Goals
Membangun model prediksi harga apartemen yang akurat dan dapat digunakan untuk estimasi harga berdasarkan input fitur properti.

### Solution Statement
Solusi dilakukan dengan membandingkan beberapa algoritma:

- **Baseline Models**:
  - Random Forest Regressor
  - XGBoost Regressor
  - Decision Tree Regressor

- **Improved Models**:
  - Random Forest (tuning: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)
  - XGBoost (tuning: `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`)
  - Decision Tree (tuning: `max_depth`, `min_samples_split`, `min_samples_leaf`)

### Evaluation Metric
Model dievaluasi menggunakan metrik:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## 3. Data Understanding

- **Sumber Data**: Dataset hasil web scraping dari situs Pinhome.id
- **Nama File**: `dataset_apartemen_jakarta1.csv`
- **Jumlah Data**: Sekitar _n_ baris setelah pembersihan
- **Fitur-fitur**:
  - Harga (target)
  - Luas Bangunan
  - Jumlah Kamar Tidur
  - Kota, Kecamatan, Kelurahan
  - Hak Guna (status lahan)

### Exploratory Data Analysis (EDA)
- Visualisasi scatter plot `Luas (m2)` dan `Kamar Tidur` terhadap `Log Harga`
- Heatmap korelasi antar fitur

---

## 4. Data Preparation

### Langkah-langkah:
1. **Pembersihan Harga**
   - Menghilangkan simbol "Rp", titik, dan konversi "Jt"/"M" ke angka
2. **Ekstraksi Luas dan Kamar Tidur**
   - Parsing angka dari kolom string deskriptif
3. **Handling Missing Value**
   - Median untuk numerik, ‘Unknown’ untuk kategori
4. **Encoding Kategori**
   - Frequency encoding untuk `Kota`, `Kecamatan`, `Kelurahan`, `Hak Guna`
5. **Feature Engineering**
   - Penambahan fitur `Harga per m2`
6. **Transformasi Target**
   - Menggunakan `Log Harga` untuk mendekati distribusi normal
7. **Scaling**
   - StandardScaler untuk `Luas` dan `Kamar Tidur`

### Alasan:
- Encoding agar kategori bisa diproses model ML
- Log transformasi mengatasi distribusi harga yang skewed
- Scaling mempercepat dan menstabilkan pelatihan model

---

## 5. Modeling

### Model yang Digunakan:
- RandomForestRegressor
- XGBRegressor
- DecisionTreeRegressor
- GridSearchCV untuk hyperparameter tuning

### Perbandingan:
| Model | Keunggulan | Kekurangan |
|-------|------------|------------|
| Random Forest | Stabil, tidak overfit | Kurang efisien untuk realtime |
| XGBoost | Cepat dan akurat | Butuh tuning detail |
| Decision Tree | Mudah diinterpretasi | Overfitting jika tidak diatur |

---

## 6. Evaluation

### Metrik:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Hasil Evaluasi Model (Baseline vs Tuned)

| Model                  | R² Score | MSE     | RMSE    | MAE     |
|------------------------|----------|---------|---------|---------|
| **Baseline Models**    |          |         |         |         |
| XGBoost                | 0.9923   | 0.0280  | 0.1672  | 0.0870  |
| Random Forest          | 0.9935   | 0.0236  | 0.1537  | 0.0368  |
| Decision Tree          | 0.9900   | 0.0364  | 0.1908  | 0.0694  |
| **Tuned Models**       |          |         |         |         |
| Random Forest Tuned    | **0.9941** | **0.0215** | **0.1468** | 0.0410  |
| XGBoost Tuned          | 0.9928   | 0.0260  | 0.1612  | 0.0622  |
| Decision Tree Tuned    | 0.9896   | 0.0377  | 0.1941  | 0.0540  |

### Model Terbaik
- **Random Forest Tuned** menjadi model terbaik berdasarkan skor **R² = 0.9941**, serta nilai MSE dan RMSE yang paling rendah.
- Model ini menunjukkan kinerja prediktif paling tinggi dalam memprediksi harga apartemen di Jakarta.


---
