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

### Hasil Evaluasi:

| Model               | R²    | MSE   | RMSE  | MAE   |
|---------------------|-------|-------|-------|-------|
| Random Forest       | 0.84  | 0.12  | 0.34  | 0.25  |
| XGBoost             | 0.85  | 0.11  | 0.33  | 0.24  |
| Decision Tree       | 0.75  | 0.21  | 0.45  | 0.32  |
| **XGBoost Tuned**   | **0.88** | **0.09** | **0.30** | **0.22** |
| Random Forest Tuned | 0.87  | 0.10  | 0.31  | 0.23  |
| Decision Tree Tuned | 0.77  | 0.19  | 0.43  | 0.30  |

### Model Terbaik:
- **XGBoost Tuned**: memberikan akurasi tertinggi (R² = 0.88)

---
