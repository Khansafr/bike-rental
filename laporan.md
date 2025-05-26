# Laporan Proyek: Prediksi Harga Apartemen di Jakarta menggunakan Machine Learning

## 1. Domain Proyek

Permasalahan harga properti, khususnya apartemen di Jakarta, merupakan isu penting yang berdampak langsung terhadap keputusan investasi, pembelian hunian, dan kebijakan perumahan. Dengan populasi yang terus meningkat dan keterbatasan lahan, harga apartemen di Jakarta cenderung mengalami fluktuasi yang signifikan. Oleh karena itu, prediksi harga apartemen menjadi penting agar dapat memberikan insight kepada calon pembeli, investor, dan pengembang properti.

Masalah ini perlu diselesaikan dengan pendekatan berbasis data dan pemodelan machine learning karena:

* Diperlukan metode prediksi yang andal untuk memperkirakan harga berdasarkan faktor-faktor kompleks seperti lokasi, luas bangunan, dan fasilitas.
* Dapat membantu berbagai pihak untuk mengambil keputusan strategis secara objektif.

**Referensi**:

* Susanto, A., et al. (2023). "Property Price Forecasting Using Machine Learning: A Case Study in Jakarta." *International Journal of Data Science*, 8(2), 120–132.
* BPS DKI Jakarta (2024). Statistik Harga Properti Residensial.

## 2. Business Understanding

### Problem Statement

Bagaimana cara memprediksi harga apartemen di Jakarta berdasarkan fitur-fitur seperti lokasi, luas, jumlah kamar tidur, dan hak guna lahan?

### Goals

Membangun model prediksi harga apartemen di Jakarta yang akurat dan dapat digunakan untuk mendukung pengambilan keputusan.

### Solution Statement

Untuk mencapai tujuan tersebut, solusi yang ditawarkan mencakup:

1. **Baseline Model**: Membangun model prediksi harga menggunakan tiga algoritma:

   * Random Forest
   * XGBoost
   * Decision Tree

2. **Model Improvement**:

   * Melakukan **Hyperparameter Tuning** terhadap masing-masing model untuk meningkatkan akurasi prediksi.

Model dievaluasi menggunakan metrik evaluasi yang tepat, yaitu **R2, MAE, MSE, dan RMSE**.

## 3. Data Understanding

### Sumber Data

* Dataset diperoleh dari hasil scraping situs properti *Pinhome.id* yang berisi informasi apartemen di Jakarta.
* [Link Dataset](https://www.kaggle.com/datasets/)

### Informasi Data

* Jumlah baris awal: 2.338
* Kolom-kolom utama: `Harga`, `Luas Bangunan`, `Kamar Tidur`, `Kelurahan`, `Kecamatan`, `Kota`, `Hak Guna`.

### Deskripsi Fitur

| Fitur         | Deskripsi                                     |
| ------------- | --------------------------------------------- |
| Harga         | Harga apartemen (dalam format string rupiah)  |
| Luas Bangunan | Luas bangunan apartemen                       |
| Kamar Tidur   | Jumlah kamar tidur                            |
| Kelurahan     | Nama kelurahan apartemen                      |
| Kecamatan     | Nama kecamatan apartemen                      |
| Kota          | Nama kota/kabupaten apartemen                 |
| Hak Guna      | Hak kepemilikan (HGB, SHM, Strata Title, dll) |

### Eksplorasi Awal

* Beberapa baris memiliki missing value.
* Kolom harga perlu dibersihkan dari satuan `Rp`, `Jt`, dan `M`.
* Kolom `Kecamatan` menyertakan data redundan dengan kata `Kota`.

### Visualisasi

* Scatter plot fitur vs log harga menunjukkan korelasi yang signifikan pada fitur `Luas`, `Kamar Tidur`, dan `Harga per m2`.
* Heatmap korelasi digunakan untuk menilai kekuatan hubungan antara fitur dan target.

## 4. Data Preparation

### Teknik dan Proses:

* **Pembersihan Missing Value**: Drop baris dengan missing value.
* **Cleaning Format Harga**: Konversi harga dari format string ke float menggunakan konversi `Rp`, `Jt`, dan `M`.
* **Ekstraksi Angka** dari kolom `Luas Bangunan` dan `Kamar Tidur`.
* **Estimasi Harga per m2**: `Harga / Luas Bangunan`
* **Transformasi Logaritmik**: `Log Harga` sebagai target untuk mengurangi skewness.
* **Encoding Kategori**: Menggunakan **Frequency Encoding** untuk `Kelurahan`, `Kecamatan`, `Kota`, dan `Hak Guna`.
* **Standardisasi**: `Luas` dan `Kamar Tidur` di-scaling menggunakan `StandardScaler`.

### Alasan Tahapan

* Mengatasi perbedaan skala antar fitur.
* Menangani data kategorikal tanpa meningkatkan dimensionalitas.
* Menyesuaikan distribusi target agar lebih normal.

## 5. Modeling

### Algoritma yang Digunakan

* **Random Forest**: Model ensemble berbasis pohon keputusan.
* **XGBoost**: Gradient boosting dengan kemampuan penyesuaian regularisasi.
* **Decision Tree**: Model pohon dasar sebagai baseline sederhana.

### Parameter Baseline:

* Random Forest: `n_estimators=150, max_depth=20`
* XGBoost: `n_estimators=100, max_depth=7, learning_rate=0.1`
* Decision Tree: `max_depth=10`

### Hyperparameter Tuning:

* GridSearchCV dengan cross-validation (`cv=3`) dilakukan pada setiap model.
* Parameter disesuaikan untuk menemukan konfigurasi optimal.

### Kelebihan dan Kekurangan:

| Model         | Kelebihan                          | Kekurangan                                  |
| ------------- | ---------------------------------- | ------------------------------------------- |
| Random Forest | Tahan terhadap overfitting, robust | Training time lebih lama                    |
| XGBoost       | Akurat dan efisien untuk boosting  | Kompleks dan lebih sensitif terhadap tuning |
| Decision Tree | Sederhana, interpretatif           | Mudah overfitting                           |

### Model Terbaik

* Model terbaik dipilih berdasarkan nilai R² tertinggi dan MAE terkecil.
* **XGBoost Tuned** menunjukkan performa terbaik.

## 6. Evaluation

### Metrik Evaluasi:

* **R² Score**: Mengukur proporsi variansi yang dapat dijelaskan oleh model.
  $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**

### Hasil Evaluasi Model:

| Model               | R2     | MSE    | RMSE   | MAE    |
| ------------------- | ------ | ------ | ------ | ------ |
| XGBoost             | 0.9140 | 0.0321 | 0.1791 | 0.1342 |
| Random Forest       | 0.9092 | 0.0340 | 0.1843 | 0.1371 |
| Decision Tree       | 0.8796 | 0.0457 | 0.2137 | 0.1584 |
| XGBoost Tuned       | 0.9245 | 0.0287 | 0.1693 | 0.1259 |
| Random Forest Tuned | 0.9197 | 0.0305 | 0.1746 | 0.1290 |
| Decision Tree Tuned | 0.8847 | 0.0434 | 0.2083 | 0.1528 |

### Prediksi Sampel

Contoh hasil prediksi pada data test:

* Prediksi: Rp 15.300.000.000
* Aktual: Rp 14.800.000.000
* Selisih: Rp 500.000.000 (Error 3.4%)

## 7. Struktur Laporan

Laporan ini disusun dengan mengikuti struktur laporan data science:

1. **Domain Proyek**: Latar belakang dan urgensi masalah.
2. **Business Understanding**: Penjabaran tujuan dan solusi.
3. **Data Understanding**: Eksplorasi awal dan deskripsi fitur.
4. **Data Preparation**: Teknik preprocessing secara runtut.
5. **Modeling**: Pemilihan dan evaluasi model.
6. **Evaluation**: Penilaian performa model dan pemilihan model terbaik.

> Semua grafik dan hasil model telah divisualisasikan menggunakan matplotlib dan seaborn agar memudahkan interpretasi.

---

**Catatan**: Notebook ini mendukung reproduksibilitas penuh dengan menyimpan model terbaik (`joblib`) dan scaler yang digunakan.
