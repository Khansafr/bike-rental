# Struktur Proyek

submission
├───dashboard
│   ├───data.csv
│   └───dashboard.py
├───data
│   ├───day.csv
│   └───hour.csv
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt

# Bike Rental Dashboard

## Setup Environment

### Menggunakan Google Colab
1. Unduh proyek ini.
2. Buka Google Colab di browser Anda.
3. Buat notebook baru.
4. Unggah file dengan ekstensi `.ipynb`.
5. Hubungkan ke runtime yang tersedia.
6. Jalankan sel-sel kode.

### Menggunakan Shell/Terminal
Jika Anda ingin menggunakan terminal untuk setup, Anda bisa mengikuti langkah-langkah berikut:

1. Buat direktori baru untuk proyek Anda:
   ```bash
   mkdir proyek_analisis_data
   cd proyek_analisis_data

2. Install pustaka yang diperlukan menggunakan Pipenv:
   ```bash
   Copy code
   pipenv install
   pipenv shell
   pip install -r requirements.txt

3. Jalankan Aplikasi Streamlit
   Untuk menjalankan dashboard, gunakan perintah berikut di terminal:
   streamlit run dashboard.py

5. Pustaka yang Dibutuhkan
   streamlit
   matplotlib
   seaborn
   pandas
   streamlit
   scikit-learn

