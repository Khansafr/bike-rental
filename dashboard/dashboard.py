import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Menggunakan style visualisasi Seaborn secara global
sns.set(style="whitegrid")

# Membaca dataset
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")

min_date = day_df['dteday'].min()
max_date = day_df['dteday'].max()

with st.sidebar:
    st.image("https://th.bing.com/th/id/OIP.Gu8wxJGUllQ7cOusL6JcMwAAAA?rs=1&pid=ImgDetMain")
    start_date, end_date = st.date_input(
        label="📅 Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Header
st.title("Bike Rental")
st.write("Visualisasi ini menganalisis pengaruh kondisi cuaca, hari kerja vs hari libur, dan analisis RFM (Recency, Frequency, Monetary) terhadap pola peminjaman sepeda. Dengan memperhatikan waktu peminjaman, pengaruh cuaca, serta status hari, visualisasi ini memberikan gambaran mendalam tentang bagaimana faktor-faktor tersebut memengaruhi intensitas penyewaan sepeda.")
st.markdown("---")

# 1. Visualisasi Pengaruh Cuaca terhadap Waktu Peminjaman Sepeda
st.subheader("Pengaruh Cuaca terhadap Waktu Peminjaman Sepeda")

# Fungsi untuk memplot pengaruh cuaca terhadap waktu peminjaman sepeda
def plot_weather_impact(hour_df, start_date, end_date):
    expected_columns = ['hr', 'weathersit', 'cnt', 'dteday']
    missing_columns = [col for col in expected_columns if col not in hour_df.columns]
    if missing_columns:
        st.write(f'Kolom berikut tidak ditemukan dalam hour_df: {missing_columns}')
        return

    # Konversi kolom 'dteday' dan rentang tanggal menjadi tipe datetime
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data berdasarkan rentang tanggal
    hour_df = hour_df[(hour_df['dteday'] >= start_date) & (hour_df['dteday'] <= end_date)]

    # Menghitung rata-rata jumlah peminjaman berdasarkan jam dan situasi cuaca
    weather_avg = hour_df.groupby(['hr', 'weathersit'])['cnt'].mean().reset_index()
    weather_avg['weathersit'] = weather_avg['weathersit'].map({
        1: 'Cerah',
        2: 'Berawan',
        3: 'Hujan',
        4: 'Ekstrem'
    })

    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    for i, (name, group) in enumerate(weather_avg.groupby('weathersit')):
        plt.bar(group['hr'] + i*0.2, group['cnt'], width=0.2, label=name, color=colors[i])
    plt.xlabel('Jam dalam Sehari (0-23)')
    plt.ylabel('Rata-rata Penggunaan Sepeda')
    plt.title('Pengaruh Cuaca terhadap Waktu Peminjaman Sepeda')
    plt.legend(title='Cuaca')
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

plot_weather_impact(hour_df, start_date, end_date)

st.write("""
**Insight:**
- Cuaca Cerah (Clear) memiliki penggunaan sepeda tertinggi hampir di setiap jam, terutama pada jam sibuk pagi (7-9 AM) dan sore (4-6 PM).
- Cuaca Berawan (Cloudy) menunjukkan tren penggunaan yang mirip dengan cuaca cerah, meskipun jumlahnya sedikit lebih rendah.
- Cuaca Hujan (Rain) secara signifikan menurunkan jumlah penggunaan sepeda, terutama pada jam-jam non-sibuk.
- Cuaca Buruk (Severe) hampir tidak ada penggunaan sepeda, menunjukkan cuaca ekstrem sangat membatasi aktivitas bersepeda.

Secara keseluruhan, penggunaan sepeda sangat dipengaruhi oleh kondisi cuaca, dengan tingkat penggunaan tertinggi terjadi pada jam-jam sibuk dan ketika cuaca dalam kondisi cerah.
""")

st.write("---")


# 2. Visualisasi Pengaruh Hari Kerja terhadap Waktu Peminjaman Sepeda
st.subheader("Pengaruh Hari Kerja terhadap Waktu Peminjaman Sepeda")

# Fungsi untuk memplot pengaruh hari kerja terhadap penggunaan sepeda
def plot_workingday_impact(hour_df, start_date, end_date):
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    hour_df = hour_df[(hour_df['dteday'] >= start_date) & (hour_df['dteday'] <= end_date)]
    workingday_avg = hour_df.groupby(['hr', 'workingday'])['cnt'].mean().reset_index()
    workingday_avg['workingday'] = workingday_avg['workingday'].map({1: 'Hari Kerja', 0: 'Hari Libur'})

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hr', y='cnt', hue='workingday', data=workingday_avg, marker='o')
    plt.title('Pengaruh Hari Kerja terhadap Waktu Peminjaman Sepeda')
    plt.xlabel('Jam dalam Sehari (0-23)')
    plt.ylabel('Jumlah Penggunaan Sepeda')
    plt.xticks(range(0, 24))
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

plot_workingday_impact(hour_df, start_date, end_date)

st.write("""
**Insight:**
- Pada hari kerja, penggunaan sepeda memuncak pada jam komuter pagi (sekitar jam 8) dan sore (sekitar jam 17-18).  
- Pada hari libur, penggunaan sepeda lebih merata sepanjang hari dengan sedikit peningkatan di siang hari.  
- Sepeda lebih sering digunakan sebagai moda transportasi untuk pergi dan pulang kerja pada hari kerja.  
- Penggunaan sepeda di hari libur lebih cenderung untuk rekreasi atau aktivitas santai.

Penggunaan sepeda sangat dipengaruhi oleh status hari kerja atau hari libur. Pada hari kerja, sepeda lebih sering digunakan untuk keperluan komuter, sementara pada hari libur, penggunaan sepeda lebih fleksibel dan cenderung untuk aktivitas rekreasi sepanjang hari.
""")

st.write("---")

# 3. Visualisasi RFM Analysis - Pengaruh Cuaca terhadap Penyewaan Sepeda
st.subheader("RFM Analysis - Pengaruh Cuaca terhadap Penyewaan Sepeda")

# Fungsi untuk memvisualisasikan RFM Analysis
def plot_rfm_analysis(hour_df):
    rfm_weather = hour_df.groupby(['weathersit', 'hr']).agg(
        recency=('hr', 'max'),
        frequency=('cnt', 'count'),
        monetary=('cnt', 'sum')
    ).reset_index()

    rfm_weather['recency'] = rfm_weather['recency'].max() - rfm_weather['recency']
    rfm_weather['weathersit'] = rfm_weather['weathersit'].map({
        1: 'Clear',
        2: 'Cloudy',
        3: 'Rain',
        4: 'Severe'
    })

    scaler = MinMaxScaler(feature_range=(0, 24))
    rfm_weather[['recency']] = scaler.fit_transform(rfm_weather[['recency']])
    scaler = MinMaxScaler(feature_range=(0, 800))
    rfm_weather[['frequency', 'monetary']] = scaler.fit_transform(rfm_weather[['frequency', 'monetary']])

    rfm_weather['RFM_Score'] = ((rfm_weather['recency'] + rfm_weather['frequency'] + rfm_weather['monetary']) / 3).round(0)

    plt.figure(figsize=(10, 6))
    heatmap_data = rfm_weather.pivot(index='weathersit', columns='hr', values='RFM_Score')
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.0f', linewidths=0.5, linecolor='gray')
    plt.title('Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda (RFM Analysis)', fontsize=14)
    plt.xlabel('Jam dalam Sehari (0-23)')
    plt.ylabel('Kondisi Cuaca')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

plot_rfm_analysis(hour_df)

st.write("""
**Insight:**

Analisis heatmap RFM di atas menunjukkan pengaruh kondisi cuaca terhadap penggunaan sepeda berdasarkan tiga metrik utama (Recency, Frequency, Monetary).

- Kondisi cuaca yang cerah (Clear) memiliki skor RFM yang tinggi di hampir semua jam, terutama pada jam-jam sibuk di pagi dan sore hari.
- Cuaca berawan (Cloudy) juga menunjukkan tren penggunaan yang cukup baik, meskipun tidak sekuat saat cuaca cerah.
- Saat hujan (Rain), penggunaan sepeda cenderung menurun secara signifikan, terutama di luar jam-jam komuter.
- Kondisi cuaca ekstrem (Severe) menunjukkan penggunaan sepeda yang sangat rendah di semua jam, mengindikasikan bahwa pengguna cenderung menghindari aktivitas bersepeda dalam kondisi tersebut.

Secara keseluruhan, cuaca cerah mendorong penggunaan sepeda yang lebih tinggi, sedangkan cuaca buruk (hujan atau ekstrem) berdampak negatif terhadap penyewaan sepeda.
""")

st.write("---")

# 4. Visualisasi RFM Analysis: Pengaruh Hari Kerja dan Hari Libur terhadap Penggunaan Sepeda
st.subheader('RFM Analysis: Pengaruh Hari Kerja dan Hari Libur terhadap Penggunaan Sepeda')

# Menghitung RFM berdasarkan hari kerja dan hari libur
rfm_workingday = hour_df.groupby(['hr', 'workingday']).agg(
    recency=('hr', 'max'),
    frequency=('cnt', 'count'),
    monetary=('cnt', 'sum')
).reset_index()

rfm_workingday['recency'] = rfm_workingday['recency'].max() - rfm_workingday['recency']
rfm_workingday['workingday'] = rfm_workingday['workingday'].map({1: 'Hari Kerja', 0: 'Hari Libur'})

# Normalisasi data RFM dengan skala yang sesuai
scaler = MinMaxScaler(feature_range=(0, 24))
rfm_workingday[['recency']] = scaler.fit_transform(rfm_workingday[['recency']])
scaler = MinMaxScaler(feature_range=(0, 800))
rfm_workingday[['frequency', 'monetary']] = scaler.fit_transform(rfm_workingday[['frequency', 'monetary']])

# Membuat skor RFM sederhana untuk visualisasi
rfm_workingday['RFM_Score'] = ((rfm_workingday['recency'] + 
                                rfm_workingday['frequency'] + 
                                rfm_workingday['monetary']) / 3).round(0)


heatmap_data = rfm_workingday.pivot(index='workingday', columns='hr', values='RFM_Score')

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.0f', linewidths=0.5, linecolor='gray', ax=ax)

plt.title('RFM Analysis: Pengaruh Hari Kerja dan Hari Libur terhadap Penggunaan Sepeda', fontsize=14)
plt.xlabel('Jam dalam Sehari (0-23)')
plt.ylabel('Kategori Hari')
plt.xticks(range(0, 24), labels=range(0, 24))
plt.tight_layout()

st.pyplot(fig)

st.write("""
**Insight:**

- Pada hari kerja (Hari Kerja), penggunaan sepeda paling tinggi terjadi pada jam-jam sibuk, yaitu sekitar pukul 7-9 pagi dan 17-19 sore. Hal ini menunjukkan bahwa sepeda banyak digunakan untuk kebutuhan komuter.
- Di hari libur (Hari Libur), penggunaan sepeda lebih merata sepanjang hari, dengan peningkatan yang terlihat pada siang hingga sore hari.
- RFM Score di hari libur tidak sepekat di jam-jam tertentu seperti di hari kerja, menunjukkan pola penggunaan yang lebih fleksibel.
- Secara umum, hari kerja menunjukkan penggunaan sepeda yang lebih tinggi di jam-jam tertentu, sedangkan hari libur memiliki pola yang lebih konsisten dan stabil sepanjang hari.

Kesimpulannya, strategi penyewaan sepeda bisa lebih dioptimalkan dengan menyediakan lebih banyak sepeda di pagi dan sore hari saat hari kerja, sedangkan di hari libur bisa difokuskan pada kenyamanan dan fleksibilitas pengguna sepanjang hari.
""")


