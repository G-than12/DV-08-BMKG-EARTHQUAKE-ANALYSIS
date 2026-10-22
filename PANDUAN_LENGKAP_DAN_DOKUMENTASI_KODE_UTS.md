# PANDUAN LENGKAP DAN DOKUMENTASI TEKNIS KODE UTS
## Mata Kuliah: Visualisasi Data (Kelas A)
**Studi Kasus:** Analisis Spasial-Statistik Kejadian Gempa Bumi Dirasakan di Indonesia (BMKG s.d. 13 Oktober 2026)  
**Penyusun:** Gathan Hilabi (NIM: 60324059)  
**Program Studi:** S1 Informatika — UIN K.H. Abdurrahman Wahid Pekalongan  

---

## DAFTAR ISI
1. [Pendahuluan & Konsep Utama Proyek](#1-pendahuluan--konsep-utama-proyek)
2. [Ekosistem & Pustaka yang Digunakan](#2-ekosistem--pustaka-yang-digunakan)
3. [Topik B1: Memuat dan Memahami Struktur Data](#3-topik-b1-memuat-dan-memahami-struktur-data)
4. [Topik B2: Pembersihan Data & Kontrol Kualitas Data](#4-topik-b2-pembersihan-data--kontrol-kualitas-data)
5. [Topik B3: Statistik Deskriptif, Rekayasa Fitur, dan Analisis Outlier IQR](#5-topik-b3-statistik-deskriptif-rekayasa-fitur-dan-analisis-outlier-iqr)
6. [Topik B4.1: Visualisasi Bar Chart Bulanan (Matplotlib)](#6-topik-b41-visualisasi-bar-chart-bulanan-matplotlib)
7. [Topik B4.2: Visualisasi Tiga Panel Komparatif & Spasial (Seaborn)](#7-topik-b42-visualisasi-tiga-panel-komparatif--spasial-seaborn)
8. [Topik B5.1: Analisis Korelasi Pearson & Heatmap](#8-topik-b51-analisis-korelasi-pearson--heatmap)
9. [Topik B5.2: Analisis Spasial Klaster Flores (Agustus 2026)](#9-topik-b52-analisis-spasial-klaster-flores-agustus-2026)
10. [Topik B5.3: Empat Kesimpulan Kritis Seismologi](#10-topik-b53-empat-kesimpulan-kritis-seismologi)
11. [Tanya Jawab Kritis & Panduan Mempertahankan Jawaban UTS](#11-tanya-jawab-kritis--panduan-mempertahankan-jawaban-uts)

---

## 1. Pendahuluan & Konsep Utama Proyek

Proyek ini bertujuan mengolah, menganalisis, dan memvisualisasikan data sekunder kejadian gempa bumi yang dirasakan oleh masyarakat di Indonesia dari kurun waktu **14 Desember 2022 hingga 13 Oktober 2026** berdasarkan katalog resmi Badan Meteorologi, Klimatologi, dan Geofisika (BMKG).

### Alur Kerja Rekayasa Data (*Data Pipeline*):
```text
[Data Mentah CSV (3.630 baris)]
             │
             ▼
[Audit Missing Value & Catatan Kualitas]
             │
             ▼
[Filter Anomali: FUTURE_EARTHQUAKE (-11 baris)]
             │
             ▼
[Filter Spasial: Koordinat di Luar Indonesia (-1 baris)]
             │
             ▼
[Imputasi Data Kategorikal: jenis_lokasi -> 'Tidak diketahui']
             │
             ▼
[Dataset Bersih: df_bersih (3.618 baris)]
             │
      ┌──────┴──────────────────────────────────────┐
      ▼                                             ▼
[Feature Engineering: kategori_kedalaman]   [Statistik & Deteksi Outlier (IQR)]
      │                                             │
      ├─────────────────────────────────────────────┘
      ▼
[Visualisasi Matplotlib: Tren Bulanan 2026 + Anotasi Puncak Gempa]
      │
      ▼
[Visualisasi Seaborn: 3 Panel (KDE Dist, Boxplot Lokasi, Scatter Spasial)]
      │
      ▼
[Analisis Korelasi Pearson & Klaster Spasial Flores M7.0]
      │
      ▼
[Kesimpulan Kritis & Pelaporan Akademik]
```

---

## 2. Ekosistem & Pustaka yang Digunakan

Setiap pustaka dipilih berdasarkan peran spesifiknya dalam alur analisis:

| Pustaka / Modul | Peran Utama | Alasan Pemilihan |
| :--- | :--- | :--- |
| `pandas` | Struktur Data & Manipulasi Tabular | Menyediakan objek `DataFrame` berkinerja tinggi untuk *filtering*, *grouping*, *pivoting*, dan audit nilai kosong. |
| `numpy` | Komputasi Numerik | Operasi vektor matematika dan penanganan nilai `NaN` (*Not a Number*). |
| `matplotlib.pyplot` | Visualisasi Grafis Tingkat Rendah (*Low-level*) | Memberikan kontrol granular pada elemen grafik (penyesuaian *spines*, *bar_label*, pembuatan anotasi panah khusus). |
| `seaborn` | Visualisasi Statistik Tingkat Tinggi (*High-level*) | Mempermudah pemetaan variabel data ke estetika visual (kurva KDE, distribusi kuartil boxplot, pemetaan *hue* koordinat spasial). |
| `io` & `google.colab` | Mekanisme Input/Output Berkas | Standar pengunggahan berkas interaktif pada lingkungan Google Colab dengan *fallback* pembacaan lokal. |

---

## 3. Topik B1: Memuat dan Memahami Struktur Data

### 3.1. Teori & Tujuan Analisis
Sebelum melakukan pembersihan atau visualisasi, langkah fundamental dalam metodologi data science adalah memahami bentuk (*shape*), sampel awal data (*head*), tipe data (*dtype*), dan keberadaan nilai hilang (*missing values*).

### 3.2. Kode Python
```python
# Sel 1: Pemuatan Modul dan Berkas
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
    df_uploaded = pd.read_csv(io.BytesIO(uploaded[fn]))

# Sel 3: Pemeriksaan Karakteristik Data
df = df_uploaded.copy()

# 1. Tampilkan jumlah baris dan kolom (shape)
print("=" * 60)
print(f"Bentuk data (shape): {df.shape[0]} baris x {df.shape[1]} kolom")
print("=" * 60)

# 2. Tampilkan 5 baris pertama
print("\n--- 5 Baris Pertama Data Gempa ---")
display(df.head())

# 3. Tampilkan ringkasan info()
print("\n--- Ringkasan Struktur Data (df.info()) ---")
df.info()

# 4. Tampilkan jumlah nilai kosong per kolom
print("\n--- Jumlah Nilai Kosong per Kolom ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
```

### 3.3. Penjelasan Baris per Baris
1. `df_uploaded = pd.read_csv(io.BytesIO(uploaded[fn]))`: Mengonversi *byte stream* dari pengunggahan menjadi DataFrame Pandas.
2. `df = df_uploaded.copy()`: Membuat salinan independen agar data mentah asli tidak termodifikasi jika terjadi kesalahan operasi.
3. `df.shape`: Mengembalikan tuple `(3630, 13)` yang merefleksikan 3.630 baris catatan gempa dan 13 fitur/kolom.
4. `df.info()`: Memeriksa tipe data. Teridentifikasi ada 5 kolom bertipe numerik integer (`int64`), 3 kolom float (`float64`), dan 5 kolom teks/objek (`object`/`string`).
5. `df.isnull().sum()`: Menjumlahkan nilai `NaN` per kolom untuk mengidentifikasi kolom yang memerlukan penanganan khusus (*cleaning*).

### 3.4. Dokumentasi Output Lengkap
```text
Bentuk data (shape): 3630 baris x 13 kolom

--- 5 Baris Pertama Data Gempa ---
         id_gempa            waktu_utc  tahun  bulan  tanggal  lintang   bujur  magnitudo  kedalaman_km                                             lokasi jenis_lokasi                          dirasakan_di catatan_kualitas
0  20221214195822  2022-12-14 12:58:22   2022     12       14    -7.50  106.76        3.9            24  Pusat gempa berada di laut 61 km Tenggara Kab....         Laut                     II - III Sukabumi              NaN
1  20221214231338  2022-12-14 16:13:38   2022     12       14    -3.72  127.90        3.0            10       Pusat gempa berada di laut 31 km barat Ambon         Laut                       II Hative Besar              NaN
2  20221215103109  2022-12-15 03:31:09   2022     12       15     2.88  127.50        5.0            13  Pusat gempa berada di laut 127 km BaratLaut Da...         Laut                    II Morotai Selatan              NaN
3  20221215151321  2022-12-15 08:13:21   2022     12       15    -8.22  115.61        3.6            12  Pusat gempa berada di laut 17 km TimurLaut Kar...         Laut            II Karangasem, II Tejakula              NaN
4  20221215164554  2022-12-15 09:45:54   2022     12       15    -8.23  115.57        4.1            10  Pusat gempa berada di laut 14 km Timurlaut Kar...         Laut  II - III Karangasem, II Lombok Barat              NaN

--- Ringkasan Struktur Data (df.info()) ---
Data columns (total 13 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   id_gempa          3630 non-null   int64  
 1   waktu_utc         3630 non-null   str    
 2   tahun             3630 non-null   int64  
 3   bulan             3630 non-null   int64  
 4   tanggal           3630 non-null   int64  
 5   lintang           3630 non-null   float64
 6   bujur             3630 non-null   float64
 7   magnitudo         3630 non-null   float64
 8   kedalaman_km      3630 non-null   int64  
 9   lokasi            3630 non-null   str    
 10  jenis_lokasi      3613 non-null   str    
 11  dirasakan_di      3619 non-null   str    
 12  catatan_kualitas  11 non-null     str    

--- Jumlah Nilai Kosong per Kolom ---
jenis_lokasi          17
dirasakan_di          11
catatan_kualitas    3619
dtype: int64
```

### 3.5. Analisis Kritis
* Kolom `catatan_kualitas` memiliki 3.619 baris kosong (`NaN`), artinya hanya **11 baris** yang memiliki nilai isi. Dalam audit data, keberadaan sedikit catatan kualitas ini mengindikasikan anomali atau penanda khusus yang harus dianalisis pada tahap *data cleaning*.
* Kolom `jenis_lokasi` kehilangan 17 baris informasi geologis (Darat/Laut), sedangkan `dirasakan_di` memiliki 11 baris kosong.

---

## 4. Topik B2: Pembersihan Data & Kontrol Kualitas Data

### 4.1. Teori & Tujuan Analisis
Data mentah dari instrumen atau katalog web kerap kali mengandung *test data*, *typo entry*, duplikasi, atau nilai *out of bounds*. Pembersihan data menjamin bahwa analisis statistik tidak bias akibat data fiktif atau koordinat yang berada di luar yurisdiksi kepulauan Indonesia ($90^\circ\text{ BT} - 145^\circ\text{ BT}$ dan $-11^\circ\text{ LS} - 6^\circ\text{ LU}$).

### 4.2. Kode Python
```python
# Sel 6: Pembersihan Data
# Langkah 1: Buang baris dengan catatan_kualitas == "FUTURE_EARTHQUAKE"
baris_awal = len(df)
baris_future = (df['catatan_kualitas'] == 'FUTURE_EARTHQUAKE').sum()
df_bersih = df[df['catatan_kualitas'] != 'FUTURE_EARTHQUAKE'].copy()
print(f"1. Baris berstatus 'FUTURE_EARTHQUAKE' yang dibuang : {baris_future} baris")
print(f"   Jumlah baris setelah langkah 1                  : {len(df_bersih)} baris")

# Langkah 2: Jalankan describe() pada kolom lintang dan bujur
print("\n2. Statistik Deskriptif Lintang dan Bujur:")
display(df_bersih[['lintang', 'bujur']].describe())

# Temukan nilai koordinat yang mustahil untuk wilayah Indonesia (acuan: 90° - 145° BT)
koordinat_mustahil = df_bersih[(df_bersih['bujur'] < 90) | (df_bersih['bujur'] > 145)]
print(f"\n   Ditemukan {len(koordinat_mustahil)} baris koordinat bujur di luar batas wilayah Indonesia:")
display(koordinat_mustahil[['id_gempa', 'waktu_utc', 'lintang', 'bujur', 'lokasi']])

# Buang baris yang memiliki nilai bujur mustahil
df_bersih = df_bersih[(df_bersih['bujur'] >= 90) & (df_bersih['bujur'] <= 145)].copy()
print(f"   Jumlah baris setelah membuang koordinat mustahil : {len(df_bersih)} baris")

# Langkah 3: Periksa duplikasi berdasarkan id_gempa
jumlah_duplikasi = df_bersih.duplicated(subset=['id_gempa']).sum()
print(f"\n3. Jumlah duplikasi berdasarkan 'id_gempa'          : {jumlah_duplikasi} baris (tidak ada duplikasi)")

# Langkah 4: Isi nilai kosong pada jenis_lokasi dengan 'Tidak diketahui'
kosong_sebelum = df_bersih['jenis_lokasi'].isnull().sum()
df_bersih['jenis_lokasi'] = df_bersih['jenis_lokasi'].fillna('Tidak diketahui')
kosong_setelah = df_bersih['jenis_lokasi'].isnull().sum()
print(f"\n4. Mengisi nilai kosong pada jenis_lokasi:")
print(f"   - Nilai kosong sebelum imputasi : {kosong_sebelum}")
print(f"   - Nilai kosong setelah imputasi : {kosong_setelah}")

# Laporan akhir ukuran df_bersih
print("\n" + "=" * 60)
print(f"LAPORAN AKHIR: DataFrame df_bersih memiliki {df_bersih.shape[0]} baris dan {df_bersih.shape[1]} kolom.")
print("=" * 60)
```

### 4.3. Penjelasan Baris per Baris
1. `df['catatan_kualitas'] != 'FUTURE_EARTHQUAKE'`: Membuang 11 baris data uji (*future earthquakes*) yang bertanggal di masa depan, mengurangi ukuran dari 3.630 menjadi 3.619 baris.
2. `describe()` Lintang dan Bujur: Mengungkap bahwa nilai minimum bujur adalah **1.39° BT**, yang berada di Benua Afrika (Samudra Atlantik), bukan Indonesia.
3. `(df_bersih['bujur'] < 90) | (df_bersih['bujur'] > 145)`: Menangkap baris indeks `147` dengan lokasi teks *"Pusat gempa berada di laut 78 km Baratlaut Jayapura"*. Nilai `1.39` jelas merupakan kesalahan ketik (*data entry error* yang seharusnya $\approx 140.39^\circ$).
4. Menghapus baris bujur mustahil tersebut (`3.619 -> 3.618 baris`).
5. `df_bersih.duplicated(subset=['id_gempa']).sum()`: Memastikan setiap kejadian memiliki pengenal unik (*primary key*) yang tidak ganda. Hasilnya `0` (bersih).
6. `.fillna('Tidak diketahui')`: Mengganti 17 nilai `NaN` pada kolom kategorikal `jenis_lokasi` agar tidak hilang saat dilakukan pengelompokan (*grouping/pivot*).

### 4.4. Dokumentasi Output Lengkap
```text
1. Baris berstatus 'FUTURE_EARTHQUAKE' yang dibuang : 11 baris
   Jumlah baris setelah langkah 1                  : 3619 baris

2. Statistik Deskriptif Lintang dan Bujur:
           lintang        bujur
count  3619.000000  3619.000000
mean     -3.611050   118.562783
std       3.932918    12.826945
min     -11.780000     1.390000
25%      -7.150000   107.600000
50%      -3.650000   120.550000
75%      -1.155000   126.465000
max       8.530000   143.070000

   Ditemukan 1 baris koordinat bujur di luar batas wilayah Indonesia:
           id_gempa            waktu_utc  lintang  bujur                                             lokasi
147  20230118223523  2023-01-18 15:35:23    -2.25   1.39  Pusat gempa berada di laut 78 km Baratlaut Jay...
   Jumlah baris setelah membuang koordinat mustahil : 3618 baris

3. Jumlah duplikasi berdasarkan 'id_gempa'          : 0 baris (tidak ada duplikasi)

4. Mengisi nilai kosong pada jenis_lokasi:
   - Nilai kosong sebelum imputasi : 17
   - Nilai kosong setelah imputasi : 0

============================================================
LAPORAN AKHIR: DataFrame df_bersih memiliki 3618 baris dan 13 kolom.
============================================================
```

---

## 5. Topik B3: Statistik Deskriptif, Rekayasa Fitur, dan Analisis Outlier IQR

### 5.1. Teori & Tujuan Analisis
* **Klasifikasi Kedalaman Seismologi:** Gempa bumi memiliki mekanisme fisik berbeda bergantung kedalamannya:
  $$\text{Kategori} = \begin{cases} \text{Dangkal}, & \text{kedalaman} < 60\text{ km} \\ \text{Menengah}, & 60 \le \text{kedalaman} \le 300\text{ km} \\ \text{Dalam}, & \text{kedalaman} > 300\text{ km} \end{cases}$$
* **Metode Tukey's IQR untuk Outlier:**
  Kuartil bawah ($Q_1$ pada persentil 25%), Kuartil atas ($Q_3$ pada persentil 75%).
  $$IQR = Q_3 - Q_1$$
  $$\text{Batas Atas Outlier} = Q_3 + 1.5 \times IQR$$
  Sebuah gempa dianggap *outlier ekstrim* jika magnitudonya melampaui batas atas ini.

### 5.2. Kode Python
```python
# Sel 9: Analisis Statistik, Pengelompokan, dan Outlier
# 1. Fungsi kategori_kedalaman(km) dan penerapannya
def kategori_kedalaman(km):
    if km < 60:
        return 'Dangkal'
    elif km <= 300:
        return 'Menengah'
    else:
        return 'Dalam'

df_bersih['kategori_kedalaman'] = df_bersih['kedalaman_km'].apply(kategori_kedalaman)
print("1. Jumlah Tiap Kategori Kedalaman:")
display(df_bersih['kategori_kedalaman'].value_counts())

# 2. Statistik deskriptif magnitudo dan kedalaman_km
print("\n2. Statistik Deskriptif Magnitudo dan Kedalaman (km):")
display(df_bersih[['magnitudo', 'kedalaman_km']].describe())

mean_depth = df_bersih['kedalaman_km'].mean()
median_depth = df_bersih['kedalaman_km'].median()
print(f"   - Mean Kedalaman   : {mean_depth:.2f} km")
print(f"   - Median Kedalaman : {median_depth:.2f} km")

# 3. Groupby per tahun (Jumlah kejadian, rata-rata magnitudo, dan rentang waktu pencatatan)
print("\n3. Jumlah Kejadian dan Rata-rata Magnitudo per Tahun:")
rekap_tahunan = df_bersih.groupby('tahun').agg(
    jumlah_kejadian=('id_gempa', 'count'),
    rata_rata_magnitudo=('magnitudo', 'mean')
).reset_index()
display(rekap_tahunan)

print("\n   Bukti Rentang Waktu Pencatatan per Tahun (min dan max waktu_utc):")
rentang_waktu = df_bersih.groupby('tahun')['waktu_utc'].agg(['min', 'max'])
display(rentang_waktu)

# 4. Pivot table: jenis_lokasi vs kategori_kedalaman
print("\n4. Pivot Table Jumlah Kejadian (jenis_lokasi x kategori_kedalaman):")
tabel_pivot = df_bersih.pivot_table(
    index='jenis_lokasi',
    columns='kategori_kedalaman',
    values='id_gempa',
    aggfunc='count',
    fill_value=0
)[['Dangkal', 'Menengah', 'Dalam']]
display(tabel_pivot)

# 5. Metode IQR pada magnitudo
q1_mag = df_bersih['magnitudo'].quantile(0.25)
q3_mag = df_bersih['magnitudo'].quantile(0.75)
iqr_mag = q3_mag - q1_mag
batas_atas_mag = q3_mag + 1.5 * iqr_mag
outlier_mag = df_bersih[df_bersih['magnitudo'] > batas_atas_mag]

print("\n5. Analisis Outlier Magnitudo Menggunakan Metode IQR:")
print(f"   - Kuartil 1 (Q1) : {q1_mag:.2f}")
print(f"   - Kuartil 3 (Q3) : {q3_mag:.2f}")
print(f"   - Rentang IQR    : {iqr_mag:.2f}")
print(f"   - Batas Atas     : {batas_atas_mag:.2f}")
print(f"   - Jumlah Outlier : {len(outlier_mag)} kejadian")
print(f"   - Status Gempa Nagekeo (M7.0) : {'OUTLIER' if 7.0 > batas_atas_mag else 'BUKAN OUTLIER'} (karena 7.0 > {batas_atas_mag:.2f})")
```

### 5.3. Penjelasan Baris per Baris
1. `apply(kategori_kedalaman)`: Menerapkan fungsi logika percabangan ke seluruh baris kolom `kedalaman_km` secara tervektorisasi.
2. `mean()` vs `median()`: Rata-rata kedalaman sebesar $20,04\text{ km}$ jauh lebih besar dari median sebesar $10,00\text{ km}$. Ini membuktikan distribusi kedalaman sangat condong ke kanan (*right-skewed*) akibat keberadaan beberapa gempa berhiposenter sangat dalam (hingga $636\text{ km}$).
3. `rekap_tahunan`: Menghitung total gempa dan magnitudo rata-rata tiap tahun kalender.
4. `agg(['min', 'max'])`: Menampilkan tanggal awal dan akhir pencatatan tiap tahun untuk menjawab anomali mengapa tahun 2022 hanya 36 gempa (hanya tercatat 14–31 Desember) dan 2026 tercatat 927 gempa (hanya s.d. 13 Oktober).
5. `pivot_table`: Menata frekuensi kejadian silang antar dua variabel kategorikal (`jenis_lokasi` dan `kategori_kedalaman`).
6. `quantile(0.25)` dan `quantile(0.75)`: Mengambil titik data persentil ke-25 dan ke-75 secara statistik murni.

### 5.4. Dokumentasi Output Lengkap
```text
1. Jumlah Tiap Kategori Kedalaman:
kategori_kedalaman
Dangkal     3366
Menengah     242
Dalam         10
Name: count, dtype: int64

2. Statistik Deskriptif Magnitudo dan Kedalaman (km):
         magnitudo  kedalaman_km
count  3618.000000   3618.000000
mean      3.969652     20.035379
std       1.034726     38.823789
min       1.400000      1.000000
25%       3.200000      6.000000
50%       4.000000     10.000000
75%       4.700000     17.000000
max       7.900000    636.000000
   - Mean Kedalaman   : 20.04 km
   - Median Kedalaman : 10.00 km

3. Jumlah Kejadian dan Rata-rata Magnitudo per Tahun:
   tahun  jumlah_kejadian  rata_rata_magnitudo
0   2022               36             3.636111
1   2023              928             3.958297
2   2024              755             4.036026
3   2025              972             3.900412
4   2026              927             4.012513

   Bukti Rentang Waktu Pencatatan per Tahun (min dan max waktu_utc):
                       min                  max
tahun                                          
2022   2022-12-14 12:58:22  2022-12-31 17:22:07
2023   2023-01-01 10:52:39  2023-12-31 15:15:10
2024   2024-01-01 03:22:46  2024-12-30 16:21:00
2025   2025-01-01 01:44:13  2025-12-31 06:28:08
2026   2026-01-01 06:02:00  2026-10-13 10:22:02

4. Pivot Table Jumlah Kejadian (jenis_lokasi x kategori_kedalaman):
kategori_kedalaman  Dangkal  Menengah  Dalam
jenis_lokasi                                
Darat                  1791        71      0
Laut                   1560       169     10
Tidak diketahui          15         2      0

5. Analisis Outlier Magnitudo Menggunakan Metode IQR:
   - Kuartil 1 (Q1) : 3.20
   - Kuartil 3 (Q3) : 4.70
   - Rentang IQR    : 1.50
   - Batas Atas     : 6.95
   - Jumlah Outlier : 14 kejadian
   - Status Gempa Nagekeo (M7.0) : OUTLIER (karena 7.0 > 6.95)
```

---

## 6. Topik B4.1: Visualisasi Bar Chart Bulanan (Matplotlib)

### 6.1. Teori & Prinsip Visual Storytelling
* **Penekanan Visual (*Preattentive Attribute*):** Jika semua batang diberi warna seragam, pembaca harus memindai seluruh grafik. Dengan memberikan warna kontras (merah bata `#e74c3c`) pada bulan Agustus dibandingkan batang lainnya (biru gelap `#2c3e50`), atensi audiens langsung tertuju pada anomali lonjakan.
* **Reduksi Beban Kognitif (*Data-to-Ink Ratio*):** 
  * Menampilkan angka tepat di atas batang (`ax.bar_label()`) mengeliminasi kebutuhan mata bergerak bolak-balik ke sumbu Y.
  * Menghilangkan garis tepi atas (*top spine*) dan kanan (*right spine*) menciptakan ruang grafik yang bersih (*clutter-free*).
  * Menempatkan kotak anotasi panah di ruang kosong bulan Juni-Juli agar tidak menutupi visual batang utama.

### 6.2. Kode Python
```python
# Sel 12: Bar Chart Matplotlib
# Filter data tahun 2026
df_2026 = df_bersih[df_bersih['tahun'] == 2026]

# Hitung jumlah kejadian per bulan (Januari s.d. Oktober)
bulan_series = df_2026.groupby('bulan')['id_gempa'].count().reindex(range(1, 11), fill_value=0)
nama_bulan = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober']

# Batang bulan Agustus (indeks bulan 8) diberi warna merah berbeda dari bulan lainnya (biru gelap)
warna_batang = ['#e74c3c' if b == 8 else '#2c3e50' for b in bulan_series.index]

# Membuat figure dan axes
fig, ax = plt.subplots(figsize=(11, 6))

# Plot bar chart
bars = ax.bar(nama_bulan, bulan_series.values, color=warna_batang, width=0.65, edgecolor='black', linewidth=0.8)

# Menampilkan nilai angka tepat di atas setiap batang
ax.bar_label(bars, padding=4, fontsize=10, fontweight='bold', color='#2c3e50')

# Judul dan label sumbu
ax.set_title('Jumlah Kejadian Gempa Dirasakan per Bulan di Indonesia (Tahun 2026)', fontsize=14, fontweight='bold', pad=16)
ax.set_xlabel('Bulan (Tahun 2026)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Jumlah Kejadian Gempa', fontsize=11, fontweight='bold', labelpad=10)

# Grid horizontal
ax.grid(axis='y', linestyle='--', alpha=0.7, color='gray')
ax.set_axisbelow(True)

# Sembunyikan garis tepi atas dan kanan
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Atur batas sumbu y agar proporsional dan lega
ax.set_ylim(0, 225)

# Anotasi yang menunjuk ke batang bulan Agustus secara rapi tanpa menutupi angka
ax.annotate(
    'Puncak Kejadian (186):\nGempa Nagekeo M7.0\n& Gempa Susulan',
    xy=(6.67, 160),        # Menunjuk ke bahu kiri batang Agustus
    xytext=(5.1, 172),      # Kotak diletakkan di atas ruang Juni-Juli
    ha='right',
    va='center',
    arrowprops=dict(
        arrowstyle='->',
        connectionstyle='arc3,rad=-0.12',
        color='#c0392b',
        lw=2
    ),
    fontsize=9.5,
    fontweight='bold',
    color='#78281f',
    bbox=dict(boxstyle='round,pad=0.55', facecolor='#fce4ec', edgecolor='#e74c3c', linewidth=1.2)
)

plt.tight_layout()
plt.show()
```

### 6.3. Nilai Distribusi Kejadian Bulanan 2026
| Bulan | Jumlah Kejadian | Warna Batang | Keterangan |
| :--- | :---: | :---: | :--- |
| Januari | 73 | Biru Gelap (`#2c3e50`) | Aktivitas normal |
| Februari | 68 | Biru Gelap (`#2c3e50`) | Aktivitas normal |
| Maret | 87 | Biru Gelap (`#2c3e50`) | Fluktuasi musiman |
| April | 93 | Biru Gelap (`#2c3e50`) | Peningkatan sedang |
| Mei | 84 | Biru Gelap (`#2c3e50`) | Aktivitas normal |
| Juni | 68 | Biru Gelap (`#2c3e50`) | Aktivitas normal |
| Juli | 81 | Biru Gelap (`#2c3e50`) | Aktivitas normal |
| **Agustus** | **186** | **Merah Bata (`#e74c3c`)** | **Puncak Kejadian (Gempa Nagekeo M7.0 & susulan)** |
| September | 104 | Biru Gelap (`#2c3e50`) | Peluruhan gempa susulan (*aftershocks*) |
| Oktober | 83 | Biru Gelap (`#2c3e50`) | Tercatat s.d. 13 Oktober 2026 |

---

## 7. Topik B4.2: Visualisasi Tiga Panel Komparatif & Spasial (Seaborn)

### 7.1. Teori & Desain Panel Visual
* **Panel 1 (Histogram & KDE):** Memperlihatkan fungsi kepekatan probabilitas kontinu di atas histogram frekuensi dengan partisi 25 baris (*bins*).
* **Panel 2 (Boxplot Sebaran & Median):** Menggambarkan 5 angka ringkasan statistik (Min, $Q_1$, Median, $Q_3$, Max) untuk menguji hipotesis perbedaan ambang terasa antara gempa darat dan laut.
* **Panel 3 (Scatter Plot Sebaran Episenter):** Memetakan koordinat spasial lintang dan bujur di peta Indonesia. Variabel ketiga (*kategori_kedalaman*) dipetakan ke atribut warna (*hue*).

### 7.2. Kode Python
```python
# Sel 13: Tiga Panel Seaborn
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# (i) Panel 1: Histogram magnitudo dengan KDE (bins = 25)
sns.histplot(
    data=df_bersih,
    x='magnitudo',
    bins=25,
    kde=True,
    color='#1f77b4',
    edgecolor='black',
    linewidth=0.5,
    ax=axes[0]
)
axes[0].set_title('(i) Distribusi Frekuensi Magnitudo Gempa', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Magnitudo', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Frekuensi Kejadian', fontsize=10, fontweight='bold')
axes[0].grid(axis='y', linestyle=':', alpha=0.6)

# (ii) Panel 2: Boxplot magnitudo per jenis_lokasi (urutan Darat, Laut, Tidak diketahui)
urutan_lokasi = ['Darat', 'Laut', 'Tidak diketahui']
palet_lokasi = {'Darat': '#2ecc71', 'Laut': '#3498db', 'Tidak diketahui': '#95a5a6'}

sns.boxplot(
    data=df_bersih,
    x='jenis_lokasi',
    y='magnitudo',
    order=urutan_lokasi,
    hue='jenis_lokasi',
    palette=palet_lokasi,
    legend=False,
    width=0.5,
    ax=axes[1]
)
axes[1].set_title('(ii) Distribusi Magnitudo per Jenis Lokasi', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Jenis Lokasi', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Magnitudo', fontsize=10, fontweight='bold')
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

# (iii) Panel 3: Scatter plot bujur vs lintang berdasarkan kategori_kedalaman
palet_kedalaman = {'Dangkal': '#3498db', 'Menengah': '#e67e22', 'Dalam': '#e74c3c'}
urutan_kedalaman = ['Dangkal', 'Menengah', 'Dalam']

sns.scatterplot(
    data=df_bersih,
    x='bujur',
    y='lintang',
    hue='kategori_kedalaman',
    hue_order=urutan_kedalaman,
    palette=palet_kedalaman,
    alpha=0.65,
    s=28,
    edgecolor='none',
    ax=axes[2]
)
axes[2].set_title('(iii) Sebaran Spasial Episenter Gempa Indonesia', fontsize=12, fontweight='bold', pad=10)
axes[2].set_xlabel('Bujur Timur (°BT)', fontsize=10, fontweight='bold')
axes[2].set_ylabel('Lintang (°LS / °LU)', fontsize=10, fontweight='bold')
axes[2].legend(title='Kategori Kedalaman', loc='lower left', framealpha=0.9)
axes[2].grid(True, linestyle=':', alpha=0.6)

# Simpan figure beresolusi tinggi sebelum ditampilkan
plt.tight_layout()
fig.savefig('uts_grafik.png', dpi=300, bbox_inches='tight')
print("Figure berhasil disimpan ke file 'uts_grafik.png' sebelum ditampilkan.")
plt.show()

# Menampilkan bukti nilai median magnitudo per jenis lokasi
print("\nBukti Nilai Median Magnitudo per Jenis Lokasi:")
display(df_bersih.groupby('jenis_lokasi')['magnitudo'].median())
```

### 7.3. Dokumentasi Output Lengkap
```text
Figure berhasil disimpan ke file 'uts_grafik.png' sebelum ditampilkan.

Bukti Nilai Median Magnitudo per Jenis Lokasi:
jenis_lokasi
Darat              3.4
Laut               4.5
Tidak diketahui    4.2
Name: magnitudo, dtype: float64
```

### 7.4. Analisis Temuan Grafik
1. **Histogram & KDE:** Puncak kurva densitas berada pada kisaran $M = 3,2 - 4,5$. Distribusi condong ke kanan (*positively skewed*), merefleksikan hukum Gutenberg-Richter dalam seismologi bahwa gempa bermagnitudo kecil terjadi dengan frekuensi eksponensial lebih besar dibanding gempa berkekuatan merusak ($M \ge 6,0$).
2. **Boxplot Komparatif:** Median gempa Laut ($M = 4,5$) secara signifikan lebih tinggi daripada gempa Darat ($M = 3,4$). Gempa di daratan sudah cukup menggetarkan bangunan dan dirasakan warga pada skala M3, sedangkan gempa di lautan membutuhkan magnitudo lebih tinggi agar gelombang energinya dapat merambat hingga ke pesisir.
3. **Scatter Spasial:** Pola titik membentang membentuk garis busur (*volcanic arc & subduction zone*) mengikuti Busur Sunda (barat Sumatera, selatan Jawa, Bali, Nusa Tenggara), Busur Banda, lengan tektonik Sulawesi, dan utara Papua. Gempa dalam ($>300\text{ km}$, warna merah) terisolasi di Laut Jawa dan Laut Flores, sesuai zona subduksi lempeng Indo-Australia yang menunjam jauh ke bawah lempeng Eurasia.

---

## 8. Topik B5.1: Analisis Korelasi Pearson & Heatmap

### 8.1. Teori & Metrik Statistik
Koefisien korelasi linier Pearson ($r$) mengukur kekuatan dan arah hubungan linear antara dua variabel kuantitatif:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
* Nilai $r \in [-1, 1]$.
* Nilai mendekati $+1$ mengindikasikan hubungan searah yang kuat.
* Nilai mendekati $0$ mengindikasikan tidak adanya hubungan linear.

### 8.2. Kode Python
```python
# Sel 16: Heatmap Korelasi Pearson
kolom_fitur = ['magnitudo', 'kedalaman_km', 'lintang', 'bujur']
matriks_kor = df_bersih[kolom_fitur].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    matriks_kor,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    vmin=-1,
    vmax=1,
    linewidths=0.8,
    cbar_kws={'label': 'Koefisien Korelasi Pearson (r)'}
)
plt.title('Heatmap Matriks Korelasi Parameter Gempa Bumi', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.show()

# Menampilkan tabel matriks korelasi
print("Matriks Korelasi Pearson:")
display(matriks_kor)

# Mencari pasangan korelasi terkuat (di luar diagonal utama)
corr_unstack = matriks_kor.unstack()
corr_offdiag = corr_unstack[corr_unstack.index.get_level_values(0) != corr_unstack.index.get_level_values(1)]
pasangan_terkuat = corr_offdiag.abs().idxmax()
nilai_korelasi_terkuat = corr_offdiag[pasangan_terkuat]

print(f"\nPasangan dengan korelasi terkuat : {pasangan_terkuat[0]} dan {pasangan_terkuat[1]}")
print(f"Nilai koefisien korelasi (r)     : {nilai_korelasi_terkuat:.4f} (atau dibulatkan {nilai_korelasi_terkuat:.2f})")
```

### 8.3. Dokumentasi Output Lengkap
```text
Matriks Korelasi Pearson:
              magnitudo  kedalaman_km   lintang     bujur
magnitudo      1.000000      0.371914  0.113686  0.004551
kedalaman_km   0.371914      1.000000 -0.015502  0.016930
lintang        0.113686     -0.015502  1.000000 -0.032776
bujur          0.004551      0.016930 -0.032776  1.000000

Pasangan dengan korelasi terkuat : magnitudo dan kedalaman_km
Nilai koefisien korelasi (r)     : 0.3719 (atau dibulatkan 0.37)
```

### 8.4. Interpretasi Seismologi Kritis
* **Pasangan Terkuat:** `magnitudo` dan `kedalaman_km` dengan koefisien korelasi $r = 0,3719$ (positif moderat).
* **Makna Fisik:** Korelasi positif ini bukan berarti kedalaman menyebabkan kekuatan gempa membesar, melainkan merupakan **bias seleksi data gempa dirasakan**. Gempa yang hiposenternya dalam ($>100\text{ km}$) energinya diredam (*attenuation*) oleh lapisan mantel dan kerak bumi. Oleh karena itu, gempa dalam **hanya akan tercatat jika magnitudonya besar** ($M \ge 5,5$) sehingga getarannya mampu mencapai permukaan bumi dan dirasakan manusia.
* **Korelasi Koordinat ($r \approx 0$):** Variabel koordinat geografis (`lintang` dan `bujur`) memiliki korelasi mendekati nol terhadap magnitudo ($r = 0,11$ dan $r = 0,004$). Hal ini membuktikan bahwa potensi kekuatan gempa bumi di Indonesia tidak bergantung secara linier pada posisi barat-ke-timur atau utara-ke-selatan, melainkan ditentukan oleh struktur tektonik lokal dan akumulasi regangan (*strain accumulation*) pada sesar aktif.

---

## 9. Topik B5.2: Analisis Spasial Klaster Flores (Agustus 2026)

### 9.1. Teori & Konteks Geotektonik
Wilayah Flores, Nusa Tenggara Timur, dipengaruhi oleh struktur sesar naik belakang busur kepulauan (*Flores Back-Arc Thrust*). Pada tanggal 15 Agustus 2026, terjadi gempa bumi utama (*mainshock*) berkekuatan signifikan di timur laut Nagekeo yang kemudian memicu pelepasan energi berupa deretan gempa susulan (*aftershock swarm*).

### 9.2. Kode Python
```python
# Sel 17: Analisis Spasial Klaster Flores
data_agu2026 = df_bersih[(df_bersih['tahun'] == 2026) & (df_bersih['bulan'] == 8)]
total_kejadian_agu2026 = len(data_agu2026)

# Filter koordinat sekitar Flores: Lintang [-9, -7] dan Bujur [120, 123]
gempa_flores = data_agu2026[
    (data_agu2026['lintang'] >= -9) & (data_agu2026['lintang'] <= -7) &
    (data_agu2026['bujur'] >= 120) & (data_agu2026['bujur'] <= 123)
]
jumlah_flores = len(gempa_flores)
persen_flores = (jumlah_flores / total_kejadian_agu2026) * 100

print(f"Total kejadian gempa dirasakan di Indonesia pada Agustus 2026 : {total_kejadian_agu2026} kejadian")
print(f"Jumlah kejadian berpusat di sekitar Flores (Lintang -9 s.d. -7, Bujur 120 s.d. 123) : {jumlah_flores} kejadian")
print(f"Persentase kejadian klaster Flores terhadap total Agustus 2026 : {persen_flores:.2f}%")
```

### 9.3. Dokumentasi Output Lengkap
```text
Total kejadian gempa dirasakan di Indonesia pada Agustus 2026 : 186 kejadian
Jumlah kejadian berpusat di sekitar Flores (Lintang -9 s.d. -7, Bujur 120 s.d. 123) : 118 kejadian
Persentase kejadian klaster Flores terhadap total Agustus 2026 : 63.44%
```

### 9.4. Kesimpulan Spasial
Mayoritas kejadian gempa bumi nasional pada bulan Agustus 2026 (**63,44%** atau 118 dari 186 kejadian) terisolasi secara spasial pada satu kotak koordinat sempit di Flores. Ini membuktikan bahwa lonjakan batang merah pada bar chart bulan Agustus bukan merupakan fenomena peningkatan aktivitas seismik serentak di seluruh kepulauan Indonesia, melainkan efek lokal dari gempa utama Nagekeo beserta gempa-gempa susulannya.

---

## 10. Topik B5.3: Empat Kesimpulan Kritis Seismologi

Berikut adalah 4 butir pernyataan kesimpulan esai resmi sebagaimana tertera pada notebook:

1. **Pemicu Lonjakan Aktivitas Agustus 2026:**
   Lonjakan aktivitas gempa pada Agustus 2026 yang mencapai **186 kejadian** dipicu oleh gempa utama berkekuatan **M7,0** di timur laut Mbay, Nagekeo pada 15 Agustus 2026, di mana rangkaian gempa susulannya menyumbang **118 kejadian (63,44%)** dari seluruh kejadian gempa dirasakan di Indonesia pada bulan tersebut.
2. **Penyebab Perbedaan Magnitudo Nagekeo (M7,0 vs M7,7 di Berita):**
   Magnitudo 7,0 pada dataset merupakan nilai estimasi awal (*preliminary parameter*) yang dirilis otomatis oleh sistem BMKG dalam beberapa menit pertama pascakejadian untuk kebutuhan peringatan dini tsunami, yang kemudian dimutakhirkan (*updated*) menjadi 7,7 di pemberitaan setelah seismolog BMKG menganalisis rekaman gelombang seismik secara lengkap dan komprehensif dari seluruh stasiun pemantau.
3. **Penyebab Selisih Median Magnitudo Darat (3,4) vs Laut (4,5):**
   Gempa laut memiliki median magnitudo yang lebih tinggi ($M = 4,5$) dibandingkan gempa darat ($M = 3,4$) karena jarak episenter di laut relatif jauh dari daratan dan permukiman warga, sehingga memerlukan pelepasan energi seismik yang lebih besar agar getarannya dapat dirasakan oleh manusia di pesisir, sementara gempa darat yang berpusat dangkal di dekat permukiman sudah dapat dirasakan meskipun berkekuatan kecil.
4. **Keterbatasan Dataset dan Peringatan Korelasi (*Correlation vs Causation*):**
   Keterbatasan dataset ini adalah hanya mencatat **3.618 gempa yang dirasakan masyarakat** (bukan sensus seluruh aktivitas seismik instrumen) serta rentang waktu tahun 2022 (mulai 14 Desember, 36 kejadian) dan 2026 (hingga 13 Oktober, 927 kejadian) yang tidak utuh satu tahun kalender; selain itu, korelasi positif sedang antara magnitudo dan kedalaman ($r = 0,37$) tidak boleh ditafsirkan sebagai hubungan kausalitas (kedalaman bukan penyebab gempa membesar), melainkan mencerminkan bias observasi bahwa gempa berhiposenter dalam hanya mampu dirasakan di permukaan apabila magnitudonya besar.

---

## 11. Tanya Jawab Kritis & Panduan Mempertahankan Jawaban UTS

Dokumentasi ini menyajikan rangkuman jawaban untuk potensi pertanyaan evaluasi dari dosen penguji:

### Q1: Mengapa pada Bagian B2 baris dengan bujur 1.39° BT dihapus dan tidak dikoreksi saja menjadi 140.39° BT?
> **Jawaban Akademik:**  
> Dalam etika rekayasa data ilmiah (*data governance*), melakukan koreksi nilai tanpa dokumen sumber resmi (*source of truth*) berisiko memasukkan asumsi subjektif peneliti (*data fabrication risk*). Meskipun deskripsi teks menyebutkan Jayapura, menghapus satu baris (dari total 3.619 baris, atau hanya 0,027% data) adalah tindakan metodologis yang paling konservatif dan terstandarisasi untuk memastikan integritas dataset spasial tetap terjaga 100%.

### Q2: Mengapa median kedalaman (10 km) lebih representatif daripada mean kedalaman (20,04 km)?
> **Jawaban Akademik:**  
> Rata-rata (*mean*) sangat rentan terdistorsi oleh nilai ekstrim (*skewed by outliers*). Dataset ini memiliki sejumlah kecil gempa dalam hingga $636\text{ km}$, yang menarik nilai mean ke atas ($20,04\text{ km}$). Sebaliknya, median sebesar $10,00\text{ km}$ menunjukkan bahwa lebih dari 50% gempa yang dirasakan di Indonesia terjadi pada kedalaman ultra-dangkal kerak bumi ($< 15\text{ km}$), yang merupakan batas kontak sesar aktif lokal.

### Q3: Apa signifikansi penggunaan `reindex(range(1, 11))` pada pengelompokan bulanan tahun 2026?
> **Jawaban Akademik:**  
> Jika pada bulan tertentu tidak ada gempa sama sekali, fungsi `groupby('bulan')` Pandas secara *default* akan melewatkan (*skip*) bulan tersebut dari indeks. Penggunaan `reindex(range(1, 11), fill_value=0)` menjamin bahwa seluruh bulan dari Januari (1) hingga Oktober (10) tetap memiliki slot batang pada grafik bar chart Matplotlib dengan nilai 0, sehingga tidak ada bulan yang hilang pada skala waktu visual.

### Q4: Mengapa korelasi koordinat spasial (lintang dan bujur) terhadap magnitudo mendekati angka nol?
> **Jawaban Akademik:**  
> Koefisien korelasi Pearson hanya mengukur hubungan linear sederhana ($y = mx + c$). Gempa bumi di Indonesia tidak terdistribusi secara linear terhadap koordinat bujur atau lintang, melainkan mengikuti batas-batas lempeng tektonik yang melengkung secara non-linear (Busur Sunda berbentuk kurva busur kepulauan). Oleh karena itu, uji korelasi linear bernilai $r \approx 0$, yang menegaskan bahwa analisis geospasial memerlukan pemetaan spasial 2D (seperti scatter plot panel 3) dan bukan sekadar analisis korelasi bivariat sederhana.

---
*Dokumen ini dibuat otomatis sebagai panduan komprehensif pembelajaran dan pertahanan teknis UTS Mata Kuliah Visualisasi Data.*
