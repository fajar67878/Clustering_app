import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
# Pastikan file netflix_titles.csv berada di folder yang sama dengan skrip ini
file_path = 'netflix_titles.csv'
if not os.path.exists(file_path):
    print(f"Error: File {file_path} tidak ditemukan!")
else:
    df = pd.read_csv(file_path)
    print("Data berhasil dimuat.")

    # 2. Preprocessing
    # Kita ambil kolom release_year dan rating (seperti di modul)
    sub_df = df[['release_year', 'rating']].dropna()

    # Mengubah teks rating menjadi angka agar bisa dihitung
    le = LabelEncoder()
    sub_df['rating_encoded'] = le.fit_transform(sub_df['rating'].astype(str))

    # 3. Clustering K-Means
    X = sub_df[['release_year', 'rating_encoded']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menentukan jumlah cluster (K=3)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    sub_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 4. Simpan Hasil ke CSV Baru
    # Menggabungkan hasil cluster ke data asli
    df_final = df.merge(sub_df[['Cluster']], left_index=True, right_index=True)
    df_final.to_csv('hasil_clustering_netflix.csv', index=False)
    print("Hasil clustering berhasil disimpan ke 'hasil_clustering_netflix.csv'")

    # 5. Visualisasi
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=sub_df, x='release_year', y='rating_encoded', hue='Cluster', palette='viridis')
    plt.title(f'Segmentasi Konten Netflix (K={k})')
    plt.xlabel('Tahun Rilis')
    plt.ylabel('Rating (Encoded)')
    
    # Simpan Gambar
    plt.savefig('grafik_clustering.png')
    print("Grafik berhasil disimpan ke 'grafik_clustering.png'")
    
    # Tampilkan Grafik di VS Code (Jika menggunakan ekstensi Jupyter/Python)
    plt.show()