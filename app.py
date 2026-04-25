import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # WAJIB: Gunakan backend non-interaktif
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Folder untuk menyimpan hasil grafik
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    # 1. Load Data
    csv_path = 'netflix_titles.csv'
    df = pd.read_csv(csv_path)
    
    # 2. Preprocessing
    # Memilih kolom dan membuang data kosong (NaN)
    sub_df = df[['release_year', 'rating']].dropna()
    
    # Encode rating (teks) menjadi angka
    le = LabelEncoder()
    sub_df['rating_encoded'] = le.fit_transform(sub_df['rating'].astype(str))
    
    # Scaling data agar seimbang
    X = sub_df[['release_year', 'rating_encoded']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Proses Elbow Method (Mencari K Optimal)
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(X_scaled)
        inertia.append(kmeans_model.inertia_)
    
    # Buat Grafik Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Inertia (Distortion)')
    plt.title('Metode Elbow untuk Menentukan K Optimal')
    plt.grid(True)
    
    elbow_path = os.path.join(app.config['UPLOAD_FOLDER'], 'elbow_plot.png')
    plt.savefig(elbow_path)
    plt.close()

    # 4. Clustering Utama (Kita gunakan K=3 sebagai contoh optimal)
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    sub_df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Buat Grafik Cluster
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sub_df, x='release_year', y='rating_encoded', hue='Cluster', palette='viridis', s=100)
    plt.title(f'Hasil Clustering Netflix (K={k_optimal})')
    
    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_netflix.png')
    plt.savefig(plot_path)
    plt.close()

    # 5. Siapkan Data untuk Tabel Web
    # Gabungkan hasil cluster ke dataframe asli
    df_result = df.merge(sub_df[['Cluster']], left_index=True, right_index=True)
    
    # Buat tabel HTML dengan class Bootstrap
    table_html = df_result.head(100).to_html(classes='table table-hover table-bordered align-middle', index=False)
    
    return render_template('results.html', 
                           table=table_html, 
                           plot_url='plot_netflix.png', 
                           elbow_url='elbow_plot.png')

if __name__ == '__main__':
    app.run(debug=True)