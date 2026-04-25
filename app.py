import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Wajib untuk server agar tidak muncul error thread
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Fungsi pembantu untuk mengubah grafik jadi kode teks (Base64)
def get_plot_url():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def index():
    # 1. Load Data
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, 'netflix_titles.csv'))
    
    # 2. Preprocessing
    sub_df = df[['release_year', 'rating']].dropna()
    le = LabelEncoder()
    sub_df['rating_encoded'] = le.fit_transform(sub_df['rating'].astype(str))
    
    X = sub_df[['release_year', 'rating_encoded']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Proses Elbow Method (Mencari K Optimal)
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_m = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_m.fit(X_scaled)
        inertia.append(kmeans_m.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'ro-')
    plt.title('Metode Elbow')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Inertia')
    elbow_url = get_plot_url() # Variabel tercipta di sini
    plt.close()

    # 4. Clustering Utama (Gunakan K=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    sub_df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sub_df, x='release_year', y='rating_encoded', hue='Cluster', palette='viridis')
    plt.title('Hasil Clustering Netflix')
    plot_url = get_plot_url() # Variabel tercipta di sini
    plt.close()

    # 5. Siapkan Tabel
    df_result = df.merge(sub_df[['Cluster']], left_index=True, right_index=True)
    table_html = df_result.head(100).to_html(classes='table table-hover table-bordered', index=False)
    
    # 6. BARU TERAKHIR: Kirim semua variabel ke HTML
    return render_template('results.html', 
                           table=table_html, 
                           plot_url=plot_url, 
                           elbow_url=elbow_url)

if __name__ == '__main__':
    app.run(debug=True)