from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import uuid
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # Clean and prepare data
        df = df.dropna(subset=['CustomerID'])
        df = df[df['Quantity'] > 0]
        df = df[df['InvoiceNo'].astype(str).str.match(r'^[0-9]{6}$')]

        # Aggregate
        df_grouped = df.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'Quantity': 'sum',
            'UnitPrice': 'mean'
        }).reset_index()

        # Normalize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_grouped.iloc[:, 1:])

        # Elbow method
        distortions = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled)
            distortions.append(kmeans.inertia_)
        elbow_path = os.path.join(PLOT_FOLDER, f'elbow_{uuid.uuid4()}.png')
        plt.figure()
        plt.plot(range(1, 10), distortions, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.savefig(elbow_path)
        plt.close()

        # PCA for 2D and 3D
        pca_2d = PCA(n_components=2)
        reduced_2d = pca_2d.fit_transform(scaled)

        pca_3d = PCA(n_components=3)
        reduced_3d = pca_3d.fit_transform(scaled)

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df_grouped['Cluster'] = kmeans.fit_predict(scaled)

        # 2D Cluster Plot
        cluster_path = os.path.join(PLOT_FOLDER, f'cluster_{uuid.uuid4()}.png')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_2d[:, 0], y=reduced_2d[:, 1], hue=df_grouped['Cluster'], palette='Set2')
        plt.title('Customer Segments (2D PCA)')
        plt.savefig(cluster_path)
        plt.close()

        # 3D Cluster Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], c=df_grouped['Cluster'], cmap='Set2')
        ax.set_title('Customer Segments (3D PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plot_3d_path = os.path.join(PLOT_FOLDER, f'3dplot_{uuid.uuid4()}.png')
        plt.savefig(plot_3d_path)
        plt.close()

        # Violin Plot
        violin_path = os.path.join(PLOT_FOLDER, f'violin_{uuid.uuid4()}.png')
        plt.figure(figsize=(10, 6))
        melted = pd.melt(df_grouped, id_vars='Cluster', value_vars=['InvoiceNo', 'Quantity', 'UnitPrice'])
        sns.violinplot(x='variable', y='value', hue='Cluster', data=melted, palette='Set2', split=True)
        plt.title('Distribution of Features by Cluster')
        plt.savefig(violin_path)
        plt.close()

        # Silhouette Score
        silhouette = silhouette_score(scaled, df_grouped['Cluster'])

        return render_template('result.html', 
                               image_file=cluster_path,
                               elbow_plot=elbow_path,
                               plot3d=plot_3d_path,
                               violin_plot=violin_path,
                               silhouette_score=round(silhouette, 3))

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
