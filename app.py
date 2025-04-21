from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    file = request.files['file']
    if not file:
        return "No file uploaded.", 400

    k = int(request.form.get("clusters", 4))

    df = pd.read_csv(file)

    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    snapshot_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Revenue']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    output_csv = BytesIO()
    rfm.to_csv(output_csv)
    output_csv.seek(0)
    encoded_csv = base64.b64encode(output_csv.read()).decode('utf-8')

    fig1 = px.histogram(rfm, x='Recency', nbins=10, title='Histogram of Recency', color_discrete_sequence=['dodgerblue'])
    fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    fig2 = px.scatter(rfm, x='Recency', y='Revenue', color=rfm['Cluster'].astype(str), title='Scatter plot of scores for Recency')
    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    corr_matrix = rfm.drop('Cluster', axis=1).corr().round(2)
    fig3 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Blues',
        showscale=True
    ))
    fig3.update_layout(title='Feature Correlation Heatmap', margin=dict(l=20, r=20, t=40, b=20))

    pca = PCA(n_components=3)
    pca.fit(rfm_scaled)
    importance = pd.Series(pca.components_[0], index=rfm.columns[:-1])
    fig4 = px.bar(importance.sort_values(), orientation='h', title='Feature Importance', color=importance.sort_values(), color_continuous_scale='Viridis')
    fig4.update_layout(margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)

    return render_template(
        'result.html',
        img1=fig1.to_json(),
        img2=fig2.to_json(),
        img3=fig3.to_json(),
        img4=fig4.to_json(),
        csv_data=encoded_csv
    )

if __name__ == '__main__':
    app.run(debug=True)
