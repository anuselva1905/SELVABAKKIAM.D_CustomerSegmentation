
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_segmentation(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Data cleaning
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[df["InvoiceNo"].str.match("^\\d{6}$") == True]
    df["StockCode"] = df["StockCode"].astype(str)
    df = df[df["CustomerID"].notna()]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Feature engineering - create RFM features (example)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (x.max() - x.min()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalPrice": "Monetary"
    })

    # Standardization
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm)

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(scaled_rfm)

    return rfm.reset_index()
