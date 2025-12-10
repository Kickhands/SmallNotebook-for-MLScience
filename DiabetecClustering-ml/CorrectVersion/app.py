import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Page title
st.title("Analisis Data Kesehatan dan Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Menampilkan informasi dasar dataset
    st.subheader("Informasi Dataset")
    st.write("Jumlah Baris dan Kolom:", df.shape)
    st.write("Informasi Dataset:")
    st.text(df.info())
    st.write("Cek Missing Values:")
    st.write(df.isnull().sum())
    st.write("Dataset Preview:")
    st.write(df.head())

    # Kolom kategorikal & numerikal
    threshold_categorical = 3
    unique_counts = df.nunique()
    categorical_cols = unique_counts[unique_counts <= threshold_categorical].index.tolist()
    numerical_cols = unique_counts[unique_counts > threshold_categorical].index.tolist()

    st.subheader("Identifikasi Kolom")
    st.write("Kolom Kategorikal:", categorical_cols)
    st.write("Kolom Numerikal:", numerical_cols)

    # Pisahkan kolom
    df_categorical = df[categorical_cols].drop(columns=['Diabetes_012'], errors='ignore')
    df_numerical = df[numerical_cols]

    # Visualisasi Histograms
    st.subheader("Histogram Kolom Kategorikal")
    st.write("Distribusi Data Kategorikal:")
    if not df_categorical.empty:
        df_categorical.hist(figsize=(12, 8), bins=20, color='red')
        st.pyplot(plt)

    st.subheader("Histogram Kolom Numerikal")
    st.write("Distribusi Data Numerikal:")
    if not df_numerical.empty:
        df_numerical.hist(figsize=(12, 8), bins=20, color='blue')
        st.pyplot(plt)

    # Heatmap Korelasi
    st.subheader("Heatmap Korelasi")
    correlation_matrix = df_numerical.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(plt)

    # Standardisasi data
    st.subheader("Standarisasi Data")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_numerical)
    normalized_df = pd.DataFrame(normalized_data, columns=df_numerical.columns)
    st.write(normalized_df.head())

    # PCA
    st.subheader("PCA untuk Reduksi Dimensi")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_data)
    st.write(f"Variansi yang dijelaskan: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # Visualisasi PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title('PCA: Komponen Utama 1 vs 2')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    st.pyplot(plt)

    # KMeans dan Elbow Method
    st.subheader("Optimal K: Elbow Method")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(pca_result)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    st.pyplot(plt)

    # Silhouette Score
    # st.subheader("Silhouette Score")
    # silhouette_scores = []
    # for k in range(2, 5):
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    #     kmeans.fit(pca_result)
    #     score = silhouette_score(pca_result, kmeans.labels_)
    #     silhouette_scores.append((k, score))

    # for k, score in silhouette_scores:
    #     st.write(f"Jumlah Klaster: {k}, Silhouette Score: {score:.4f}")

    # KMeans Final
    st.subheader("KMeans Clustering (3 Klaster)")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(pca_result)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('Clustering Result')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    st.pyplot(plt)

    st.write("Centroids:")
    st.write(centroids)
