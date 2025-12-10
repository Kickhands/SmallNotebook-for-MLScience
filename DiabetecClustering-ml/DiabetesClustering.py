from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Tugas Besar Kelompok")
st.header("Anggota Kelompok")
st.markdown("""
1. Gigih Haidar Falah  
2. Abyan Setyanvea  
3. David Christian Batubara  
4. Rizqi Wildan
""")

# Import Dataset
st.header("Import Dataset")
dataset_path = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(dataset_path)
st.dataframe(df.head(20))

# Dataset Summary
st.subheader("Dataset Summary")
st.write("Shape of the dataset:", df.shape)
st.write("Info:")
st.text(df.info())
st.write("Null values per column:")
st.dataframe(df.isnull().sum())

# Histogram Tabs
st.header("Histogram Per Feature")
hist_tabs = st.tabs(list(df.columns))
for i, col in enumerate(df.columns):
    with hist_tabs[i]:
        st.subheader(f"Histogram: {col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].hist(bins=20, color='red', ax=ax)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

# Boxplot Tabs
st.header("Boxplot Per Feature")
box_tabs = st.tabs(list(df.columns))
for i, col in enumerate(df.columns):
    with box_tabs[i]:
        st.subheader(f"Boxplot: {col}")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].plot(kind='box', ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

# Correlation Heatmaps
st.subheader("Correlation Heatmaps")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_features].corr()
filtered_correlation_matrix = correlation_matrix.where((abs(correlation_matrix) < 0.5) & (correlation_matrix != 1))

heatmap_tabs = st.tabs(["Unfiltered Heatmap", "Filtered Heatmap"])
with heatmap_tabs[0]:
    st.subheader("Unfiltered Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, square=True)
    ax.set_title("Unfiltered Heatmap")
    st.pyplot(fig)
with heatmap_tabs[1]:
    st.subheader("Filtered Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(filtered_correlation_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, square=True,
                mask=filtered_correlation_matrix.isnull())
    ax.set_title("Filtered Heatmap (|Correlation| < 0.5)")
    st.pyplot(fig)

# Data Standardization
st.subheader("Standardized Data")
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
normalized_data = scaler.fit_transform(df[numeric_columns])
st.write("Data has been standardized.")
st.dataframe(pd.DataFrame(normalized_data, columns=numeric_columns))

# KMeans Clustering
st.header("KMeans Clustering")

# Number of Clusters Selection (based on elbow method)
st.subheader("Elbow Method to Find Optimal Number of Clusters")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto', init='k-means++', max_iter=1000, random_state=42)
    kmeans.fit(normalized_data)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS (Within-cluster sum of squares)")
st.pyplot(fig)

# Choose optimal clusters and build KMeans model
n_clusters = 5  # Adjust based on elbow method
model = KMeans(n_clusters=n_clusters, n_init='auto', init='k-means++', max_iter=1000, random_state=42)
df['Cluster'] = model.fit_predict(normalized_data)

# Display Cluster Labels and Centroids
st.subheader("Cluster Labels and Centroids")
st.write("Cluster Labels:")
st.dataframe(df[['Cluster']].head(10))
st.write("Cluster Centroids:")
st.dataframe(pd.DataFrame(model.cluster_centers_, columns=numeric_columns))

# Silhouette Score
st.subheader("Silhouette Score")
silhouette_avg = silhouette_score(normalized_data, model.labels_)
st.write(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")

# Silhouette Scores for Different Numbers of Clusters
st.subheader("Silhouette Scores for Different Numbers of Clusters")
silhouette_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init='auto', init='k-means++', max_iter=1000, random_state=42)
    km.fit(normalized_data)
    silhouette_scores[k] = silhouette_score(normalized_data, km.labels_)

fig, ax = plt.subplots(figsize=(6, 4))
sns.pointplot(x=list(silhouette_scores.keys()), y=list(silhouette_scores.values()), ax=ax)
ax.set_title("Silhouette Scores for Different Numbers of Clusters")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")
st.pyplot(fig)

# PCA for 2D Visualization
st.subheader("2D Visualization of Clusters")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(normalized_data)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
ax.set_title("PCA Visualization of Clusters")
st.pyplot(fig)
