import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dan siapkan data
df = pd.read_csv('Mall_Customers.csv')
df.rename(columns={'Annual Income (k$)': 'Income',
                   'Spending Score (1-100)': 'Score'},
          inplace=True)
x = df.drop(['CustomerID', 'Gender'], axis=1)

# Tampilkan data
st.header("Isi Dataset")
st.write(df)

# Hitung inertia untuk berbagai jumlah cluster
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(x)
    clusters.append(km.inertia_)

# Buat plot Elbow
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, marker='o', ax=ax)
ax.set_title('Mencari Elbow Point')
ax.set_xlabel('Jumlah Cluster (k)')
ax.set_ylabel('Inertia')
ax.grid(True)

# Tambahkan anotasi elbow
ax.annotate('Possible elbow point',
            xy=(3, clusters[2]), xytext=(4, clusters[2] + 50000),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2),
            fontsize=12, color='blue')
ax.annotate('Possible elbow point',
            xy=(5, clusters[4]), xytext=(6, clusters[4] + 50000),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2),
            fontsize=12, color='blue')

# Tampilkan plot di Streamlit
st.pyplot(fig)

st.sidebar.subheader("Nilai jumlah k")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2, 10, 3, 1)

# Fungsi k-means dan visualisasi
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust, random_state=42)
    x['Labels'] = kmean.fit_predict(x)

    # Buat scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=x, x='Income', y='Score',
        hue='Labels',
        palette=sns.color_palette('hls', n_clust),
        legend='full', ax=ax
    )

    # Anotasi centroid tiap cluster
    for label in x['Labels'].unique():
        income_mean = x[x['Labels'] == label]['Income'].mean()
        score_mean = x[x['Labels'] == label]['Score'].mean()
        ax.annotate(str(label),
                    xy=(income_mean, score_mean),
                    ha='center', va='center',
                    size=20, weight='bold',
                    color='black')

    # Tampilkan hasil di Streamlit
    st.header('Cluster Plot')
    st.pyplot(fig)
    st.write("Hasil Labeling Cluster:")
    st.dataframe(x)

# Jalankan fungsi dengan jumlah cluster dari slider
k_means(clust)     

