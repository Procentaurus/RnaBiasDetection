import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .params import ROOT_DIR


# -----------------------------
# 1. Load the normalized counts
# -----------------------------
df = pd.read_csv(
        ROOT_DIR / "data/counts_normalized.csv",
        sep=";",
        decimal=",",
        index_col=0
    )

df = df.apply(pd.to_numeric, errors='coerce')

# -----------------------------
# 2. Compute Spearman distance
# -----------------------------
corr = df.T.corr(method='spearman')  # correlation between genes
distance = 1 - corr                 # convert correlation to distance

# -----------------------------
# 3. Hierarchical clustering
# -----------------------------
linked = linkage(squareform(distance), method='average')  # average linkage

plt.figure(figsize=(12, 10))
dendrogram(linked, labels=df.index.tolist(), orientation='right', leaf_rotation=0)
plt.title('Hierarchical Clustering of Genes (Spearman)')
plt.xlabel('Distance (1 - Spearman correlation)')
plt.ylabel('Genes')
plt.tight_layout()
plt.savefig('hierarchical_clustering_dendrogram.png')  # save figure
plt.close()

# -----------------------------
# 4. K-Means clustering
# -----------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

k = 11
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

df['Cluster'] = clusters

# -----------------------------
# 5. Heatmap with clusters
# -----------------------------
df_sorted = df.sort_values('Cluster')
data_sorted = df_sorted.drop(columns='Cluster')

plt.figure(figsize=(14, 12))
sns.heatmap(data_sorted, cmap='vlag', yticklabels=True)
plt.title(f'Gene Expression Heatmap (K-Means, k={k})')
plt.xlabel('Samples')
plt.ylabel('Genes')
plt.tight_layout()
plt.savefig('kmeans_gene_heatmap.png')  # save heatmap
plt.close()

# -----------------------------
# 6. Save cluster assignments
# -----------------------------
df[['Cluster']].to_csv('gene_cluster_assignments.csv')
