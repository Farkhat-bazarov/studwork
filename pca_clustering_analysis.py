import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


print("=== 1. DATA PREPARATION ===")

np.random.seed(42)
n_samples = 200


f1 = np.random.normal(50, 10, n_samples)
f2 = f1 * 0.8 + np.random.normal(0, 3, n_samples)
f3 = np.random.normal(100, 25, n_samples)
f4 = f3 * -0.6 + np.random.normal(0, 5, n_samples)

df = pd.DataFrame({'Feature_1': f1, 'Feature_2': f2, 'Feature_3': f3, 'Feature_4': f4})


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)



print("\n=== 2. DIMENSIONALITY REDUCTION (PCA) ===")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print(f"Explained Variance (PC1): {explained_variance[0]:.4f}")
print(f"Explained Variance (PC2): {explained_variance[1]:.4f}")
print(f"Total Cumulative Variance Explained: {np.sum(explained_variance):.4f}")


print("\n=== 3. K-MEANS CLUSTERING & SILHOUETTE SCORE ===")


k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)


sil_score = silhouette_score(X_pca, cluster_labels)

df['Cluster'] = cluster_labels
print(f"Number of Clusters: {k}")
print(f"Silhouette Score: {sil_score:.4f}")
print("\nMean Feature Values per Cluster:")
print(df.groupby('Cluster').mean().round(2))
