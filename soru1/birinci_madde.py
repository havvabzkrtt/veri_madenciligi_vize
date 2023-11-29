from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])

# Veriyi görselleştir (kümeleme öncesi)
plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
plt.title('K-means Kümeleme Öncesi')
plt.show()

# K-means modelini oluştur
kmeans = KMeans(n_clusters=4)

# Modeli eğit
kmeans.fit(X)

# Küme merkezlerini ve etiketleri al
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_


# Veriyi ve küme merkezlerini görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-means Kümeleme Sonrası')
plt.legend()
plt.show()


