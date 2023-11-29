
# Silhouette 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])

# Silhouette skorlarını kaydet
silhouette_scores = []

# Farklı küme sayıları için K-means modellerini eğit ve skorları kaydet
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Silhouette skorlarını görselleştir
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('Silhouette Skoru')
plt.title('Silhouette Skoru ile Küme Sayısı Belirleme')
plt.show()

# En iyi skoru veren küme sayısını bul
best_k = np.argmax(silhouette_scores) + 2  # +2 çünkü küme sayısı 2'den başlıyor
print("En iyi küme sayısı (k):", best_k)

# En iyi küme sayısı ile K-means modelini eğit
best_kmeans = KMeans(n_clusters=best_k)
best_kmeans.fit(X)

best_cluster_centers = best_kmeans.cluster_centers_
best_labels = best_kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(best_cluster_centers[:, 0], best_cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('En İyi Küme Sayısı ile K-means Kümeleme')
plt.legend()
plt.show()


# elbow 
"""
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])

# Farklı küme sayıları için toplam kare hata (inertia) değerlerini kaydet
inertia_values = []

# Farklı küme sayıları için K-means modellerini eğit ve inertiaları kaydet
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Inertia değerlerini görselleştir
plt.plot(range(1, 11), inertia_values, marker='o')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('İnertia (Toplam Kare Hata)')
plt.title('Elbow Method ile Küme Sayısı Belirleme')
plt.show()

# Elbow Method ile en iyi küme sayısını belirle
best_k = np.argmin(np.diff(inertia_values)) + 1

# En iyi küme sayısı ile K-means modelini eğit
best_kmeans = KMeans(n_clusters=best_k)
best_kmeans.fit(X)

# Küme merkezlerini ve etiketleri al
best_cluster_centers = best_kmeans.cluster_centers_
best_labels = best_kmeans.labels_

# Veriyi ve küme merkezlerini görselleştir
plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(best_cluster_centers[:, 0], best_cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('En İyi Küme Sayısı ile K-means Clustering')
plt.legend()
plt.show()

print("En iyi küme sayısı (k):", best_k)
"""