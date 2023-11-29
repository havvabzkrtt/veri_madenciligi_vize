from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])


silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_ 
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

best_k = np.argmax(silhouette_scores) + 2  # +2 çünkü küme sayısı 2'den başlıyor


best_kmeans = KMeans(n_clusters=best_k)
best_kmeans.fit(X)

best_cluster_centers = best_kmeans.cluster_centers_
best_labels = best_kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=best_labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(best_cluster_centers[:, 0], best_cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('En İyi Küme Sayısı ile K-means Clustering')
plt.legend()
plt.show()

# Yeni bir nokta oluştur
new_point = np.array([[2, 5]])  # bu değişebilir, bunu değiştirerek sonuca bak

# Bu noktanın hangi küme merkezine daha yakın olduğunu belirle
distance_to_centers = np.linalg.norm(new_point - best_cluster_centers, axis=1) # diziye atanır öklidyen kullanılarak yeni nokta ile merkezler arasında hesaplanan uzaklıklar 
predicted_class = np.argmin(distance_to_centers)  # kümesi belirlenir

# Bir anomali eşik değeri belirle (örneğin, 5 birim uzaklık)
anomaly_threshold = 5  # bu değişebilir, bunu değiştirerek sonuca bak

if np.min(distance_to_centers) > anomaly_threshold:
    print("Nokta bir anomali olarak sınıflandırıldı.")
else:
    print(f"Nokta {predicted_class} numaralı sınıfa ait.")

# Bir anomali eşik değeri belirle (örneğin, 8 birim uzaklık)
anomaly_threshold2 = 3  # bu değişebilir, bunu değiştirerek sonuca bak

if np.min(distance_to_centers) > anomaly_threshold2:
    print("Nokta bir anomali olarak sınıflandırıldı.")
else:
    print(f"Nokta {predicted_class} numaralı sınıfa ait.")
    
"""
Anomaly threshold (anomali eşik değeri), bir veri noktasının anomali olarak sınıflandırılması için belirlenen bir sınırdır.
Bu eşik değeri seçerken, veri setinizin özelliklerine, problem bağlamına ve tolerans düzeyinize bağlı olarak değişen faktörlere dikkat etmek önemlidir. 
"""