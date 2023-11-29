# linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])

# Hiyerarşik kümeleme modelini oluştur

model = AgglomerativeClustering(n_clusters=4, linkage='ward')  # linkage parametresi burada kullanılıyor
y_pred = model.fit_predict(X)
model1 = AgglomerativeClustering(n_clusters=4, linkage='single')
y_pred1 = model1.fit_predict(X)
model2 = AgglomerativeClustering(n_clusters=4, linkage='complete')
y_pred2 = model2.fit_predict(X)
model3 = AgglomerativeClustering(n_clusters=4, linkage='average')
y_pred3 = model3.fit_predict(X)
# complete ve single dene ward daha iyi sanki? görselleri karşılaştır 
# Modeli eğit


# Kümeleme sonuçlarını görselleştir
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
plt.title('Ward Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

# Kümeleme sonuçlarını görselleştir
plt.scatter(X[:, 0], X[:, 1], c=y_pred1, cmap='viridis', s=20, alpha=0.7)
plt.title('Single Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

# Kümeleme sonuçlarını görselleştir
plt.scatter(X[:, 0], X[:, 1], c=y_pred2, cmap='viridis', s=20, alpha=0.7)
plt.title('Complete Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

# Kümeleme sonuçlarını görselleştir
plt.scatter(X[:, 0], X[:, 1], c=y_pred3, cmap='viridis', s=20, alpha=0.7)
plt.title('Average Bağlantı ile Hiyerarşik Kümeleme')
plt.show()


# 1. soruyu eklee karşılaştırma için 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

y_pred = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
plt.title('K-means Kümeleme')
plt.show()

