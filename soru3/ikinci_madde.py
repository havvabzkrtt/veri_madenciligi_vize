from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

# Veri setini oluştur
from sklearn import datasets
import numpy as np
X, y = datasets.make_moons(n_samples=1000, noise=0.05,
random_state=np.random.randint(80))

def plot_dendrogram(model, **kwargs):
    # İlgili bağlantı matrisini al
    linkage_matrix = linkage(X, model.linkage, metric='euclidean')
    
    # Dendrogramı çiz
    dendrogram(linkage_matrix, **kwargs)

# Hiyerarşik kümeleme modelini oluştur
model = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=0)

# Modeli eğit ve dendrogramı çiz
plot_dendrogram(model)
plt.title('Dendrogram')
plt.xlabel('Veri Noktaları')
plt.ylabel('Uzaklık')
plt.show()

# Dendrogramdaki kesilme noktalarını belirle
distance_threshold = 30  # Uyarı: Bu değeri görsel olarak inceleyerek belirlemeniz gerekebilir.

# Hiyerarşik kümeleme modelini oluştur ve eğit
model = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=distance_threshold)
y_pred = model.fit_predict(X)

# Optimal küme sayısını ekrana yazdır
num_clusters = len(set(y_pred))
print(f"Optimal Küme Sayısı: {num_clusters}")

# Kümeleme sonuçlarını görselleştir
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
plt.title('Hiyerarşik Clustering')
plt.show()

# Optimal Küme Sayısı: 2 doğru çünkü bu şekilde kümelemeişti birinci_madde de de  
