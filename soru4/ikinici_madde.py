# ÖKLİD
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

# Noktalar arasındaki uzaklıkları hesapla (iç içe döngülerle)
num_points = len(S_points)
distances = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        distances[i, j] = np.sqrt(np.sum((S_points[i] - S_points[j]) ** 2))

print(distances.shape) # (1500,1500)

# distances matrisinin görselleştirilmesi
# Belirli bir bölümü alarak görselleştirme
subset_size = 50
subset_distances = distances[:subset_size, :subset_size]

plt.figure(figsize=(10, 8))
sns.heatmap(subset_distances, cmap='viridis', annot=False)
plt.title('Öklidyen Uzaklık Matrisi')
plt.show()

# PCA UYGULANMASI 
# Temel bileşen analizi için hazırlık
# Uzaklık matrisinin merkezini sıfıra çek
distances_centered = distances - np.mean(distances, axis=0)

cov_matrix = np.cov(distances_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

num_components = 2
distances_reduced = np.dot(distances_centered, eigenvectors_sorted[:, :num_components])

plt.scatter(distances_reduced[:, 0], distances_reduced[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title('PCA ile İndirgenmiş 2D Uzaklık Veri Seti')
plt.show()




# MANHATTAN
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

# Noktalar arasındaki Manhattan uzaklıkları hesapla (iç içe döngülerle)
num_points = len(S_points)
distances_manhattan = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        #distances_manhattan[i, j] = np.sum(np.abs(S_points[i] - S_points[j])) # MANHATTAN
        distances_manhattan[i, j] = np.sqrt(np.sum((S_points[i] - S_points[j]) ** 2))  # ÖKLİD

# Temel bileşen analizi için hazırlık
# Uzaklık matrisinin merkezini sıfıra çek
distances_manhattan_centered = distances_manhattan - np.mean(distances_manhattan, axis=0)

# Kovaryans matrisini hesapla
cov_matrix_manhattan = np.cov(distances_manhattan_centered, rowvar=False)

# Özdeğer ve özvektörleri hesapla
eigenvalues_manhattan, eigenvectors_manhattan = np.linalg.eigh(cov_matrix_manhattan)

# Özvektörleri büyükten küçüğe sırala
sorted_indices_manhattan = np.argsort(eigenvalues_manhattan)[::-1]
eigenvectors_sorted_manhattan = eigenvectors_manhattan[:, sorted_indices_manhattan]

# İlk iki özvektörü seç ve yeni veri setini oluştur
num_components_manhattan = 2
distances_reduced_manhattan = np.dot(distances_manhattan_centered, eigenvectors_sorted_manhattan[:, :num_components_manhattan])

# Sonuçları görselleştir
plt.scatter(distances_reduced_manhattan[:, 0], distances_reduced_manhattan[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title('PCA ile İndirgenmiş 2D Manhattan Uzaklık Veri Seti (NumPy)')
plt.show()
"""