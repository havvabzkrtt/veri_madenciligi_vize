import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)


# NORMAL VERİ SETİ GÖRÜNÜŞÜ
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S_points[:, 0], S_points[:, 1], S_points[:, 2], c=S_color, cmap=plt.cm.Spectral)

# Grafik ayarları
ax.set_title("3D Scatter Plot - Orijinal Veri Seti")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


original_dimensions = S_points.shape[1]
print(f"Orijinal veri setinin boyutu: {original_dimensions}")  

# Veriyi düzenleme
S_points_centered = S_points - np.mean(S_points, axis=0)

# Kovaryans matrisini hesapla
cov_matrix = np.cov(S_points_centered, rowvar=False)

# Özdeğer ve özvektörleri hesapla
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Özvektörleri büyükten küçüğe sırala
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# İlk iki özvektörü seç ve yeni veri setini oluştur
num_components = 2
S_reduced = np.dot(S_points_centered, eigenvectors_sorted[:, :num_components])

# Sonuçları görselleştir
plt.scatter(S_reduced[:, 0], S_reduced[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title('PCA ile İndirgenmiş 2D Veri Seti')
plt.show()




# İki çıktıyı karşılaitır.

"""
# 3 boyut 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

# Veriyi düzenleme
S_points_centered = S_points - np.mean(S_points, axis=0)

# Kovaryans matrisini hesapla
cov_matrix = np.cov(S_points_centered, rowvar=False)

# Özdeğer ve özvektörleri hesapla
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Özvektörleri büyükten küçüğe sırala
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# İlk iki özvektörü seç ve yeni veri setini oluştur
num_components = 3
S_reduced = np.dot(S_points_centered, eigenvectors_sorted[:, :num_components])


# Sonuçları görselleştir
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S_reduced[:, 0], S_reduced[:, 1], S_reduced[:, 2], c=S_color, cmap=plt.cm.Spectral)
ax.set_title('Temel Bileşen Analizi ile İndirgenmiş 3D Veri Seti')
plt.show()

"""