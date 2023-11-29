import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import LocallyLinearEmbedding
from scipy.spatial.distance import cdist

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

# LLE ile boyut indirgeme
lle = LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=42)
S_points_2D = lle.fit_transform(S_points)

# 2D S eğrisini çiz
plt.figure(figsize=(8, 8))
plt.scatter(S_points_2D[:, 0], S_points_2D[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title('LLE ile Boyut İndirgenmiş 2D Veri Seti')
plt.show()

# En uzak iki noktayı bul
distances = cdist(S_points_2D, S_points_2D)
max_distance_idx = np.unravel_index(np.argmax(distances), distances.shape)
max_point1, max_point2 = S_points_2D[max_distance_idx[0]], S_points_2D[max_distance_idx[1]]

print("En uzak iki nokta:")
print("Nokta 1:", max_point1)
print("Nokta 2:", max_point2)
print("Uzaklık:", np.linalg.norm(max_point1 - max_point2))

# Yeni bir noktanın konumunu belirle
new_point = np.array([[0.5, 0.5]])  # 2D uzayda örnek bir nokta
distance_to_curve = np.min(np.linalg.norm(S_points_2D - new_point, axis=1))

if distance_to_curve < 0.1:  # Örnek bir eşik değeri
    print("Yeni nokta, indirgenmiş S eğrisi üzerinde.")
else:
    print("Yeni nokta, indirgenmiş S eğrisi dışında.")
