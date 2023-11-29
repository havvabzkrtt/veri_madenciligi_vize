# yakın komşular parametresini az veya çok seçtiğimizde sonuçların nasıl değiştiğiniz gösteriniz ve yorumlayınız.
from sklearn import datasets, manifold
import matplotlib.pyplot as plt

# Veri setini oluştur
n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)

## 12 Komşu ile LLE uygula
lle1 = manifold.LocallyLinearEmbedding(n_neighbors=12, n_components=2, method='standard')
X_lle1 = lle1.fit_transform(S_points)

# Sonuçları görselleştir
plt.scatter(X_lle1[:, 0], X_lle1[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title("12 Komşlu LLE ile 2 Boyuta İndirgenmiş Veri Seti")
plt.show()


## 3 Komşu ile LLE uygula
lle2 = manifold.LocallyLinearEmbedding(n_neighbors=3, n_components=2, method='standard')
X_lle2 = lle2.fit_transform(S_points)

# Sonuçları görselleştir
plt.scatter(X_lle2[:, 0], X_lle2[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title("3 Komşulu LLE ile 2 Boyuta İndirgenmiş Veri Seti")
plt.show()


## 100 Komşu ile LLE uygula
lle3 = manifold.LocallyLinearEmbedding(n_neighbors=100, n_components=2, method='standard')
X_lle3 = lle3.fit_transform(S_points)

# Sonuçları görselleştir
plt.scatter(X_lle3[:, 0], X_lle3[:, 1], c=S_color, cmap=plt.cm.Spectral)
plt.title("100 Komşulu LLE ile 2 Boyuta İndirgenmiş Veri Seti")
plt.show()
