Optimal Küme Sayısının Belirlenmesi

Bu soru için en optimal küme sayısını bulmada kullanılan “Elbow yöntemi”, “Silhouette Skoru”, “Gap İstatistiği”, “Davies-Bouldin İndeksi” gibi farklı metrikler bulunmaktadır. 
Bu programda en optimal kümenin bulunması için “Silhouette Skoru” kullanıldı. 

“Silhouette Skoru” kümeleme sonuçlarının başarısını ölçen bir yöntemdir. Küme içindeki bir verinin, kendi kümesine benzerliğinin ve en yakın olan kümeye benzerliğinin farkını ölçer. En yüksek Silhouette skoru en optimal küme sayısını temsil eder. 

## Kullanılan Kütüphaneler
```markdown
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
```

## Veri Seti Oluşturma
```python
X, y = datasets.make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
```

## Silhouette Skorları İçin Dizi Oluşturma
```python
silhouette_scores = []
```

## Silhouette Skorları Elde Etme
```python
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)
```

## Silhouette Skorları Dizisi Görselleştirme ve Küme Sayısı Belirleme
```python
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Skorları')
plt.xlabel('Küme Sayısı')
plt.ylabel('Silhouette Skoru')
plt.show()

best_cluster_index = np.argmax(silhouette_scores)
best_cluster_size = best_cluster_index + 2
print(f"En iyi küme sayısı: {best_cluster_size}")
```

## En İyi Küme Sayısını Belirleme
```python
best_cluster_index = np.argmax(silhouette_scores)
best_cluster_size = best_cluster_index + 2
print(f"En iyi küme sayısı: {best_cluster_size}")
```

## K-Means Kümeleme ve Görselleştirme
```python
# En iyi küme sayısına göre K-means modelini eğit ve görselleştir
kmeans = KMeans(n_clusters=best_cluster_size)
kmeans.fit(X)
labels = kmeans.labels_

# Veriyi görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.title(f'K-Means Kümeleme (En İyi Küme Sayısı: {best_cluster_size})')
plt.show()
```

## Görselleştirmeler
- Silhouette Skorları ve En İyi Küme Sayısı:
- ![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/7efd6396-e9b9-4b54-a2f9-48dc23159862)


- En İyi Küme Sayısına Göre K-Means Kümeleme:
- ![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/6aa964ce-dfe1-4c90-a516-882765cf8d6b)

```

Bu Markdown kodu, Silhouette skorlarını kullanarak en iyi küme sayısını belirleyen ve bu küme sayısıyla K-Means kümeleme uygulayan bir programın açıklamalarını içerir. 
