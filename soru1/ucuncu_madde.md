ikinci_maddede.py modelin yeni bir noktayı sınıflandırma yeteneği aşağıdaki kodda yer almaktadır. 

```markdown
# 1.3 

## Kullanılan Kütüphaneler
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
```

## Veri Seti Oluşturma
```python
X, y = datasets.make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
```

## Silhouette skoru kullanarak En İyi Küme Sayısı Belirlenmesi ve Kümeleme
```python
best_cluster_size = 4
kmeans = KMeans(n_clusters=best_cluster_size)
kmeans.fit(X)
labels = kmeans.labels_
```

## Yeni Bir Noktanın Belirlenmesi
```python
new_point = np.array([[0, 5]])
```

## Noktanın Hangi Küme Merkezine Daha Yakın Olduğunun Belirlenmesi
```python
distance_to_centers = np.linalg.norm(kmeans.cluster_centers_ - new_point, axis=1)
predicted_class = np.argmin(distance_to_centers)
```

## Anomali Eşik Değerinin Belirlenmesi
```python
anomaly_threshold = 5
```

## Anomali veya Sınıf Sınıflandırması
```python
if np.min(distance_to_centers) > anomaly_threshold:
    print("Nokta bir anomali olarak sınıflandırıldı.")
else:
    print(f"Nokta {predicted_class} sınıfına aittir.")
```

## K-Means Kümeleme ve Görselleştirme (1.3.8)
```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(new_point[0, 0], new_point[0, 1], c='red', marker='X', s=200, label='Yeni Nokta')
plt.title('K-Means Kümeleme ve Yeni Nokta (Anomaly Eşik: 5)')
plt.legend()
plt.show()
```

## K-Means Kümeleme ve Görselleştirme (1.3.9)
```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(new_point[0, 0], new_point[0, 1], c='red', marker='X', s=200, label='Yeni Nokta (Anomali)')
plt.title('K-Means Kümeleme ve Yeni Nokta (Anomaly Eşik: 3)')
plt.legend()
plt.show()
```

Bu Markdown kodu, yeni bir noktanın belirlenmesi ve bu noktanın bir sınıfa veya bir anomali olarak sınıflandırılması süreçlerini açıklar. 
