Üretilen 1000 örneği k-means kümeleme yöntemi ile kümeleyiniz.

K-Means Kümeleme, veri noktalarını birbirlerine benzerliklerine göre gruplandıran bir kümeleme algoritmasıdır. Veri noktaları arasındaki benzerlik, genellikle öklidyen uzaklık ölçüsü kullanılarak hesaplanır. Bu algoritma, veri setini belirli sayıda (k adet) küme veya grup içinde sınıflandırmaya çalışır.

K-means algoritması, başlangıçta rastgele seçilen küme merkezleri üzerinden kümeleme yapar. Bu nedenle, algoritma her çalıştığında farklı bir başlangıç durumu ile başlar ve bu da farklı kümeleme sonuçlarına yol açabilir.

## Kullanılan Kütüphaneler
```markdown

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

## Kümeleme Öncesi Veri Görselleştirme
```python
plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
plt.title('Kümeleme Öncesi Veri Seti')
plt.show()
```

## K-Means Kümeleme Modeli Oluşturma
```python
kmeans = KMeans(n_clusters=4)
```

## Modelin Eğitilmesi
```python
kmeans.fit(X)
```

## Küme Merkezleri ve Etiketleri Elde Etme
```python
centers = kmeans.cluster_centers_
labels = kmeans.labels_
```

## Kümelenmiş Veri Görselleştirme
```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
plt.title('K-Means Kümeleme Sonrası Veri Seti')
plt.legend()
plt.show()
```

## Görselleştirmeler
- Kümeleme Öncesi:
- ![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/b5c11cab-7448-4acd-b19e-35229fa6849e)


- Kümeleme Sonrası:
- ![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/59703698-08c5-41b0-af0a-e0eb6c0d4acf)

```

Bu Markdown kodu, belirli bir kümeleme öncesi ve sonrası veri setini ve sonuçları gösterir. 
