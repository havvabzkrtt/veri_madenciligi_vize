Açıklamaları ve görselleştirmeleri içeren bu bilgilerin Markdown formatına dönüştürülmüş hali aşağıda verilmiştir:

```markdown
# 1.1 K-Means Kümeleme

## Kullanılan Kütüphaneler
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
```

## Veri Seti Oluşturma
```python
# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
```

## Kümeleme Öncesi Veri Görselleştirme
```python
# Kümeleme öncesi verileri görselleştir
plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
plt.title('Kümeleme Öncesi Veri Seti')
plt.show()
```

## K-Means Kümeleme Modeli Oluşturma
```python
# KMeans modelini oluştur
kmeans = KMeans(n_clusters=4)
```

## Modelin Eğitilmesi
```python
# Modeli eğit
kmeans.fit(X)
```

## Küme Merkezleri ve Etiketleri Elde Etme
```python
# Küme merkezlerini ve etiketleri al
centers = kmeans.cluster_centers_
labels = kmeans.labels_
```

## Kümelenmiş Veri Görselleştirme
```python
# Kümeleme sonrası verileri görselleştir
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
plt.title('K-Means Kümeleme Sonrası Veri Seti')
plt.legend()
plt.show()
```

## Görselleştirmeler
- Kümeleme Öncesi:
![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/b5c11cab-7448-4acd-b19e-35229fa6849e)


- Kümeleme Sonrası:
![image](https://github.com/havvabzkrtt/veri_madenciligi_vize/assets/81237002/59703698-08c5-41b0-af0a-e0eb6c0d4acf)

```

Bu Markdown kodu, belirli bir kümeleme öncesi ve sonrası veri setini ve sonuçları gösterir. Resim bağlantıları (`link_to_image1` ve `link_to_image2`) gerçek resim dosyalarınıza uygun olarak değiştirilmelidir.
