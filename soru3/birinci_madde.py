# linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


X, y = datasets.make_moons(n_samples=1000, noise=0.05,
random_state=np.random.randint(80))


# Veriyi görselleştir (kümeleme öncesi)
plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
plt.title('Kümeleme Öncesi Veri Seti')
plt.show()

# Hiyerarşik kümeleme modelini oluştur
model = AgglomerativeClustering(n_clusters=2, linkage='single')  # linkage parametresi burada kullanılıyor
y_pred = model.fit_predict(X)
model1 = AgglomerativeClustering(n_clusters=2, linkage='ward')  # linkage parametresi burada kullanılıyor
y_pred1 = model1.fit_predict(X)
model2 = AgglomerativeClustering(n_clusters=2, linkage='complete')  # linkage parametresi burada kullanılıyor
y_pred2 = model2.fit_predict(X)
model3 = AgglomerativeClustering(n_clusters=2, linkage='average')  # linkage parametresi burada kullanılıyor
y_pred3 = model3.fit_predict(X)

# warda 
# single EN İYİSİ ZİNCİRLEMEDEN DOLAYI     / ARAŞTIRR
# complete 
# average 

# Modeli eğit
y_pred = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
plt.title('Single Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred1, cmap='viridis', s=20, alpha=0.7)
plt.title('Ward Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred2, cmap='viridis', s=20, alpha=0.7)
plt.title('Complete Bağlantı ile Hiyerarşik Kümeleme')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred3, cmap='viridis', s=20, alpha=0.7)
plt.title('Average Bağlantı ileHiyerarşik Kümeleme')
plt.show()


"""
Hiyerarşik kümeleme algoritmaları, belirli bir bağlantı (linkage) yöntemine dayanır. 
Linkage yöntemi, her iki küme arasındaki uzaklığı ölçer ve bu uzaklıklara göre yeni bir küme oluşturur. 
Bu yöntemler arasında "single", "complete", "average" ve "ward" (genellikle Euclidean uzaklık kullanır) gibi farklı seçenekler bulunur.

"Single" linkage yöntemi, iki küme arasındaki en yakın iki veri noktası arasındaki uzaklığı ölçer ve bu uzaklık değerine göre kümeleme yapar. 
Bu, "zincirleme" olarak adlandırılır çünkü birbirine çok yakın olan veri noktaları arasında zincir benzeri bir bağlantı oluşturulur. 
Bu durum, veri kümesinde lineer yapıların veya uzun zincir benzeri kümelerin varlığını varsayar.

Eğer veri kümenizde birbirine çok yakın olan ve belirgin bir şekilde farklı olan kümeler varsa, "single" linkage yöntemi bu durumu iyi tespit edebilir. 
Ancak, bu yöntem diğer durumlarda performansının düşük olabileceği bir özellik taşır. 
"Ward" veya "average" linkage yöntemleri, genelde daha dengeli ve genel amaçlı sonuçlar elde etmeye yöneliktir.

Bu nedenle, "single" linkage yönteminin neden bu özel veri kümesi için daha iyi performans gösterdiğini anlamak için, 
veri kümenizin özelliklerini ve içerdiği kümelerin yapısını incelemeniz gerekebilir. 
Eğer veri kümenizde lineer yapılar veya zincir benzeri kümeler varsa, "single" linkage tercih edilebilir. 
Ancak, genel olarak, farklı linkage yöntemlerini deneyerek ve sonuçları karşılaştırarak en iyi performansı sağlayanı seçmek önemlidir.
"""