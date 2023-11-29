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


"""
Bu çıktı, Locally Linear Embedding (LLE) algoritması tarafından gerçekleştirilen boyut azaltma işlemi sonucunda elde edilen iki boyutlu bir gösterimdir. 
Elde edilen görselleştirmeyi yorumlamak için, renklerin noktaların orijinal manifold yapısındaki konumlarını temsil ettiğini unutmayın.

Eğer n_neighbors parametresini değiştirirseniz, yani komşu sayısını artırırsanız veya azaltırsanız, bu LLE algoritmasının performansını etkiler. 
Bu parametre, her bir noktanın lokal bir manifold yapısı oluşturulurken kullanılan komşu sayısını belirler.

Azaltılırsa (n_neighbors azalırsa): Daha az komşu kullanmak, daha genel bir manifold yapısını temsil etme eğilimindedir. 
Ancak, çok az komşu kullanmak, lokal yapıyı yeterince yakalayamayabilir ve çözümü daha düşük kaliteli hale getirebilir. 
Bu durumda, görselleştirmede daha fazla düzensizlik ve kayıp ayrıntılar görebilirsiniz.

Arttırılırsa (n_neighbors arttırılırsa): Daha fazla komşu kullanmak, her noktanın lokal yapısını daha iyi yakalamaya çalışacaktır. 
Ancak, çok fazla komşu kullanmak da aşırı uyum problemlerine yol açabilir ve manifold yapısını bozabilir. 
Aşırı uyum, görselleştirmenin gereksiz ayrıntılar içermesine neden olabilir.

Bu nedenle, n_neighbors parametresini ayarlarken bir denge bulmanız önemlidir. 
Deneme yanılma yoluyla bu parametreyi optimize etmek, belirli bir veri seti için en iyi sonuçları elde etmenize yardımcı olabilir.
"""