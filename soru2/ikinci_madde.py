from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

# Veri setini oluştur
X, y = datasets.make_blobs(n_samples=1000, centers=4,
                            cluster_std=[np.random.rand()*2, np.random.rand()*2,
                                        np.random.rand()*2, np.random.rand()*2])

def plot_dendrogram(model, **kwargs):
    linkage_matrix = linkage(X, model.linkage, metric='euclidean')       # “linkage” fonksiyonu fonksiyonu, “X” veri setindeki örnekler arasındaki benzerlikleri ölçerek “linkage_matrix” bağlantı matrisini oluşturur.
    dendrogram(linkage_matrix, **kwargs)                                 # Dendrogramı çiz

# Hiyerarşik kümeleme modelini oluştur
model = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=0)

# Modeli eğit ve dendrogramı çiz
plot_dendrogram(model)
plt.title('Dendrogram')
plt.xlabel('Veri Noktaları')
plt.ylabel('Uzaklık')
plt.show()

# Dendrogramdaki kesilme noktalarını belirle
distance_threshold = 30  # Uyarı: Bu değeri görsel olarak inceleyerek belirlemeniz gerekebilir.

# Hiyerarşik kümeleme modelini oluştur ve eğit
model = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=distance_threshold)
y_pred = model.fit_predict(X)

# Optimal küme sayısını ekrana yazdır
num_clusters = len(set(y_pred))
print(f"Optimal Küme Sayısı: {num_clusters}")

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20, alpha=0.7)
plt.title('Hiyerarşik Clustering')
plt.show()



"""
Elbette, dendrogram analizi yapmak ve optimal küme sayısını belirlemek biraz daha ayrıntılı bir yaklaşım gerektirir. 
İlk olarak, dendrogramın nasıl oluşturulduğunu anlayalım.
Dendrogram, hiyerarşik kümeleme işlemi sırasında birleşen kümeleri ve bu birleşmelerin ne kadar uzaklıkta olduğunu gösteren bir ağaç yapısıdır. 
Dendrogramın yatay ekseni, veri noktalarını, dikey ekseni ise birbirleriyle birleşen küme veya alt kümeleri temsil eder. 
Dendrogramın alt kısmında yer alan çizgiler, birleşen kümelerin uzaklıklarını temsil eder.
Optimal küme sayısını belirlemek için genellikle, dendrogramın altındaki çizgilerin uzunluğuna dikkat edilir. 
İdeal durumda, birbirine yakın kümeleşen veri noktalarını birleştiren çizgilerin uzunluğu küçük olmalıdır. 
Ancak, bu uzunlukların tam olarak ne kadar küçük olması gerektiği konusunda bir kesinlik yoktur.
Bu nedenle, bir eşik (threshold) belirlemeniz gerekebilir. Bu eşik, dendrogramdaki çizgilerin uzunluğu tarafından belirlenir. 
Dendrogramda uzunlukları görsel olarak inceledikten sonra, veri setiniz ve analiz amacınıza bağlı olarak bir eşik değeri belirlersiniz.
Bu eşik değeri, birleşen kümeleşmelerin belirli bir uzaklık eşiğini temsil eder.
Kod örneğinde, `distance_threshold` değeri bu eşiği belirler. 
Bu değer, dendrogramdaki çizgilerin uzunluğunu temsil eder ve birleşen kümeleşmelerin belirli bir uzaklık eşiği üzerinde durmasını sağlar. 
Eğer bu eşik değeri belirlenmiş bir değilse (örneğin, 20 gibi bir değer), 
o zaman bu değeri kullanarak hiyerarşik kümeleme modelini tekrar eğitirsiniz ve elde ettiğiniz küme sayısını belirlersiniz.
Bu, bir tür görsel analiz ve öznel bir karardır. Hangi eşik değerinin uygun olduğunu belirlemek, veri setinize ve analiz amacınıza bağlı olarak değişebilir. 
Bu nedenle, farklı eşik değerlerini deneyerek sonuçları değerlendirmek faydalı olabilir.
"""