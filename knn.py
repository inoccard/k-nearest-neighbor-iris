# ============================== biblioteca de carregamento ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter

# ============================== pré-processamento de dados ===========================================

# definir nomes de coluna
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# carregando dados de treinamento
df = pd.read_csv('../k-nearest-neighbor-iris/iris.data.txt', header=None, names=names)
print(df.head())

# criar matriz de design X e vector de destino y
X = np.array(df.ix[:, 0:4]) 	# índice final e exclusivo
y = np.array(df['class']) 	# mostrando duas maneiras de indexar um pandas df

# dividido em trem e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.33, random_state=42)


# modelo de aprendizagem essencial (k = 4)
knn = KNeighborsClassifier(n_neighbors=4)

# ajustando o modelo
knn.fit(X_treino, y_treino)

# prever a resposta
prev = knn.predict(X_teste)

# avaliar a precisão
acc = accuracy_score(y_teste, prev) * 100
print('\nA precisão do classificador knn para k = 3 é %d%%' % acc)

# ============================== ajuste de parâmetro ======================================
# criando lista impar de K para KNN
myList = list(range(0,40))
# subconjunto apenas os mais estranhos
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# lista vazia que conterá pontuações de cv
cv_scores = []

# realizar validação cruzada de 10 vezes
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_treino, y_treino, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# mudando para erro de classificação incorreta
MSE = [1 - x for x in cv_scores]

# determinando melhor k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nO número ideal de neighbors  (vizinhos) é %d.' % optimal_k)

# erro de classificação incorreta do enredo vs k 
plt.plot(neighbors, MSE)
plt.xlabel('Número de Neighbors (vizinhos) K')
plt.ylabel('Erro de classificação incorreta')
plt.show()


def train(X_treino, y_treino):
	# fazer nada 
	return

def predict(X_treino, y_treino, x_teste, k):
	# cria lista para distâncias e alvos
	distances = []
	targets   = []

	for i in range(len(X_treino)):
		# primeiro calculamos a distância euclidiana
		distance = np.sqrt(np.sum(np.square(x_teste - X_treino[i, :])))
		# adicione-o à lista de distâncias
		distances.append([distance, i])

	# ordena a lista
	distances = sorted(distances)

	# faça uma lista dos alvos dos neighbors K
	for i in range(k):
		index = distances[i][1]
		#print(y_treino[index])
		targets.append(y_treino[index])

	# retorna o alvo mais em comun
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_treino, y_treino, X_teste, previsoes, k):
	# verifique se k não é maior que n
	if k > len(X_treino):
		raise ValueError
		
	# treina nos dados de entrada
	train(X_treino, y_treino)

	# prevê para cada observação de teste
	for i in range(len(X_teste)):
		previsoes.append(predict(X_treino, y_treino, X_teste[i, :], k))
        
# =========================== testando KNN =====================================
# Fazendo as previsões
previsoes = []
try:
	kNearestNeighbor(X_treino, y_treino, X_teste, previsoes, 8)
    
    # transforma a lista em um array
	previsoes = np.asarray(previsoes)

	# avaliando a precisão
	accuracy = accuracy_score(y_teste, previsoes) * 100
	print('\nA precisão do nosso classificador é %d%%' % accuracy)

except ValueError:
	print('Não se pode ter mais vizinhos do que as amostras de treinamento!!')
