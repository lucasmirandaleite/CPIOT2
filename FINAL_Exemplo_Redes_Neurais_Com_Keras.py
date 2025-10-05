# Importando bibliotecas
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense

# Carregar o conjunto de dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension', 'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']
data = pd.read_csv(url, header=None, names=column_names)

data.head()

data.Diagnosis

data.shape

# Seleção das features:
X = data.iloc[:, 2:]

X.shape

X.head()

# Dados da coluna target:
y = data['Diagnosis']

y

# Usando o Label Encoder para converter colunas categóricas em numéricas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y

X.shape

y.shape

# Separação de dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape

X_test.shape

# Sequential é um recurso do Keras que permite a criação da nossa MLP (Multi Layer Perceptron)
my_nn = Sequential()

X.shape[1]

# Criando a primeira camada oculta
# Esta camada estará conectada à camada de entrada (Input Layer)
# input_dim -----> número de neurônios da camada de entrada

# ATENÇÃO: Número de neurônios = número de features (X.shape[1])

# Número de neurônios da primeira camada oculta (Hidden Layer)
# 16 neurônios ---> arbitrário para o exemplo
# Função de ativação para camadas ocultas  ---> 'relu'
my_nn.add(Dense(16, input_dim=X.shape[1], activation='relu'))

# Número de neurônios da segunda camada oculta (Hidden Layer)
# 8 neurônios ---> arbitrário para o exemplo
# Função de ativação para camadas ocultas  ---> 'relu'
my_nn.add(Dense(8, activation='relu'))

# Números de neurônios da camada de saída
# Classificação binária: 1 neurônio (0 ou 1)
# Classificação multiclasse: Número de classes a serem previstas
# Regressão: 1 neurônio (1 número)

# Função de ativação para camadas de saída:
# Classificação binária: 'sigmoid'
# Classificação multiclasse: 'softmax'
# Regressão: 'linear'

my_nn.add(Dense(1, activation='sigmoid'))

my_nn.summary()

# Função de Loss Function (Função de Custo)
# e Otimizador ----> permitem o cálculo dos pesos e viéses da rede
# buscando reduzir o erro na saída.

# Loss Function por tarefa:
# Classificação binária: 'binary_crossentropy'
# Classificação multiclasse: 'categorical_crossentropy'
# Regressão: 'mse'
# Otimizador ----> ADAM (qualquer tarefa ---> class ou regr)

# Méttrica do exemplo: Acurácia

# Compilar o modelo
my_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# my_nn

# Treinamento do modelo
my_nn.fit(X_train, y_train, epochs=50, batch_size=10)

loss, accuracy = my_nn.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
