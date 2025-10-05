
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# 1. Carregar o Wine Dataset (UCI)
# O dataset Wine está disponível no scikit-learn
from sklearn.datasets import load_wine
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Converter as labels para o formato one-hot encoding para Keras (multiclasse)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# 2. Treinar uma rede neural em Keras
model_keras = Sequential()
model_keras.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_keras.add(Dense(32, activation='relu'))
model_keras.add(Dense(3, activation='softmax')) # 3 neurônios para 3 classes

model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTreinando o modelo Keras...")
history = model_keras.fit(X_train_scaled, y_train_categorical, epochs=100, batch_size=10, verbose=0)

# Avaliar o modelo Keras
loss_keras, accuracy_keras = model_keras.evaluate(X_test_scaled, y_test_categorical, verbose=0)
print(f'Acurácia do modelo Keras: {accuracy_keras * 100:.2f}%')

# 3. Comparar com um modelo do scikit-learn (RandomForestClassifier)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("\nTreinando o modelo RandomForestClassifier...")
model_rf.fit(X_train_scaled, y_train)

# Fazer previsões e avaliar o modelo RandomForest
y_pred_rf = model_rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Acurácia do modelo RandomForestClassifier: {accuracy_rf * 100:.2f}%')

# 4. Registrar métricas e discutir desempenho
print("\n--- Comparação de Desempenho ---")
print(f"Modelo Keras (Rede Neural): Acurácia = {accuracy_keras * 100:.2f}%")
print(f"Modelo Scikit-learn (RandomForestClassifier): Acurácia = {accuracy_rf * 100:.2f}%")

if accuracy_keras > accuracy_rf:
    print("O modelo Keras teve um desempenho ligeiramente melhor.")
elif accuracy_rf > accuracy_keras:
    print("O modelo RandomForestClassifier teve um desempenho ligeiramente melhor.")
else:
    print("Ambos os modelos tiveram desempenho semelhante.")

# Salvar resultados em um arquivo para o README.md
with open('exercise_1_results.txt', 'w') as f:
    f.write("### Exercício 1: Classificação Multiclasse (Wine Dataset)\n\n")
    f.write(f"Acurácia do modelo Keras (Rede Neural): {accuracy_keras * 100:.2f}%\n")
    f.write(f"Acurácia do modelo Scikit-learn (RandomForestClassifier): {accuracy_rf * 100:.2f}%\n")
    if accuracy_keras > accuracy_rf:
        f.write("Discussão: O modelo Keras teve um desempenho ligeiramente melhor neste conjunto de dados.\n")
    elif accuracy_rf > accuracy_keras:
        f.write("Discussão: O modelo RandomForestClassifier teve um desempenho ligeiramente melhor neste conjunto de dados.\n")
    else:
        f.write("Discussão: Ambos os modelos tiveram desempenho semelhante neste conjunto de dados.\n")

