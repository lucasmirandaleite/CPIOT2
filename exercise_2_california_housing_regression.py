
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense

# 1. Carregar o California Housing Dataset (Scikit-learn)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Treinar uma rede neural em Keras
model_keras = Sequential()
model_keras.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_keras.add(Dense(32, activation='relu'))
model_keras.add(Dense(16, activation='relu'))
model_keras.add(Dense(1, activation='linear')) # 1 neurônio para regressão

model_keras.compile(optimizer='adam', loss='mse')

print("\nTreinando o modelo Keras...")
history = model_keras.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# Avaliar o modelo Keras
y_pred_keras = model_keras.predict(X_test_scaled, verbose=0)
mse_keras = mean_squared_error(y_test, y_pred_keras)
rmse_keras = mse_keras**0.5
mae_keras = mean_absolute_error(y_test, y_pred_keras)
print(f'RMSE do modelo Keras: {rmse_keras:.2f}')
print(f'MAE do modelo Keras: {mae_keras:.2f}')

# 3. Comparar com um modelo do scikit-learn (RandomForestRegressor)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nTreinando o modelo RandomForestRegressor...")
model_rf.fit(X_train_scaled, y_train)

# Fazer previsões e avaliar o modelo RandomForest
y_pred_rf = model_rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf**0.5
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'RMSE do modelo RandomForestRegressor: {rmse_rf:.2f}')
print(f'MAE do modelo RandomForestRegressor: {mae_rf:.2f}')

# 4. Registrar métricas e discutir desempenho
print("\n--- Comparação de Desempenho ---")
print(f'Modelo Keras (Rede Neural): RMSE = {rmse_keras:.2f}, MAE = {mae_keras:.2f}')
print(f'Modelo Scikit-learn (RandomForestRegressor): RMSE = {rmse_rf:.2f}, MAE = {mae_rf:.2f}')

if rmse_keras < rmse_rf:
    print("O modelo Keras teve um desempenho ligeiramente melhor (menor RMSE).")
elif rmse_rf < rmse_keras:
    print("O modelo RandomForestRegressor teve um desempenho ligeiramente melhor (menor RMSE).")
else:
    print("Ambos os modelos tiveram desempenho semelhante em termos de RMSE.")

# Salvar resultados em um arquivo para o README.md
with open('exercise_2_results.txt', 'w') as f:
    f.write("### Exercício 2: Regressão (California Housing Dataset)\n\n")
    f.write(f'RMSE do modelo Keras (Rede Neural): {rmse_keras:.2f}\n')
    f.write(f'MAE do modelo Keras (Rede Neural): {mae_keras:.2f}\n')
    f.write(f'RMSE do modelo Scikit-learn (RandomForestRegressor): {rmse_rf:.2f}\n')
    f.write(f'MAE do modelo Scikit-learn (RandomForestRegressor): {mae_rf:.2f}\n')
    if rmse_keras < rmse_rf:
        f.write("Discussão: O modelo Keras teve um desempenho ligeiramente melhor (menor RMSE) neste conjunto de dados.\n")
    elif rmse_rf > rmse_keras:
        f.write("Discussão: O modelo RandomForestRegressor teve um desempenho ligeiramente melhor (menor RMSE) neste conjunto de dados.\n")
    else:
        f.write("Discussão: Ambos os modelos tiveram desempenho semelhante em termos de RMSE neste conjunto de dados.\n")

