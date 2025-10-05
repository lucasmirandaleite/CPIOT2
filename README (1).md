# 🧠 Projetos de Redes Neurais com Keras e Scikit-learn

Este repositório contém três exemplos práticos de aprendizado de máquina utilizando **Python**, **Keras** e **Scikit-learn**, com foco em **classificação e regressão**.

---

## 📋 Estrutura do Projeto

| Arquivo | Descrição |
|----------|------------|
| `FINAL_Exemplo_Redes_Neurais_Com_Keras.py` | Exemplo completo de classificação binária utilizando o dataset de **Câncer de Mama (Breast Cancer Wisconsin)**. Demonstra o uso de redes neurais (MLP) com Keras. |
| `exercise_1_wine_classification.py` | Exercício 1 — Classificação Multiclasse com o dataset **Wine (UCI)**. Compara o desempenho de uma rede neural (Keras) e um modelo de floresta aleatória (RandomForestClassifier). |
| `exercise_2_california_housing_regression.py` | Exercício 2 — Regressão com o dataset **California Housing**. Compara o desempenho de uma rede neural (Keras) e de um modelo de regressão via **RandomForestRegressor**. |

---

## ⚙️ Requisitos

Certifique-se de ter o **Python 3.8+** instalado e execute o comando abaixo para instalar as dependências:

```bash
pip3 install pandas scikit-learn keras tensorflow
```

---

## ▶️ Como Executar

### 🔹 Exemplo Principal — Rede Neural com Keras
Executa uma classificação binária com o dataset de câncer de mama:

```bash
python3 FINAL_Exemplo_Redes_Neurais_Com_Keras.py
```

**Saída esperada:**  
- Exibe o resumo da arquitetura da rede (camadas e parâmetros)  
- Exibe a acurácia final do modelo após o treinamento

---

### 🔹 Exercício 1 — Classificação Multiclasse (Wine Dataset)
Executa a comparação entre uma rede neural e um modelo de floresta aleatória:

```bash
python3 exercise_1_wine_classification.py
```

**Saída esperada:**  
- Exibe a acurácia dos dois modelos  
- Informa qual modelo teve melhor desempenho  
- Gera um arquivo `exercise_1_results.txt` com os resultados e discussão

---

### 🔹 Exercício 2 — Regressão (California Housing)
Executa a regressão com uma rede neural e um modelo de árvore:

```bash
python3 exercise_2_california_housing_regression.py
```

**Saída esperada:**  
- Exibe métricas de erro (RMSE e MAE)  
- Compara o desempenho entre a rede neural e o RandomForestRegressor  
- Gera um arquivo `exercise_2_results.txt` com os resultados e discussão

---

## 🧩 Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** → manipulação de dados
- **Scikit-learn** → datasets, pré-processamento e modelos tradicionais
- **Keras / TensorFlow** → construção e treinamento de redes neurais
- **RandomForest** → modelo de referência para comparação

---

## 📊 Resultados Esperados (Resumo)

| Exercício | Tipo de Problema | Modelos Comparados | Métricas |
|------------|------------------|--------------------|-----------|
| Exemplo Principal | Classificação Binária | MLP (Keras) | Acurácia |
| Exercício 1 | Classificação Multiclasse | MLP (Keras) × RandomForestClassifier | Acurácia |
| Exercício 2 | Regressão | MLP (Keras) × RandomForestRegressor | RMSE / MAE |

---

## 🧠 Conceitos Trabalhados

- Redes Neurais do tipo **MLP (Perceptron Multicamadas)**  
- Funções de ativação (`relu`, `sigmoid`, `softmax`, `linear`)  
- Funções de perda (`binary_crossentropy`, `categorical_crossentropy`, `mse`)  
- Normalização e pré-processamento de dados  
- Avaliação e comparação de modelos  
- Treinamento supervisionado (classificação e regressão)

---

## ✍️ Autor
**Lucas Leite**  
Projeto educacional — FIAP  
