import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("vendas.csv")

X = df[["marketing", "preco", "concorrencia"]]
y = df["vendas"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

score = modelo.score(X_test, y_test)
print("Acurácia:", round(score, 2))

entrada = pd.DataFrame({
    "marketing": [3000],
    "preco": [49],
    "concorrencia": [2]
})

print("Previsão:", round(modelo.predict(entrada)[0], 2))
