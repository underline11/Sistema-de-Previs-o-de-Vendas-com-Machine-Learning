import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X = df[data.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Acurácia:", round(acc, 2))

entrada = pd.DataFrame([{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}])

print("Previsão:", modelo.predict(entrada)[0])
