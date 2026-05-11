import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# 2. Делим на выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Обучаем модель (параметры прописаны сразу внутри)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Проверяем точность
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Точность accuracy: {accuracy:.2f}')