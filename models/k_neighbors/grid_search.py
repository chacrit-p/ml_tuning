import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# อ่านข้อมูล
file_dataset_path = "assets/dataset/updated_pollution_dataset_cleaned.csv"
df = pd.read_csv(file_dataset_path)

param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 12, 13, 14],
    "weights": ["uniform", "distance"],
    "p": [1, 2],  # 1 = Manhattan Distance, 2 = Euclidean Distance
}

# แยกฟีเจอร์ (X) และเป้าหมาย (y)
X = df.drop("Air Quality", axis=1)  # ฟีเจอร์
y = df["Air Quality"]  # เป้าหมาย

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# แสดงพารามิเตอร์ที่ดีที่สุด
print("Best Parameters:", grid_search.best_params_)

# ใช้พารามิเตอร์ที่ดีที่สุด
best_model = grid_search.best_estimator_

# ทดสอบบน Test Set
y_pred_best = best_model.predict(X_test)
print("Accuracy (Best Model):", accuracy_score(y_test, y_pred_best))
