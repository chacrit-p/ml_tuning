import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # ใช้สำหรับบันทึกและโหลดโมเดล

# อ่านข้อมูล
file_dataset_path = "assets/dataset/updated_pollution_dataset_cleaned.csv"
df = pd.read_csv(file_dataset_path)

# แยกฟีเจอร์ (X) และเป้าหมาย (y)
X = df.drop("Air Quality", axis=1)  # ฟีเจอร์
y = df["Air Quality"]  # เป้าหมาย

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# สร้างและเทรนโมเดล Random Forest
model = RandomForestClassifier(
    n_estimators=100,  # จำนวนต้นไม้
    max_depth=10,  # ความลึกสูงสุดของต้นไม้
    min_samples_split=4,  # จำนวนตัวอย่างขั้นต่ำในการแบ่งโหนด
    random_state=0,
)
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินโมเดล
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# บันทึกโมเดลที่เทรนแล้ว
model_file_path = "models/pkl/random_forest_model.pkl"
joblib.dump(model, model_file_path)

print(f"โมเดลถูกบันทึกเรียบร้อยในไฟล์ {model_file_path}")
