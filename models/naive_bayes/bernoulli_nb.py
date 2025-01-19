import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # ใช้สำหรับบันทึกและโหลดโมเดล

# อ่านข้อมูล
file_dataset_path = "assets/dataset/updated_pollution_dataset_cleaned.csv"
df = pd.read_csv(file_dataset_path)

# แยกฟีเจอร์ (X) และเป้าหมาย (y)
X = df.drop("Air Quality", axis=1)  # ฟีเจอร์
y = df["Air Quality"]  # เป้าหมาย

# แปลงข้อมูลให้เป็น Binary (ค่า 0 และ 1) สำหรับ BernoulliNB
X_binary = (X > 0).astype(int)

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y, test_size=0.3, random_state=0
)

# สร้างและเทรนโมเดล Bernoulli Naive Bayes
model = BernoulliNB()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินโมเดล
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# บันทึกโมเดลที่เทรนแล้ว
model_file_path = "models/pkl/bernoulli_nb_model.pkl"
joblib.dump(model, model_file_path)

print(f"โมเดลถูกบันทึกเรียบร้อยในไฟล์ {model_file_path}")
