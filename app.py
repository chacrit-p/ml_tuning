import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model_file_path = "models/pkl/random_forest_model.pkl"
model = joblib.load(model_file_path)


# ฟังก์ชันสำหรับพยากรณ์
def predict():
    try:
        # อ่านค่าที่กรอกในช่อง
        inputs = [
            float(entry_temperature.get()),
            float(entry_humidity.get()),
            float(entry_pm25.get()),
            float(entry_pm10.get()),
            float(entry_no2.get()),
            float(entry_so2.get()),
            float(entry_co.get()),
            float(entry_proximity.get()),
            float(entry_population_density.get()),
        ]

        # แปลงข้อมูลเป็นรูปแบบที่โมเดลต้องการ
        inputs = np.array(inputs).reshape(1, -1)

        # พยากรณ์ผลลัพธ์
        prediction = model.predict(inputs)
        result_label.config(text=f"ผลลัพธ์การพยากรณ์: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Error", "กรุณากรอกค่าที่ถูกต้องในทุกช่อง")


# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("แอปพลิเคชันพยากรณ์คุณภาพอากาศ")
root.geometry("400x600")

# สร้างส่วนอินพุต
tk.Label(root, text="Temperature (°C):").pack()
entry_temperature = tk.Entry(root)
entry_temperature.pack()

tk.Label(root, text="Humidity (%):").pack()
entry_humidity = tk.Entry(root)
entry_humidity.pack()

tk.Label(root, text="PM2.5 (µg/m³):").pack()
entry_pm25 = tk.Entry(root)
entry_pm25.pack()

tk.Label(root, text="PM10 (µg/m³):").pack()
entry_pm10 = tk.Entry(root)
entry_pm10.pack()

tk.Label(root, text="NO2 (ppb):").pack()
entry_no2 = tk.Entry(root)
entry_no2.pack()

tk.Label(root, text="SO2 (ppb):").pack()
entry_so2 = tk.Entry(root)
entry_so2.pack()

tk.Label(root, text="CO (ppm):").pack()
entry_co = tk.Entry(root)
entry_co.pack()

tk.Label(root, text="Proximity to Industrial Areas (km):").pack()
entry_proximity = tk.Entry(root)
entry_proximity.pack()

tk.Label(root, text="Population Density (people/km²):").pack()
entry_population_density = tk.Entry(root)
entry_population_density.pack()

# สร้างปุ่มสำหรับพยากรณ์
predict_button = tk.Button(root, text="พยากรณ์", command=predict)
predict_button.pack(pady=20)

# แสดงผลลัพธ์
result_label = tk.Label(root, text="ผลลัพธ์การพยากรณ์: ", font=("Arial", 14))
result_label.pack()

# เริ่มแอปพลิเคชัน
root.mainloop()
