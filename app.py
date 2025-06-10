from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import pandas as pd
import random
from datetime import datetime
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("defect_detection_model.h5")

casting_types = ["Die Casting", "Sand Casting"]
materials = ["Aluminum", "Iron", "Steel"]
part_types = ["Engine Block", "Brake Disc", "Cylinder Head", "Gearbox", "Piston"]
solutions = ["Replace component", "Inspect further", "No action needed", "Schedule repair"]
defect_types = ["Crack", "Hole", "Scratch", "Surface Defect", "Discoloration", "Burn Mark", "Edge Deformation", "Air Bubble", "Shrinkage", "Foreign Particle"]
HISTORY_CSV = "prediction_history.csv"

def save_to_history(data):
    df = pd.DataFrame([data])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

def determine_shift(hour):
    if 6 <= hour < 14:
        return "Shift 1"
    elif 14 <= hour < 22:
        return "Shift 2"
    else:
        return "Shift 3"

def generate_graphs():
    if not os.path.exists(HISTORY_CSV):
        return

    df = pd.read_csv(HISTORY_CSV)

    defect_counts = df["detected_defects"].value_counts().head(6)
    plt.figure(figsize=(5, 5))
    explode = [0.05] * len(defect_counts)
    defect_counts.plot.pie(autopct="%1.1f%%", explode=explode, shadow=True)
    plt.title("Defect Types Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("static/uploads/graph1_pie_defect_types.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    defect_counts.plot(kind='bar', color='skyblue')
    plt.title("Defect Counts")
    plt.xlabel("Defect Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("static/uploads/graph2_bar_defects.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    material_defects = df[df['result'].str.contains("DEFECTIVE")].groupby('material').size()
    material_defects.plot(kind='bar', color='orange')
    plt.title("Defects per Material")
    plt.xlabel("Material")
    plt.ylabel("Defect Count")
    plt.tight_layout()
    plt.savefig("static/uploads/graph3_material_defects.png")
    plt.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_trend = df[df['result'].str.contains("DEFECTIVE")].set_index('timestamp').resample('D').size()
    plt.figure(figsize=(6, 4))
    time_trend.plot(marker='o')
    plt.title("Defect Trend Over Time")
    plt.ylabel("Defects per Day")
    plt.tight_layout()
    plt.savefig("static/uploads/graph4_defect_trend.png")
    plt.close()

    types = df["result"].apply(lambda x: "Defective" if "DEFECTIVE" in x else "Non-Defective")
    plt.figure(figsize=(4, 4))
    types.value_counts().plot.pie(autopct="%1.1f%%", colors=["red", "green"])
    plt.title("Defective vs Non-Defective")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("static/uploads/graph5_defect_vs_non.png")
    plt.close()

    # Shift-wise parts count
    df['shift'] = df['timestamp'].apply(lambda x: determine_shift(pd.to_datetime(x).hour))
    shift_counts = df.groupby('shift').size()
    plt.figure(figsize=(5.5, 4))
    shift_counts.plot(kind='bar', color='purple')
    plt.title("Parts Tested per Shift")
    plt.xlabel("Shift")
    plt.ylabel("Total Parts")
    plt.tight_layout()
    plt.savefig("static/uploads/graph6_shift_wise.png")
    plt.close()

    # NEW: Shift-wise DEFECTIVE only
    shift_defective = df[df['result'].str.contains("DEFECTIVE")]
    defective_by_shift = shift_defective.groupby('shift').size()
    plt.figure(figsize=(5.5, 4))
    defective_by_shift.plot(kind='bar', color='crimson')
    plt.title("Defective Parts per Shift")
    plt.xlabel("Shift")
    plt.ylabel("Defective Count")
    plt.tight_layout()
    plt.savefig("static/uploads/graph7_defective_parts_by_shift.png")
    plt.close()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    from_webcam = request.form.get('webcam')

    if from_webcam == 'true':
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()
        if not ret:
            return "Webcam capture failed"
        filename = "webcam_capture.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
    elif file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        return "No image selected or uploaded"

    img = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    if predicted_class == 1:
        result = "🔴 DEFECTIVE part detected"
        detected_defects = ", ".join(random.sample(defect_types, random.randint(2, 4)))
        severity = "High"
        repair_time = "2–4 hours"
        solution = random.choice(solutions[:-1])
    else:
        result = "🟢 NON-DEFECTIVE part"
        detected_defects = "None"
        severity = "None"
        repair_time = "None"
        solution = "No action needed"

    casting_type = random.choice(casting_types)
    material = random.choice(materials)
    part_type = random.choice(part_types)
    serial_number = "PART-" + str(random.randint(1000, 9999))

    labels = ['Non-Defective', 'Defective']
    plt.figure(figsize=(4, 4))
    plt.pie(prediction[0], labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Confidence Chart")
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], "chart.png")
    plt.savefig(chart_path)
    plt.close()

    save_to_history({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "confidence": round(confidence * 100, 2),
        "casting_type": casting_type,
        "material": material,
        "part_type": part_type,
        "detected_defects": detected_defects
    })

    generate_graphs()

    return render_template("index.html",
        result=result,
        confidence=f"{confidence*100:.2f}%",
        casting_type=casting_type,
        material=material,
        part_type=part_type,
        detected_defects=detected_defects,
        severity=severity,
        repair_time=repair_time,
        solution=solution,
        serial_number=serial_number,
        image_path=filepath,
        chart_path=chart_path,
        graph1="graph1_pie_defect_types.png",
        graph2="graph2_bar_defects.png",
        graph3="graph3_material_defects.png",
        graph4="graph4_defect_trend.png",
        graph5="graph5_defect_vs_non.png",
        graph6="graph6_shift_wise.png",
        graph7="graph7_defective_parts_by_shift.png"  # ✅ NEW
    )

if __name__ == "__main__":
    app.run(debug=True)
