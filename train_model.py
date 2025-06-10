import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# ✅ 1. Load the dataset
df = pd.read_csv("newdatasets/final_training_dataset.csv")  # merged CSV

# ✅ 2. Prepare images and labels
image_dir = "newdatasets/images/"
images = []
labels = []

for index, row in df.iterrows():
    img_path = os.path.join(image_dir, row['label'])  # label column has filenames
    if os.path.exists(img_path):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)

        # ✅ Determine label (0 = Non-defective, 1 = Defective)
        if str(row['defects_detected']).lower() == 'none':
            labels.append(0)
        else:
            labels.append(1)

# ✅ 3. Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# ✅ 4. Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 5. Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 0: Non-defective, 1: Defective
])

# ✅ 6. Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ 7. Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# ✅ 8. Save
model.save("defect_detection_model.h5")
print("✅ Model training complete and saved.")
