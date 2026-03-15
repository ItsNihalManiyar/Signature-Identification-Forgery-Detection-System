import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = []
    y_id = []
    y_verify = []

    for index, row in df.iterrows():
        img_path = row['File_Path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Ensure proper shape (224, 224) and normalize
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        X.append(img)
        # ID uses 1-55, map to 0-54 for softmax
        y_id.append(row['ID'] - 1)
        y_verify.append(row['Label'])

    X = np.array(X).reshape(-1, 224, 224, 1)
    y_id = np.array(y_id)
    y_verify = np.array(y_verify)
    return X, y_id, y_verify

def build_id_model(input_shape=(224, 224, 1), num_classes=55):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_verify_model(input_shape=(224, 224, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'dataset.csv')
    
    if not os.path.exists(csv_path):
        print("dataset.csv not found. Please run create_dataset.py first.")
        return

    print("Loading data...")
    X, y_id, y_verify = load_data(csv_path)

    # Train ID Model
    print("Training ID Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_id, test_size=0.2, random_state=42)
    id_model = build_id_model()
    # Using 5 epochs to demonstrate training while keeping execution time manageable
    id_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
    id_model.save('student_id_model.h5')
    print("Saved student_id_model.h5")

    # Train Verify Model
    print("Training Verify Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_verify, test_size=0.2, random_state=42)
    verify_model = build_verify_model()
    verify_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
    verify_model.save('forgery_detect_model.h5')
    print("Saved forgery_detect_model.h5")

if __name__ == "__main__":
    main()
