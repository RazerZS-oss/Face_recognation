import os
import cv2
import numpy as np
import face_recognition
import dlib
import albumentations as A
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Inisialisasi model YOLOv8
yolo_model = YOLO('yolov8x.pt')

# Fungsi untuk memuat dataset dari folder
def load_dataset(dataset_paths):
    face_encodings = []
    face_labels = []
    
    for dataset_path in dataset_paths:
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = preprocess_image(image_path)
                faces = detect_faces_yolo(image)
                encodings = extract_face_encodings(image, faces)
                
                face_encodings.extend(encodings)
                face_labels.extend([person_name] * len(encodings))
    
    return face_encodings, face_labels

# Fungsi untuk preprocess gambar (ubah ke RGB dan resize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gagal memuat gambar: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    return image_resized

# Fungsi untuk deteksi wajah menggunakan YOLO
def detect_faces_yolo(image):
    results = yolo_model(image)
    faces = []

    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            if conf > 0.5:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    
    return faces

# Fungsi untuk ekstraksi encoding wajah menggunakan face_recognition / Dlib
def extract_face_encodings(image, faces):
    encodings = []
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        try:
            encoding = face_recognition.face_encodings(face_image_rgb)[0]
            encodings.append(encoding)
        except IndexError:
            print("Wajah tidak terdeteksi dengan baik.")
    
    return encodings

# Latih model KNN
def train_knn(train_encodings, train_labels):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(train_encodings, train_labels)
    return knn_model

# Evaluasi model KNN
def evaluate_model(knn_model, encodings, labels, dataset_type="Test"):
    correct_predictions = 0
    total_predictions = len(encodings)
    
    for encoding, true_label in zip(encodings, labels):
        predicted_label = knn_model.predict([encoding])[0]
        if predicted_label == true_label:
            correct_predictions += 1
        print(f"{dataset_type} - Predicted: {predicted_label}, Actual: {true_label}")
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Akurasi {dataset_type}: {accuracy:.2f}%")

# Path dataset
train_path = "dataset-r/train"
augment_path = "dataset-r/augment"
val_path = "dataset-r/val"
test_path = "dataset-r/test"

# Load data dari semua folder
dataset_paths = [train_path, augment_path, val_path]
encodings, labels = load_dataset(dataset_paths)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(encodings)):
    print(f"\nFold {fold + 1}")
    
    train_encodings = [encodings[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    
    val_encodings = [encodings[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    # Latih model KNN
    knn_model = train_knn(train_encodings, train_labels)

    # Evaluasi pada data validasi
    print("Evaluasi Validasi:")
    evaluate_model(knn_model, val_encodings, val_labels, dataset_type="Validation")

# Jika Anda ingin mengevaluasi model pada dataset test, Anda dapat melakukannya setelah K-fold cross-validation
test_encodings, test_labels = load_dataset([test_path])
print("\nEvaluasi Testing:")
evaluate_model(knn_model, test_encodings, test_labels, dataset_type="Test")
