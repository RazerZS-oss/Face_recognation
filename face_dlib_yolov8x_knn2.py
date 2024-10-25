import cv2
import os
import albumentations as A
from ultralytics import YOLO
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Inisialisasi model YOLO sekali di awal
yolo_model = YOLO('yolov8x.pt')  # Sesuaikan model YOLOv8 yang Anda gunakan

# Fungsi untuk preprocess gambar
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    return image_resized

# Fungsi untuk augmentasi gambar
def augment_image(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=1.0),
        A.RandomBrightnessContrast(p=0.5),
    ])
    augmented = transform(image=image)
    return augmented['image']

# Fungsi untuk deteksi wajah dengan YOLO
def detect_faces_yolo(image):
    results = yolo_model(image)
    
    faces = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2 = map(int, box[:4])  # Hanya ambil 4 nilai pertama untuk bounding box
            conf = box[4]  # Ambil nilai confidence
            if conf > 0.5:  # Ambang batas confidence
                faces.append((x1, y1, x2 - x1, y2 - y1))
    
    return faces

# Fungsi untuk ekstraksi fitur wajah
def extract_face_encodings(image, faces):
    encodings = []
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        # Konversi ke RGB karena face_recognition mengharapkan gambar dalam format ini
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        try:
            # Ambil encoding wajah menggunakan deteksi wajah yang benar
            encoding = face_recognition.face_encodings(face_image_rgb)[0]
            encodings.append(encoding)
        except IndexError:
            print("Wajah tidak terdeteksi pada gambar.")
    
    return encodings


# Fungsi untuk memuat dataset
def load_dataset(dataset_path):
    face_encodings = []
    face_labels = []
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = preprocess_image(image_path)
            faces = detect_faces_yolo(image)
            encodings = extract_face_encodings(image, faces)
            
            # Augmentasi gambar dan ekstraksi encoding
            augmented_image = augment_image(image)
            augmented_faces = detect_faces_yolo(augmented_image)
            augmented_encodings = extract_face_encodings(augmented_image, augmented_faces)
            
            face_encodings.extend(encodings + augmented_encodings)
            face_labels.extend([person_name] * (len(encodings) + len(augmented_encodings)))
    
    return face_encodings, face_labels

# Latih model KNN
def train_knn(train_encodings, train_labels):
    knn_model = KNeighborsClassifier(n_neighbors=3)  # Sesuaikan jumlah tetangga
    knn_model.fit(train_encodings, train_labels)
    return knn_model

# Evaluasi model
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

# Load dataset
train_encodings, train_labels = load_dataset("dataset-r/train")
val_encodings, val_labels = load_dataset("dataset-r/val")
test_encodings, test_labels = load_dataset("dataset-r/test")

# Latih KNN
knn_model = train_knn(train_encodings, train_labels)

# Evaluasi pada validasi dan test
print("Evaluasi Validasi:")
evaluate_model(knn_model, val_encodings, val_labels, dataset_type="Validation")

print("Evaluasi Testing:")
evaluate_model(knn_model, test_encodings, test_labels, dataset_type="Test")
