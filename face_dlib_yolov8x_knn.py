import os
import cv2
import dlib
import numpy as np
import face_recognition
import albumentations as A
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO  # Import untuk YOLOv8
import matplotlib.pyplot as plt  # Import untuk menampilkan gambar

# Inisialisasi dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fungsi untuk memuat data dari folder dataset (train/validasi/test)
def load_dataset(dataset_path):
    face_encodings = []
    face_names = []
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = preprocess_image(image_path)  # Menggunakan fungsi preprocess
            try:
                # Deteksi wajah menggunakan dlib
                dets = detector(image, 1)
                for d in dets:
                    shape = sp(image, d)
                    face_encoding = np.array(face_recognition.face_encodings(image)[0])
                    face_encodings.append(face_encoding)
                    face_names.append(person_name)
            except IndexError:
                print(f"Wajah tidak terdeteksi di {image_name}. Lewati...")

    return face_encodings, face_names

# Augmentasi gambar
def augment_image(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Flip horizontal
        A.Rotate(limit=25, p=1.0),  # Rotasi
        A.RandomBrightnessContrast(p=0.5),  # Ubah kecerahan
    ])
    augmented = transform(image=image)
    return augmented['image']

# Fungsi untuk preprocess gambar (RGB dan Resize)
def preprocess_image(image_path):
    # Muat gambar menggunakan OpenCV
    image = cv2.imread(image_path)

    # Ubah gambar ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize gambar menjadi 
    image_resized = cv2.resize(image_rgb, (640, 640))

    return image_resized

# Load data train, validasi, dan test
train_path = "dataset-r/train"
augmentasi_path = "dataset-r/augment"
validasi_path = "dataset-r/val"
test_path = "dataset-r/test"

# Load data asli
train_encodings, train_names = load_dataset(train_path)
validasi_encodings, validasi_names = load_dataset(validasi_path)
test_encodings, test_names = load_dataset(test_path)

# Terapkan augmentasi ke dataset augmentasi
augmentasi_encodings, augmentasi_names = load_dataset(augmentasi_path)
for person_name in os.listdir(augmentasi_path):
    person_folder = os.path.join(augmentasi_path, person_name)
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = preprocess_image(image_path)  # Menggunakan fungsi preprocess
        augmented_image = augment_image(image)
        try:
            augmented_encoding = np.array(face_recognition.face_encodings(augmented_image)[0])
            augmentasi_encodings.append(augmented_encoding)
            augmentasi_names.append(person_name)
        except IndexError:
            print(f"Augmentasi wajah tidak terdeteksi di {image_name}. Lewati...")

# Tambahkan data augmentasi ke data training
train_encodings.extend(augmentasi_encodings)
train_names.extend(augmentasi_names)

# Latih model KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Sesuaikan jumlah neighbors sesuai kebutuhan
knn_model.fit(train_encodings, train_names)

# Fungsi untuk menguji model
def test_model(encodings, names, dataset_type="Test"):
    correct_predictions = 0
    total_predictions = len(encodings)
    
    for test_encoding, test_name in zip(encodings, names):
        predicted_name = knn_model.predict([test_encoding])[0]  # Prediksi menggunakan KNN

        if predicted_name == test_name:
            correct_predictions += 1
        print(f"{dataset_type} - Predicted: {predicted_name}, Actual: {test_name}")
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Akurasi {dataset_type}: {accuracy:.2f}%")

# Uji model dengan data validasi
print("Validasi:")
test_model(validasi_encodings, validasi_names, dataset_type="Validation")

# Uji model dengan data test
print("\nTesting:")
test_model(test_encodings, test_names, dataset_type="Testing")

# Menggunakan YOLO untuk mendeteksi wajah pada gambar
def detect_faces_yolo(image):
    # Load YOLO model dari file .pt
    model = YOLO('yolov8x.pt')  # Ganti dengan path ke file .pt

    # Deteksi wajah
    results = model(image)
    
    # Ambil hasil deteksi
    faces = []
    for result in results:  # Mengambil hasil deteksi dari model YOLO
        for *box, conf in result.boxes.data.tolist():  # Format: [x1, y1, x2, y2, confidence]
            if conf > 0.5:  # Ambang batas kepercayaan
                x1, y1, x2, y2 = map(int, box)  # Konversi ke integer
                faces.append((x1, y1, x2 - x1, y2 - y1))  # Menyimpan bounding box

    return faces

# Fungsi untuk menguji model YOLO dan mengenali wajah
def test_yolo_and_recognize(image_path):
    # Preprocess gambar
    image = preprocess_image(image_path)
    faces = detect_faces_yolo(image)

    for (x, y, w, h) in faces:
        # Potong wajah dari gambar
        face_image = image[y:y+h, x:x+w]
        
        # Ubah wajah menjadi encoding menggunakan face_recognition
        try:
            face_encoding = np.array(face_recognition.face_encodings(face_image)[0])
            predicted_name = knn_model.predict([face_encoding])[0]  # Prediksi menggunakan KNN
            print(f"Predicted Name: {predicted_name}")
            
            # Gambar bounding box dan nama di atas wajah
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except IndexError:
            print("Wajah tidak terdeteksi untuk bounding box ini.")

    # Tampilkan gambar dengan matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR ke RGB untuk tampilan plt
    plt.axis('off')  # Hilangkan axis pada tampilan gambar
    plt.show()

# Uji pada gambar
test_yolo_and_recognize ("dataset/test1/S426/aug_3_S426-01-t10_01.jpg")  # Ganti dengan path ke gambar yang ingin diuji