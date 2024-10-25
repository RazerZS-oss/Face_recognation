import os
import cv2
import dlib
import numpy as np
import face_recognition
import albumentations as A
from sklearn.neighbors import KNeighborsClassifier

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

    # Resize gambar menjadi 500x500
    image_resized = cv2.resize(image_rgb, (500, 500))

    return image_resized

# Load data train, validasi, dan test
train_path = "dataset/train1"
augmentasi_path = "dataset/augment1"
validasi_path = "dataset/val1"
test_path = "dataset/test1"

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
    # Load YOLO model
    yolo_net = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    # Deteksi wajah
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    faces = []
    for out in outs:
        for detection in out:
            scores = detection[5:]  # Ambil skor dari deteksi
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Ambang batas kepercayaan
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                # Koordinat untuk wajah
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                faces.append((x, y, w, h))

    return faces

# Uji model YOLO
def test_yolo_model(image_path):
    image = preprocess_image(image_path)
    faces = detect_faces_yolo(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("YOLO Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uji YOLO pada gambar
test_yolo_model("dataset/test1/S428/S428-05-t10_01.jpg")
