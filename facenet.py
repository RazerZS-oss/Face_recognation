from tensorflow.keras.models import load_model
import numpy as np

# Fungsi untuk memuat model FaceNet
def load_facenet_model(model_path='facenet_keras.h5'):
    # Memuat model FaceNet pretrained
    model = load_model(model_path)
    print("Model FaceNet berhasil dimuat")
    return model

# Fungsi untuk menghasilkan embedding dari wajah
def get_embedding(facenet_model, face_pixels):
    # Normalisasi input wajah ke nilai pixel [0,1]
    face_pixels = face_pixels.astype('float32')
    # Standarisasi berdasarkan statistik gambar (mean, std deviasi)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Transformasi input wajah ke ukuran 160x160 yang diperlukan oleh FaceNet
    face_pixels = np.expand_dims(face_pixels, axis=0)
    # Dapatkan embedding dari FaceNet
    embedding = facenet_model.predict(face_pixels)
    return embedding[0]

# Contoh memuat model
facenet_model = load_facenet_model('facenet_keras.h5')

face_pixels = extract_face(image)  # Mendeteksi dan memproses gambar wajah
if face_pixels is not None:
    embedding = get_embedding(facenet_model, face_pixels)
    print("Embedding:", embedding)

