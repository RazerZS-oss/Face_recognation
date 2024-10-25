import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from imgaug import augmenters as iaa

# Load model FaceNet
def load_facenet_model(model_path='model/keras/facenet_keras.h5'):
    if not os.path.exists(model_path):
        print(f"Error: File model {model_path} tidak ditemukan.")
        return None
    try:
        model = load_model(model_path)
        print("Model FaceNet berhasil dimuat")
        return model
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None

# Fungsi untuk ekstraksi wajah menggunakan MTCNN
def extract_face(image, required_size=(160, 160)):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face

# Fungsi untuk augmentasi gambar
def augment_image(image):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # Flip horizontal
        iaa.Multiply((0.8, 1.2)),  # Random brightness
        iaa.GaussianBlur(sigma=(0, 1.0))  # Gaussian blur
    ])
    return seq.augment_image(image)

# Fungsi untuk mendapatkan embedding dari wajah
def get_embedding(facenet_model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.predict(face_pixels)
    return embedding[0]

# Fungsi untuk memproses dataset
def process_dataset(dataset_dir, facenet_model, augment=False):
    embeddings = {}
    for subdir in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(path):
            continue
        embeddings[subdir] = []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            image = cv2.imread(filepath)
            
            # Validasi apakah gambar berhasil di-load
            if image is None:
                print(f"Error: Gambar {filepath} tidak dapat dibuka.")
                continue
            
            face = extract_face(image)
            if face is None:
                print(f"Error: Tidak ada wajah terdeteksi di {filepath}.")
                continue
            
            # Augmentasi jika diminta
            if augment:
                face = augment_image(face)
                
            embedding = get_embedding(facenet_model, face)
            embeddings[subdir].append(embedding)
    return embeddings

# Fungsi utama untuk membuat embedding dan menyimpan ke file
def save_embeddings():
    facenet_model = load_facenet_model('facenet_keras.h5')
    
    if facenet_model is None:
        print("Error: Model tidak dimuat. Proses dihentikan.")
        return
    
    # Proses data train dengan augmentasi
    train_embeddings = process_dataset('dataset/train', facenet_model, augment=True)
    np.save('train_embeddings.npy', train_embeddings)
    
    # Proses data validasi tanpa augmentasi
    val_embeddings = process_dataset('dataset/val', facenet_model, augment=False)
    np.save('val_embeddings.npy', val_embeddings)
    
    print("Embedding untuk train dan val telah disimpan.")

# Panggil fungsi untuk menyimpan embedding
save_embeddings()
