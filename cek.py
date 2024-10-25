def load_facenet_model(model_path='facenet_keras.h5'):
    try:
        model = load_model(model_path)
        print("Model FaceNet berhasil dimuat")
    except Exception as e:
        print(f"Error memuat model: {e}")
    return model
