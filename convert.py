import os
from PIL import Image

# Tentukan folder utama dataset
main_folder = "dataset2"  # ganti dengan path folder dataset utama kamu
output_main_folder = "dataset-r"  # ganti dengan path folder output

# Pastikan folder output ada, jika tidak, buat foldernya
if not os.path.exists(output_main_folder):
    os.makedirs(output_main_folder)

# Fungsi untuk konversi file PPM ke JPG dalam folder tertentu
def convert_ppm_to_jpg(input_folder, output_folder):
    # Buat folder output jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(input_folder):
        print(f"Memproses folder: {root}")
        print(f"File yang ditemukan: {files}")
        for filename in files:
            if filename.lower().endswith(".ppm"):
                ppm_path = os.path.join(root, filename)
                output_dir = os.path.join(output_folder, os.path.relpath(root, input_folder))  # Menyesuaikan struktur folder output
                
                # Buat folder output jika belum ada
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_filename = os.path.splitext(filename)[0] + ".jpg"
                output_path = os.path.join(output_dir, output_filename)

                try:
                    # Buka file PPM
                    with Image.open(ppm_path) as img:
                        # Simpan file sebagai JPG
                        img.save(output_path, "JPEG")
                        print(f"Konversi selesai: {ppm_path} -> {output_path}")
                except Exception as e:
                    print(f"Gagal membuka file {ppm_path}: {e}")

# Loop untuk memproses setiap folder (train, val, augment, test)
for dataset_type in ['train', 'val', 'augment', 'test']:
    input_folder = os.path.join(main_folder, dataset_type)
    output_folder = os.path.join(output_main_folder, dataset_type)
    
    convert_ppm_to_jpg(input_folder, output_folder)
