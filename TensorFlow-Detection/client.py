import requests
from PIL import Image
import os

# URL endpoint untuk prediksi
predict_url = "http://localhost:5000/predict"  # Sesuaikan dengan alamat server Flask Anda

# Path gambar yang akan diuji
image_path = r"D:\WPy64-31090\scripts\BotolDet\myenv\Image\TestImage\Neg_000002.jpg"

# Menangani karakter khusus dalam path file
image_path = os.path.normpath(image_path)

# Baca gambar dan kirimkan ke server Flask
with open(image_path, "rb") as file:
    files = {"image": (os.path.basename(image_path), file, "image/jpeg")}
    response = requests.post(predict_url, files=files)

# Tampilkan hasil prediksi
if response.status_code == 200:
    result = response.json()
    predicted_class = result["predicted_class"]
    print(f"Predicted class: {predicted_class}")
else:
    print(f"Error: {response.status_code}")
