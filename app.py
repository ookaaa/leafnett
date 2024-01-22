import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
import cv2
import os
from fungsi import make_model

# =[Variabel Global]=============================
app = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/upload")
def upload_page():
    return render_template('upload.html')

@app.route("/inner-page")
def inner_page():
    return render_template('inner-page.html')

# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Load model yang telah ditraining
    global model
    if model is None:
        model = make_model()
        model.load_weights('model_grape_tf.h5')

    if request.method == 'POST':
        # Menerima file gambar yang dikirim dari frontend
        file = request.files['file']

        # Simpan file gambar ke direktori temporary
        file_path = 'static/temp/temp.txt'
        file.save(file_path)

        # Membaca dan memproses gambar dengan OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Melakukan prediksi menggunakan model
        prediksi = model.predict(image)
        class_index = np.argmax(prediksi)

        # Daftar kelas
        classes = ['class_0', 'class_1', 'class_2', 'class_3']

        # Mengambil label prediksi
        hasil_prediksi = classes[class_index]

        # Mengubah label prediksi menjadi teks yang lebih deskriptif
        if hasil_prediksi == 'class_0':
            hasil_prediksi = 'Scab'
            keterangan_penyakit = 'Apple Scab (Embun Apel) adalah penyakit jamur yang umum pada buah apel. Gejalanya meliputi adanya bintik-bintik berwarna coklat kehitaman pada daun, buah, dan ranting apel. Bintik-bintik tersebut dapat membesar dan menggabung menjadi kerak-kerak kasar yang dapat mengganggu pertumbuhan dan perkembangan buah. Penyakit ini dapat menyebabkan penurunan kualitas buah dan rendahnya hasil panen jika tidak dikelola dengan baik.'
        elif hasil_prediksi == 'class_1':
            hasil_prediksi = 'Black Rot'
            keterangan_penyakit = 'Black Rot (Busuk Hitam) juga merupakan penyakit jamur yang mempengaruhi buah apel. Gejalanya meliputi bintik-bintik berwarna coklat kehitaman yang muncul pada buah dan daun. Bintik-bintik tersebut kemudian membesar dan menghasilkan area busuk berwarna hitam yang membusuk pada buah. Buah yang terinfeksi Black Rot umumnya menjadi lunak, berkerut, dan tidak layak untuk dikonsumsi. Penyakit ini dapat menyebar melalui spora jamur yang terbawa oleh air hujan atau angin.'
        elif hasil_prediksi == 'class_2':
            hasil_prediksi = 'Cedar Apple Rust'
            keterangan_penyakit = 'Cedar Apple Rust (Karat Apel Cedar) adalah penyakit yang melibatkan dua inang, yaitu tanaman apel dan pohon cedar. Penyakit ini disebabkan oleh jamur yang menghasilkan bintik-bintik berwarna oranye-bisul pada daun dan buah apel. Pada pohon cedar, jamur tersebut menghasilkan gumpalan-gumpalan berwarna coklat kehitaman. Penyakit ini dapat mengurangi produksi buah apel dan menyebabkan kerugian ekonomi pada pertanian apel.'
        else:
            hasil_prediksi = 'Healthy'
            keterangan_penyakit = 'Healthy (Sehat) mengacu pada kondisi daun buah apel yang tidak terinfeksi oleh penyakit atau gangguan lainnya. Daun sehat umumnya memiliki warna hijau cerah, bebas dari bintik-bintik, kerak, atau tanda-tanda kerusakan lainnya. Daun yang sehat sangat penting untuk menjaga kesehatan dan produktivitas pohon apel.'

        # Menghapus file gambar temporary
        os.remove(file_path)

        # Return hasil prediksi dengan format JSON
        return jsonify({
            "prediksi": hasil_prediksi
        })

# =[Main]========================================

if __name__ == '__main__':
    # Run Flask di localhost
    run_with_ngrok(app)
    app.run()
