
# 🐶🐱 Cat vs Dog Image Classifier

Proyek ini bertujuan membangun model klasifikasi gambar untuk membedakan antara kucing dan anjing menggunakan deep learning dengan TensorFlow dan Keras. Notebook ini dimodifikasi untuk menunjukkan pipeline end-to-end dari persiapan data hingga evaluasi model.

## 📁 Struktur Proyek

- `CatDog_Classifier_Modified.ipynb`: Notebook utama yang berisi seluruh proses klasifikasi.
- `dataset/`: Folder berisi dataset gambar anjing dan kucing (tidak disertakan di sini).
- `model/`: Direktori (opsional) untuk menyimpan model yang sudah dilatih.

---

## 📝 Langkah-langkah Proses

### 1. **Import Library**
Notebook dimulai dengan mengimpor library penting seperti:
- TensorFlow dan Keras untuk deep learning.
- NumPy dan Matplotlib untuk manipulasi data dan visualisasi.

### 2. **Pengaturan Direktori Dataset**
Menentukan path untuk folder dataset:
- `train_dir`, `validation_dir`, dan `test_dir` dipetakan ke folder masing-masing.

### 3. **Preprocessing Data**
Menggunakan `ImageDataGenerator` untuk:
- Augmentasi gambar pada dataset pelatihan.
- Normalisasi piksel (rescale ke 1/255).
- Memastikan gambar dikonversi ke ukuran dan format yang sesuai (mis. 150x150 px).

### 4. **Visualisasi Sampel Data**
Menampilkan beberapa gambar dari dataset untuk memastikan data terbaca dengan benar.

### 5. **Pembangunan Model CNN**
Membangun model CNN menggunakan `Sequential API`, biasanya terdiri dari:
- Beberapa layer konvolusi + max pooling.
- Flatten layer untuk mengubah output 2D ke 1D.
- Dense layer untuk klasifikasi.
- Output layer dengan sigmoid (karena klasifikasi biner).

### 6. **Kompilasi Model**
Model dikompilasi menggunakan:
- `binary_crossentropy` untuk loss function.
- `adam` sebagai optimizer.
- `accuracy` sebagai metrik evaluasi.

### 7. **Training Model**
Melatih model dengan `fit()` atau `fit_generator()` selama beberapa epoch.
- Ditampilkan grafik akurasi dan loss untuk training dan validation set.

### 8. **Evaluasi Model**
Model dievaluasi menggunakan dataset validasi atau test.
- Dihitung akurasi dan ditampilkan grafik evaluasi.

### 9. **Prediksi**
Menguji model dengan gambar baru untuk melihat apakah model dapat mengklasifikasikan sebagai kucing atau anjing.

### 10. **Simpan Model**
Model disimpan ke dalam file `.h5` untuk digunakan kembali di masa depan.

---

## 🚀 Cara Menjalankan

1. Pastikan Anda memiliki Python ≥ 3.7, TensorFlow ≥ 2.x, dan Jupyter Notebook.
2. Jalankan Jupyter Notebook dan buka file `CatDog_Classifier_Modified.ipynb`.
3. Ikuti sel-sel kode secara berurutan.
4. Dataset harus sudah tersedia dalam struktur direktori yang tepat.

---

## 📌 Catatan
- Dataset bisa diperoleh dari [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).
- Jika Anda menggunakan Google Colab, pastikan untuk meng-upload dataset ke Google Drive dan mount ke environment.

---

## 🧠 Model yang Digunakan

- CNN dasar tanpa pretrained model (kecuali disebut sebaliknya).
- Untuk hasil lebih baik, bisa eksplorasi dengan transfer learning seperti VGG16, MobileNet, dsb.

---

## 🧪 Evaluasi & Performa

- Akurasi model tergantung pada jumlah epoch, data augmentasi, dan arsitektur model.
- Biasanya mencapai akurasi >85% dengan tuning sederhana.

---

## 📬 Kontak

Jika Anda memiliki pertanyaan atau saran, silakan hubungi [YourName](mailto:your.email@example.com).
