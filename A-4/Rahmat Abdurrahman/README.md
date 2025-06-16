# Hijab vs NonHijab Image Classifier

Proyek ini merupakan implementasi deep learning menggunakan **TensorFlow (Keras)** dan **PyTorch** untuk klasifikasi gambar antara dua kelas: `Hijab` dan `NonHijab`. Dataset digunakan dari repositori GitHub yang berisi 2500+ gambar.

---

## 📁 Clone Dataset

```bash
!git clone https://github.com/mnajamudinridha/naja-dataset.git
```
Dataset akan tersedia di path:
```python
dataset_path = '/content/naja-dataset/dataset-2500'
```

---

## 🧪 Preprocessing & Augmentasi

```python
ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
```
Melakukan normalisasi dan augmentasi (rotasi & flipping) untuk data latih dan validasi.

---

## 📤 Load Dataset dari Directory

```python
train_generator = train_datagen.flow_from_directory(...)
val_generator = val_datagen.flow_from_directory(...)
```
Dataset dibagi menjadi dua folder (`train/` dan `val/`) dan di-load dengan ukuran `(224, 224)` dalam batch 32 gambar.

---

## 🧠 Arsitektur Model

Menggunakan **MobileNetV2** sebagai feature extractor dengan pre-trained weight dari `ImageNet`, kemudian ditambahkan:

- GlobalAveragePooling2D
- Dense(128, relu)
- Dense(1, sigmoid)

---

## ⚙️ Training Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(..., epochs=10)
```
Model dilatih selama 10 epoch menggunakan Adam optimizer.

---

## 📈 Evaluasi Model

Menggunakan Confusion Matrix dan Classification Report:

```python
confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)
```

Visualisasi akurasi dan loss:

```python
plt.plot(history.history['accuracy'], ...)
plt.plot(history.history['loss'], ...)
```

---

## 💾 Simpan Model

```python
model.save('naja-dataset.keras')
```

---

## 🖼️ Prediksi Gambar Tunggal (Keras)

```python
predict_single_image(img_path, model, class_names)
```
Melakukan prediksi pada gambar tunggal dan menampilkan label prediksi serta confidence.

---

## 🔥 Alternatif Inference PyTorch

Menggunakan `resnet18` dan model dari checkpoint `.pth`, dapat digunakan dengan fungsi:

```python
predict_image(image_path, model, device, class_names)
```

---

## 🖼️ Visualisasi Hasil Prediksi

```python
for path in img_path:
    predict_single_image(path, model, class_names)
```

Menampilkan hasil prediksi model pada beberapa gambar sekaligus menggunakan subplot.

---

## 🔍 Visualisasi Prediksi vs Ground Truth

Menampilkan 5 gambar pertama dari validation set dengan label prediksi dan label sebenarnya:

```python
plt.imshow(images[i])
plt.title(f"Pred: {pred_label} | True: {true_label}")
```

---

## 📌 Kelas

```python
class_names = ['Hijab', 'NonHijab']
```

---

## 🧠 Library yang Digunakan

- TensorFlow & Keras
- MobileNetV2 Pretrained Model
- Scikit-learn (Confusion Matrix & Classification Report)
- PyTorch (Opsional untuk model `.pth`)
- Matplotlib & NumPy

---

## 📂 Struktur Dataset

```
dataset-2500/
├── train/
│   ├── Hijab/
│   └── NonHijab/
└── val/
    ├── Hijab/
    └── NonHijab/
```

---

## ✨ Output Model
- Model disimpan sebagai `naja-dataset.keras`
- Visualisasi Confusion Matrix dan akurasi/loss per epoch

---

## 📬 Kontak
Proyek ini dikelola oleh [@mnajamudinridha](https://github.com/mnajamudinridha).