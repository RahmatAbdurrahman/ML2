
# 📒 Klasifikasi Gambar Sampah dengan PyTorch & ResNet18

Proyek ini membangun model klasifikasi gambar sampah berdasarkan dataset dari Kaggle. Model menggunakan arsitektur **ResNet18** dengan PyTorch.

## 📦 1. Instalasi dan Konfigurasi

### Install Library Kaggle
```bash
!pip install kaggle
```
Menginstal pustaka `kaggle` yang dibutuhkan untuk mengunduh dataset dari platform Kaggle.

### Upload `kaggle.json`
```python
from google.colab import files
files.upload()
```
Unggah file `kaggle.json` milikmu untuk otentikasi API Kaggle.

### Pindahkan dan Atur Permission
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Membuat direktori `.kaggle`, menyalin file JSON ke sana, dan mengatur izin akses agar dapat digunakan.

---

## 📥 2. Unduh & Ekstrak Dataset

### Unduh Dataset dari Kaggle
```bash
!kaggle datasets download -d farzadnekouei/trash-type-image-dataset
```

### Ekstrak Dataset
```bash
!unzip trash-type-image-dataset.zip -d sampah
```

---

## 🗂️ 3. Preprocessing & Splitting Dataset

```python
source_dir = '/content/sampah/TrashType_Image_Dataset/'
target_dir = 'dataset'
```

Dataset dibagi menjadi train (70%), validation (15%), dan test (15%).

---

## 🧲 4. Transformasi Gambar

Mengatur resize, augmentasi (flip, jitter, rotate), dan normalisasi.

---

## 📤 5. Dataloader

Menggunakan `ImageFolder` dan `DataLoader` untuk manajemen batch saat training dan evaluasi.

---

## 🧠 6. Bangun Model ResNet18

Menggunakan pretrained model ResNet18 dan mengganti layer akhir sesuai jumlah kelas.

---

## ⚙️ 7. Pelatihan Model

Melatih selama 10 epoch dan mencatat akurasi training dan validasi.

---

## 📈 8. Evaluasi Model

Menggunakan confusion matrix dan classification report.

---

## 💾 9. Simpan Model

Model disimpan dalam file `sampah.pth`.

---

## 🖼️ 10. Prediksi Gambar Baru

Memuat model dan melakukan prediksi pada gambar input dengan visualisasi hasil.

---

## 🔄 11. Finalisasi

### Update Library
```bash
!pip install --upgrade sympy torch torchvision
```

---

## 📚 Dataset

Dataset: [Trash Type Image Dataset on Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)

---

## ✅ Output Akhir
- Model ResNet18 yang dilatih dan disimpan dalam file `sampah.pth`.
- Visualisasi akurasi training/validation.
- Confusion Matrix dan Classification Report.
- Fitur prediksi gambar tunggal dari file.
