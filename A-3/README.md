# 🧠 MNIST Handwritten Digit Classifier (PyTorch)

Proyek ini membangun dan melatih model Convolutional Neural Network (CNN) untuk mengklasifikasi gambar digit tulisan tangan menggunakan dataset MNIST dengan framework **PyTorch**.

---

## 📦 Dependencies

Install semua library yang dibutuhkan:

```bash
pip install torch torchvision matplotlib tqdm scikit-learn kaggle
```

---

## 📁 Dataset Preparation

1. **Autentikasi Kaggle**
   
   Unggah file `kaggle.json` ke direktori Colab:

   ```python
   from google.colab import files
   files.upload()  # Upload file kaggle.json
   ```

   Atur direktori dan permission:

   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Gunakan file dataset MNIST asli:**

   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`

---

## 🧾 Dataset Loader

Dataset dibaca dari file IDX dan di-transform ke tensor menggunakan `torch.utils.data.Dataset`.

```python
class MNISTDataset(Dataset)
```

- Membaca file binary MNIST (format asli).
- Transformasi yang diterapkan:
  - `RandomRotation(10°)`
  - `RandomAffine(translate=10%)`
  - `ToTensor()`
  - `Normalize(mean=0.1307, std=0.3081)`

---

## 🧠 CNN Arsitektur

Model klasifikasi dibangun menggunakan CNN sederhana:

```python
class MnistClassifier(nn.Module)
```

- Conv2D(1 → 32) → ReLU  
- Conv2D(32 → 64) → ReLU  
- MaxPooling(2x2)  
- Dropout(25%)  
- FC(128) → Dropout → Output(10 kelas)

---

## ⚙️ Training Configuration

- Epochs: `10`  
- Batch size: `16`  
- Optimizer: `Adam`  
- Learning Rate: `1e-4`  
- Loss: `CrossEntropyLoss`

Cek parameter:

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 🏋️ Training Loop

```python
for epoch in range(epochs):
```

- **Training Phase**:
  - Forward → Loss → Backward → Optimizer Step
- **Validation Phase**:
  - No gradient, hanya evaluasi
- **Logging:**
  - Akurasi, rata-rata loss, dan total prediksi disimpan.

---

## 💾 Save Model

Model dan training history disimpan dalam satu file `.pth`:

```python
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'test_loss': test_losses,
}, 'model.pth')
```

---

## 📈 Loss Visualization

Plot kurva perbandingan antara train dan test loss tiap epoch:

```python
plt.plot(epochs_range, train_losses)
plt.plot(epochs_range, test_losses)
```

---

## 📊 Confusion Matrix

Evaluasi model dengan confusion matrix:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

Menampilkan jumlah prediksi benar/salah per kelas dalam bentuk visual.

---

## ✅ Hasil Evaluasi

- Akurasi model akan ditampilkan setiap epoch.
- Grafik dan confusion matrix membantu mengevaluasi kinerja model.
- Model sederhana namun cukup akurat untuk baseline.
