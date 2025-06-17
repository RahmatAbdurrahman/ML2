from sklearn.metrics import ConfusionMatrixDisplay #Menampilkan confusion matrix sebagai visualisasi 
from sklearn.datasets import load_iris  # Dataset Iris
from sklearn.model_selection import train_test_split  # Membagi data train/test
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Normalisasi dan encoding
from sklearn.linear_model import LogisticRegression  # Model baseline
from sklearn.metrics import accuracy_score  # Evaluasi akurasi

# Visualisasi perbandingan akurasi
plt.bar(['Logistic Regression', 'MLP'], [acc_logreg, acc_mlp], color=['skyblue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_logreg, display_labels=iris.target_names)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# MLP
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp.numpy(), display_labels=iris.target_names)
plt.title("Confusion Matrix - MLP")
plt.show()

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Accuracy Logistic Regression: {acc_logreg:.4f}")

model = MLP()
model.load_state_dict(torch.load("model.pth"))  # Path ke model hasil training
model.eval()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred_mlp = torch.max(outputs, 1)

acc_mlp = accuracy_score(y_test, y_pred_mlp.numpy())
print(f"Accuracy MLP: {acc_mlp:.4f}")

print("\n Analisis:")
print(f"- Logistic Regression Accuracy: {acc_logreg:.4f}")
print(f"- MLP Accuracy: {acc_mlp:.4f}")

if acc_mlp > acc_logreg:
    print(" MLP lebih unggul: model mampu menangkap hubungan non-linear dalam data.")
elif acc_mlp < acc_logreg:
    print(" Logistic Regression lebih unggul: kemungkinan karena Iris dataset cukup linear.")
else:
    print(" Kedua model sama kuat: MLP tidak memberikan keunggulan berarti di dataset ini.")

print("\n Insight:")
print("- Logistic Regression lebih cepat, cocok untuk data sederhana.")
print("- MLP cocok untuk data kompleks, tapi butuh waktu lebih lama & tuning parameter.")
