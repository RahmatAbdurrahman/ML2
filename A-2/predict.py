torch.save(model.state_dict(), "mlp_iris.pth")

# Fungsi prediksi untuk data baru
def predict_new(sample):
    model.eval()
    with torch.no_grad():
        sample_scaled = scaler.transform([sample])  # Normalisasi
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)
        output = model(sample_tensor)
        _, predicted = torch.max(output, 1)
        return encoder.inverse_transform(predicted.numpy())[0]

print(predict_new([5.1, 3.5, 1.4, 0.2]))
