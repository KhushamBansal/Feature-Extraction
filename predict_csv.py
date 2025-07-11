import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

# === Configuration ===
MODEL_PATH = 'models/distraction_classifier.h5'
LABEL_PATH = 'models/label_classes.npy'
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000
CLASSES = ['baby_crying', 'music', 'animal_sound', 'yawning', 'talking']

# === Ensure label_classes.npy exists ===
if not os.path.exists(LABEL_PATH):
    print("📁 label_classes.npy not found. Generating from CLASSES list...")
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    os.makedirs(os.path.dirname(LABEL_PATH), exist_ok=True)
    np.save(LABEL_PATH, label_encoder.classes_)
else:
    print("✅ Found label_classes.npy")

# === Load models and labels ===
print("🔄 Loading trained classifier...")
model = tf.keras.models.load_model(MODEL_PATH)

print("🔄 Loading label encoder classes...")
class_names = np.load(LABEL_PATH, allow_pickle=True)

print("🔄 Loading YAMNet model...")
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)

# === Prediction Function ===
def predict_audio_class(file_path):
    print(f"\n🎧 Predicting: {os.path.basename(file_path)}")

    # Load and preprocess audio
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    waveform = waveform.astype(np.float32)

    # Extract YAMNet embedding
    _, embeddings, _ = yamnet_model(waveform)
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

    # Predict using trained classifier
    probabilities = model.predict(mean_embedding)
    predicted_index = np.argmax(probabilities)
    predicted_label = class_names[predicted_index]
    confidence = probabilities[0][predicted_index]

    print(f"✅ Prediction: {predicted_label} ({confidence:.2%} confidence)")
    return predicted_label, confidence

# === Main ===
if __name__ == "__main__":
    test_file = "dataset/baby-crying-loud-100441.mp3"  # 🔁 Change to your test audio file

    if os.path.exists(test_file):
        predict_audio_class(test_file)
    else:
        print("❌ Please add your .wav file as 'test_audio.wav' or update the path in the script.")
