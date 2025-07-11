import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import sys

# Load models
print("Loading models...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
classifier = tf.keras.models.load_model('models/distraction_classifier.h5')
classes = np.load('models/label_classes.npy')

def predict_audio(file_path):
    # Load audio
    audio, _ = librosa.load(file_path, sr=16000, mono=True)
    
    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(audio)
    avg_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    
    # Predict
    prediction = classifier.predict(avg_embedding.reshape(1, -1))
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

# Test prediction
if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        predicted_class, confidence = predict_audio(audio_file)
        print(f"\nPredicted: {predicted_class} (Confidence: {confidence:.1f}%)")
    else:
        print("Usage: python src/predict.py <audio_file_path>")