import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# Load the pretrained YAMNet model from TensorFlow Hub
print("⏳ Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet loaded successfully.")

# Function to extract a 1024-dim embedding for an audio file
def extract_yamnet_embedding(file_path, target_sr=16000):
    try:
        # Load and resample audio
        waveform, _ = librosa.load(file_path, sr=target_sr)
        waveform = waveform.astype(np.float32)

        # Run the waveform through YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform)

        # Average across time frames to get a single 1024-dim feature
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        return mean_embedding
    except Exception as e:
        print(f"❌ Error extracting from {file_path}: {e}")
        return None

# Main function to extract features from dataset folder
def extract_features_from_dataset(dataset_path):
    features = []
    labels = []
    filenames = []

    for label_name in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_path):
            continue

        print(f"\n📂 Processing class: '{label_name}'")
        for file in tqdm(os.listdir(label_path)):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(label_path, file)
                embedding = extract_yamnet_embedding(file_path)
                if embedding is not None:
                    features.append(embedding)
                    labels.append(label_name)
                    filenames.append(file)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    df['filename'] = filenames
    return df

# Run the script
if __name__ == "__main__":
    dataset_folder = "dataset"  # Your main folder containing class subfolders
    output_csv = "yamnet_features.csv"

    print("🚀 Starting feature extraction...")
    df = extract_features_from_dataset(dataset_folder)

    print(f"\n💾 Saving features to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("✅ All features extracted and saved successfully!")
